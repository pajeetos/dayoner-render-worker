"""RunPod Serverless handler: renders one 3D voxel cartoon.

The image carries the tools (Blender, ffmpeg, Whisper, Node); the render scripts come with every job, so the
engine is versioned with the app that sends it and this public image holds no product code.

Job input:
  kind      "cartoon" (default) or "probe" (GPU, Blender and engine check, no upload)
  scripts   {"faces.py": "...", "plan.py": "...", "scene.py": "...", "assemble.py": "...", "fonts.py": "..."}
  dialogue  the run's dialogue.json
  voices    one https link per line, in order (mp3 or wav)
  models    {"name": "https link to a picture"}: props to make as 3D models first (scripts["models.py"], TripoSR);
            a model that fails leaves the scene to its fallback boxes
  upload    {"pathname": "...", "token": "..."}: a one-file Vercel Blob token for the result
  engine    "eevee" (default) or "cycles"; Eevee falls back to Cycles when it cannot start
  split     optional, one scene on two GPUs:
            {"role": "lead" | "helper", "api": "https link that returns the handoff state",
             "files": {"<mark>": {"pathname": "...", "token": "..."}}}
            The lead makes the timing, faces and models, shares them (files.prep), renders the first half, then
            takes the helper's half (its frames) or renders it itself when the helper is late, failed or never
            started; then it tells the helper to stop (files.take) and assembles the video as usual.
            The helper says it is alive (files.start), waits for the prep, says it started rendering (files.go),
            renders the second half and shares the frames (files.frames); on any failure it says so (files.fail).
            A helper job needs only scripts, split and engine.
            The api answers {"prep": url|null, "frames": url|null, "started", "failed", "taken": bool,
            "renderingFor": seconds since files.go or null}.

Output: {"url", "bytes", "durationSec", "renderSec", "totalSec", "frames", "fps", "beats" ([first, last] frame of
         every beat), "engine", "gpu", "models", "modelSec", "split" (lead only)}; a helper returns {"helper": true, "frames", "engine", "gpu", "renderSec", "totalSec"}
         or {"skipped": "why"}.
"""
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

import runpod

BLENDER = os.environ.get("BLENDER_BIN", "/opt/blender/blender")
SCRIPT_NAME = re.compile(r"^[a-z_]+\.py$")
MAX_VOICE = 20 * 1024 * 1024
MAX_SHARE = 1024 * 1024 * 1024
MAX_LINES = 60
MAX_MODELS = 6
MODEL_NAME = re.compile(r"^[a-z0-9_]{1,16}$")
REQUIRED = ("faces.py", "plan.py", "scene.py", "assemble.py")
UA = {"User-Agent": "dayoner-render-worker"}
# Splitting pays only when each half is worth more than a second Blender start (~25 s).
MIN_SPLIT_FRAMES = 96
POLL_SEC = 3
PREP_WAIT_SEC = 8 * 60
HELPER_MARKS = ("start", "go", "fail", "frames")
LEAD_MARKS = ("prep", "take")


class JobError(Exception):
    pass


class Stopped(JobError):
    """The helper was told to stop: the lead renders its half."""


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def fetch(url, dest, limit=MAX_VOICE):
    if not isinstance(url, str) or not url.startswith("https://"):
        raise JobError(f"bad link: {str(url)[:80]}")
    total = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise JobError("file too big")
            f.write(chunk)


def run(cmd, cwd, env, timeout, on_line=None):
    """Runs a step, streaming its output; keeps the tail for the error message. on_line may return "stop"."""
    tail = []
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    started = time.time()
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        if on_line and on_line(line) == "stop":
            p.kill()
            p.wait()
            raise Stopped("stopped")
        if time.time() - started > timeout:
            p.kill()
            raise JobError(f"{os.path.basename(cmd[0])} timed out")
    if p.wait() != 0:
        raise JobError(f"{os.path.basename(cmd[0])} {' '.join(cmd[1:3])} failed: " + " | ".join(tail[-12:]))


def job_env():
    # the CPUs this container may use (a host can report 256), capped: more threads only fight each other
    threads = str(max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4)))
    return {**os.environ, "CARTOON_NO_BLEND": "1", "CARTOON_CPU_THREADS": threads, "CARTOON_FFMPEG_THREADS": threads,
            "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}


def write_scripts(scripts, eng):
    if not all(k in scripts for k in REQUIRED) or not all(SCRIPT_NAME.match(k) for k in scripts):
        raise JobError("scripts missing")
    os.makedirs(eng)
    for name, body in scripts.items():
        with open(f"{eng}/{name}", "w", encoding="utf-8") as f:
            f.write(body)


def frame_path(run_dir, n):
    return f"{run_dir}/frames/{n:04d}.jpg"


def render_once(job, run_dir, eng, env, engine, first, last, label, should_stop):
    count = last - first + 1
    st = {"n": 0, "first": None, "last": None}
    t0 = time.time()

    def on_line(line):
        if "Saved:" in line:                # Blender 5 prefixes its log lines with a time
            now = time.time()
            st["n"] += 1
            st["first"] = st["first"] or now
            st["last"] = now
            if st["n"] % 24 == 0:
                runpod.serverless.progress_update(job, f"render {label}{st['n']}/{count}")
                if should_stop and should_stop():
                    return "stop"
        return None

    # -t: Blender would otherwise start a thread per reported CPU (256 on some hosts) and crawl
    run([BLENDER, "-b", "--factory-startup", "-t", env.get("CARTOON_CPU_THREADS", "8"), "-P", f"{eng}/scene.py", "--", run_dir,
         "--start", str(first), "--end", str(last)], eng, {**env, "CARTOON_ENGINE": engine}, 3 * 3600, on_line)
    missing = [n for n in range(first, last + 1) if not os.path.exists(frame_path(run_dir, n))]
    if missing:
        raise JobError(f"{engine} rendered {count - len(missing)} of {count} frames")
    now = time.time()
    per = (st["last"] - st["first"]) / (st["n"] - 1) if st["n"] > 1 else (now - t0) / max(1, count)
    return {"sec": now - t0, "perFrame": per, "startup": (st["first"] or now) - t0}


def render(job, run_dir, eng, env, engine, first, last, label="", should_stop=None):
    """Frames first..last. Eevee needs EGL from the host driver; where it has none (or is too slow, scene.py
    stops it), Cycles on the GPU renders the same frames."""
    try:
        return {**render_once(job, run_dir, eng, env, engine, first, last, label, should_stop), "engine": engine}
    except Stopped:
        raise
    except JobError as err:
        if engine != "eevee":
            raise
        print(f"eevee failed, falling back to cycles: {err}", flush=True)
        for n in range(first, last + 1):
            if os.path.exists(frame_path(run_dir, n)):
                os.remove(frame_path(run_dir, n))
        return {**render_once(job, run_dir, eng, env, "cycles", first, last, label, should_stop), "engine": "cycles"}


def probe(job):
    """GPU, Blender and engine check: one tiny frame in Eevee and in Cycles."""
    out = {"gpu": gpu_name()}
    out["blender"] = subprocess.run([BLENDER, "-b", "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[:1]
    with tempfile.TemporaryDirectory() as tmp:
        for engine in ("BLENDER_EEVEE", "CYCLES"):
            py = (
                "import bpy\n"
                "s=bpy.context.scene\n"
                f"s.render.engine='{engine}'\n"
                "if s.render.engine=='CYCLES':\n"
                "  p=bpy.context.preferences.addons['cycles'].preferences\n"
                "  p.compute_device_type='OPTIX'; p.get_devices(); [setattr(d,'use',True) for d in p.devices]; s.cycles.device='GPU'; s.cycles.samples=4\n"
                "s.render.resolution_x=s.render.resolution_y=64\n"
                f"s.render.filepath='{tmp}/{engine}.png'\n"
                "bpy.ops.render.render(write_still=True)\n"
            )
            t = time.time()
            r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-expr", py], capture_output=True, text=True, timeout=300)
            ok = os.path.exists(f"{tmp}/{engine}.png")
            out[engine] = {"ok": ok, "sec": round(time.time() - t, 1), "log": "" if ok else (r.stdout + r.stderr)[-1500:]}
    return out


# ---------------- one scene on two GPUs ----------------

def share(path, spec, content_type):
    """Uploads a file with the one-file token the job brought; returns {"url", "bytes"}."""
    if not isinstance(spec, dict) or not spec.get("pathname") or not spec.get("token"):
        raise JobError("no upload token")
    up = subprocess.run(["node", "/app/upload.mjs", path, spec["pathname"], content_type], capture_output=True, text=True,
                        timeout=900, env={**os.environ, "BLOB_TOKEN": spec["token"]})
    if up.returncode != 0:
        raise JobError("upload failed: " + up.stderr[-400:])
    return json.loads(up.stdout.strip().splitlines()[-1])


def mark(tmp, spec, name):
    path = f"{tmp}/{name}.txt"
    with open(path, "w") as f:
        f.write(name)
    return share(path, spec, "text/plain")


def split_state(api):
    try:
        with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=20) as r:
            st = json.loads(r.read(65536))
        return st if isinstance(st, dict) else {}
    except Exception as err:  # noqa: BLE001
        print(f"split state unreadable: {str(err)[:200]}", flush=True)
        return {}


def pack(src, out, skip=()):
    with tarfile.open(out, "w") as t:
        for name in sorted(os.listdir(src)):
            if name not in skip:
                t.add(f"{src}/{name}", arcname=name)


def unpack(path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path) as t:
        members = t.getmembers()
        for m in members:
            if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                raise JobError(f"bad entry in the shared files: {m.name[:80]}")
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, members=members, filter="data")
        else:
            t.extractall(dest, members=members)  # every entry checked above


def split_point(frames):
    return (frames + 1) // 2


def take_helper_frames(tmp, run_dir, url, first, last, engine):
    """The helper's half into frames/, when it is complete and rendered the same way as ours."""
    try:
        path = f"{tmp}/helper.tar"
        fetch(url, path, MAX_SHARE)
        got = f"{tmp}/helper"
        unpack(path, got)
        with open(f"{got}/helper.json") as f:
            meta = json.load(f)
        if meta.get("engine") != engine:
            print(f"helper rendered in {meta.get('engine')}, we in {engine}: not mixing them", flush=True)
            return False
        if any(not os.path.exists(f"{got}/{n:04d}.jpg") for n in range(first, last + 1)):
            print("helper frames incomplete", flush=True)
            return False
        for n in range(first, last + 1):
            shutil.move(f"{got}/{n:04d}.jpg", frame_path(run_dir, n))
        return True
    except Exception as err:  # noqa: BLE001
        print(f"helper frames unusable: {str(err)[:300]}", flush=True)
        return False


def lead_render(job, tmp, run_dir, eng, env, engine, frames, split):
    """Our half, then the helper's (or ours again when waiting for it would take longer). Returns the engine and a
    summary for the output."""
    files = split.get("files") or {}
    api = split.get("api")
    cut = split_point(frames)
    out = {"cut": cut, "shared": False, "helper": False, "taken": False, "waitSec": 0.0}
    try:
        bundle = f"{tmp}/prep.tar"
        pack(run_dir, bundle, skip=("voice", "frames"))
        share(bundle, files.get("prep"), "application/x-tar")
        out["shared"] = True
    except Exception as err:  # noqa: BLE001
        print(f"prep not shared, rendering alone: {str(err)[:300]}", flush=True)
        r = render(job, run_dir, eng, env, engine, 1, frames)
        return r["engine"], out

    own = render(job, run_dir, eng, env, engine, 1, cut, "1/2 ")
    engine = own["engine"]
    rest = frames - cut
    alone = rest * own["perFrame"] + own["startup"]  # rendering the second half ourselves, from now
    waited = time.time()
    while True:
        st = split_state(api)
        if st.get("frames"):
            if take_helper_frames(tmp, run_dir, st["frames"], cut + 1, frames, engine):
                out.update(helper=True, waitSec=round(time.time() - waited, 1))
                return engine, out
            break
        if not st.get("started") or st.get("failed") or st.get("taken"):
            break
        going = st.get("renderingFor")
        if not isinstance(going, (int, float)):
            break  # alive but not rendering yet: it is a whole half behind us
        helper_left = max(0.0, rest * own["perFrame"] + own["startup"] - float(going)) + 20  # + the frames' trip
        if helper_left > alone or time.time() - waited > alone + 60:
            break
        runpod.serverless.progress_update(job, f"render 2/2 by the helper, ~{int(helper_left)} s")
        time.sleep(POLL_SEC)
    out.update(taken=True, waitSec=round(time.time() - waited, 1))
    try:
        mark(tmp, files.get("take"), "take")
    except Exception as err:  # noqa: BLE001
        print(f"take mark failed: {str(err)[:200]}", flush=True)
    render(job, run_dir, eng, env, engine, cut + 1, frames, "2/2 ")
    return engine, out


def helper(job):
    inp = job["input"]
    split = inp.get("split") or {}
    files = split.get("files") or {}
    api = split.get("api")
    if not isinstance(api, str) or not api.startswith("https://") or not all(k in files for k in HELPER_MARKS):
        raise JobError("split: api and files missing")
    engine = inp.get("engine", "eevee")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        eng, run_dir = f"{tmp}/engine", f"{tmp}/run"
        write_scripts(inp.get("scripts") or {}, eng)
        try:
            mark(tmp, files["start"], "start")
        except Exception as err:  # noqa: BLE001
            return {"skipped": f"could not say it started: {str(err)[:200]}"}
        try:
            prep, deadline = None, time.time() + PREP_WAIT_SEC
            while time.time() < deadline:
                st = split_state(api)
                if st.get("taken"):
                    return {"skipped": "the lead rendered it all"}
                if st.get("prep"):
                    prep = st["prep"]
                    break
                runpod.serverless.progress_update(job, "waiting for the lead")
                time.sleep(POLL_SEC)
            if not prep:
                raise JobError("the lead shared nothing in 8 minutes")
            fetch(prep, f"{tmp}/prep.tar", MAX_SHARE)
            unpack(f"{tmp}/prep.tar", run_dir)
            with open(f"{run_dir}/timeline.json") as f:
                frames = int(json.load(f)["frames"])
            first = split_point(frames) + 1
            env = job_env()
            mark(tmp, files["go"], "go")

            def taken():
                return bool(split_state(api).get("taken"))

            r = render(job, run_dir, eng, env, engine, first, frames, "2/2 ", taken)
            if taken():
                return {"skipped": "the lead rendered it all"}
            share_dir = f"{tmp}/share"
            os.makedirs(share_dir)
            for n in range(first, frames + 1):
                shutil.move(frame_path(run_dir, n), f"{share_dir}/{n:04d}.jpg")
            with open(f"{share_dir}/helper.json", "w") as f:
                json.dump({"engine": r["engine"], "first": first, "last": frames}, f)
            pack(share_dir, f"{tmp}/frames.tar")
            share(f"{tmp}/frames.tar", files["frames"], "application/x-tar")
            return {"helper": True, "frames": frames - first + 1, "engine": r["engine"], "gpu": gpu_name(),
                    "renderSec": round(r["sec"], 1), "totalSec": round(time.time() - t0, 1)}
        except Stopped:
            return {"skipped": "the lead rendered it all"}
        except Exception:
            try:
                mark(tmp, files["fail"], "fail")
            except Exception:  # noqa: BLE001
                pass
            raise


def cartoon(job):
    inp = job["input"]
    scripts = inp.get("scripts") or {}
    voices = inp.get("voices") or []
    dialogue = inp.get("dialogue") or {}
    upload = inp.get("upload") or {}
    # One voice per spoken line: the two-character format lists "lines", the scene format "beats" with "say".
    spoken = [b for b in dialogue.get("beats") or [] if isinstance(b, dict) and b.get("say")] if dialogue.get("beats") else dialogue.get("lines") or []
    if not all(k in scripts for k in REQUIRED) or not all(SCRIPT_NAME.match(k) for k in scripts):
        raise JobError("scripts missing")
    if len(spoken) > MAX_LINES or len(voices) != len(spoken):
        raise JobError("one voice per spoken line, up to 60 lines")
    if not upload.get("pathname") or not upload.get("token"):
        raise JobError("no upload token")
    models = inp.get("models") or {}
    if not isinstance(models, dict) or len(models) > MAX_MODELS or not all(MODEL_NAME.match(k) for k in models):
        raise JobError("models: up to 6, names a-z0-9_")
    split = inp.get("split") if isinstance(inp.get("split"), dict) and inp["split"].get("role") == "lead" else None
    engine = inp.get("engine", "eevee")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        eng, run_dir = f"{tmp}/engine", f"{tmp}/run"
        write_scripts(scripts, eng)
        os.makedirs(f"{run_dir}/voice")
        with open(f"{run_dir}/dialogue.json", "w", encoding="utf-8") as f:
            json.dump(dialogue, f, ensure_ascii=False)
        for i, url in enumerate(voices, 1):
            ext = "wav" if url.split("?")[0].endswith(".wav") else "mp3"
            fetch(url, f"{run_dir}/voice/{i:02d}.{ext}")
        env = job_env()
        threads = env["CARTOON_CPU_THREADS"]

        made, model_sec = 0, 0.0
        if models and "models.py" in scripts:
            # Props made to order: picture -> TripoSR -> voxels. A failure only costs the scene those models.
            t_m = time.time()
            runpod.serverless.progress_update(job, f"3D models ({len(models)})")
            os.makedirs(f"{run_dir}/models")
            try:
                for name, url in models.items():
                    fetch(url, f"{run_dir}/models/{name}.img")
                run(["python", f"{eng}/models.py", run_dir], eng, env, 900, lambda line: print(line, end="", flush=True))
            except JobError as err:
                print(f"models failed: {err}", flush=True)
            made = len([f for f in os.listdir(f"{run_dir}/models") if f.endswith(".json")])
            model_sec = time.time() - t_m

        runpod.serverless.progress_update(job, f"textures and timing ({threads} threads)")
        run(["python", f"{eng}/faces.py", run_dir], eng, env, 300)
        run(["python", f"{eng}/plan.py", run_dir], eng, env, 900)
        with open(f"{run_dir}/timeline.json") as f:
            tl = json.load(f)
        frames = tl["frames"]

        t1 = time.time()
        runpod.serverless.progress_update(job, f"render 0/{frames}")
        split_out = None
        if split and frames >= MIN_SPLIT_FRAMES:
            engine, split_out = lead_render(job, tmp, run_dir, eng, env, engine, frames, split)
        else:
            if split:
                stop_helper(split)  # too short to share: the helper need not wait for us
            engine = render(job, run_dir, eng, env, engine, 1, frames)["engine"]
        render_sec = time.time() - t1

        runpod.serverless.progress_update(job, "assembly")
        out = f"{tmp}/out.mp4"
        run(["python", f"{eng}/assemble.py", run_dir, out], eng, env, 900)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                                   capture_output=True, text=True, timeout=60).stdout.strip() or 0)
        result = share(out, upload, "video/mp4")
    return {**result, "durationSec": round(dur, 2), "renderSec": round(render_sec, 1), "totalSec": round(time.time() - t0, 1),
            "frames": frames, "engine": engine, "gpu": gpu_name(), "models": made, "modelSec": round(model_sec, 1),
            # [first, last] frame of every beat, so a later edit can say what plays at 0:12
            "fps": tl.get("fps"), "beats": tl.get("beats") or [],
            **({"split": split_out} if split_out else {})}


def stop_helper(split):
    """Best effort: tells a helper still waiting for our prep that none is coming."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mark(tmp, (split.get("files") or {}).get("take"), "take")
    except Exception as err:  # noqa: BLE001
        print(f"take mark failed: {str(err)[:200]}", flush=True)


def handler(job):
    inp = job.get("input") or {}
    split = inp.get("split") if isinstance(inp.get("split"), dict) else None
    try:
        if inp.get("kind") == "probe":
            return probe(job)
        if split and split.get("role") == "helper":
            return helper(job)
        return cartoon(job)
    except Exception as err:
        if split and split.get("role") == "lead":
            stop_helper(split)
        if isinstance(err, JobError):
            return {"error": str(err)[:1500]}
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
