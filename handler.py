"""RunPod Serverless handler: renders one 3D voxel cartoon.

The image carries the tools (Blender, ffmpeg, Whisper, Node); the render scripts come with every job, so the
engine is versioned with the app that sends it and this public image holds no product code.

Job input:
  kind      "cartoon" (default) or "probe" (GPU, Blender and engine check, no upload)
  scripts   {"faces.py": "...", "plan.py": "...", "scene.py": "...", "assemble.py": "...", "fonts.py": "..."}
  dialogue  the run's dialogue.json
  voices    one https link per line, in order (mp3 or wav)
  upload    {"pathname": "...", "token": "..."}: a one-file Vercel Blob token for the result
  engine    "eevee" (default) or "cycles"; Eevee falls back to Cycles when it cannot start

Output: {"url", "bytes", "durationSec", "renderSec", "totalSec", "frames", "engine", "gpu"}
"""
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request

import runpod

BLENDER = os.environ.get("BLENDER_BIN", "/opt/blender/blender")
SCRIPT_NAME = re.compile(r"^[a-z_]+\.py$")
MAX_VOICE = 20 * 1024 * 1024
MAX_LINES = 60
REQUIRED = ("faces.py", "plan.py", "scene.py", "assemble.py")


class JobError(Exception):
    pass


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def fetch(url, dest):
    if not isinstance(url, str) or not url.startswith("https://"):
        raise JobError(f"bad voice link: {str(url)[:80]}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dayoner-render-worker"}), timeout=120) as r:
        data = r.read(MAX_VOICE + 1)
    if len(data) > MAX_VOICE:
        raise JobError("voice file too big")
    with open(dest, "wb") as f:
        f.write(data)


def run(cmd, cwd, env, timeout, on_line=None):
    """Runs a step, streaming its output; keeps the tail for the error message."""
    tail = []
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    started = time.time()
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        if on_line:
            on_line(line)
        if time.time() - started > timeout:
            p.kill()
            raise JobError(f"{os.path.basename(cmd[0])} timed out")
    if p.wait() != 0:
        raise JobError(f"{os.path.basename(cmd[0])} {' '.join(cmd[1:3])} failed: " + " | ".join(tail[-12:]))


def render(job, run_dir, eng, env, engine, frames):
    done = {"n": 0}

    def on_line(line):
        if line.startswith("Saved:"):
            done["n"] += 1
            if done["n"] % 24 == 0:
                runpod.serverless.progress_update(job, f"render {done['n']}/{frames}")

    run([BLENDER, "-b", "--factory-startup", "-P", f"{eng}/scene.py", "--", run_dir], eng, {**env, "CARTOON_ENGINE": engine}, 3 * 3600, on_line)
    if len([f for f in os.listdir(f"{run_dir}/frames") if f.endswith(".jpg")]) < frames:
        raise JobError(f"{engine} rendered {done['n']} of {frames} frames")


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


def cartoon(job):
    inp = job["input"]
    scripts = inp.get("scripts") or {}
    voices = inp.get("voices") or []
    dialogue = inp.get("dialogue") or {}
    upload = inp.get("upload") or {}
    lines = dialogue.get("lines") or []
    if not all(k in scripts for k in REQUIRED) or not all(SCRIPT_NAME.match(k) for k in scripts):
        raise JobError("scripts missing")
    if not lines or len(lines) > MAX_LINES or len(voices) != len(lines):
        raise JobError("one voice per line, up to 60 lines")
    if not upload.get("pathname") or not upload.get("token"):
        raise JobError("no upload token")
    engine = inp.get("engine", "eevee")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        eng, run_dir = f"{tmp}/engine", f"{tmp}/run"
        os.makedirs(eng)
        os.makedirs(f"{run_dir}/voice")
        for name, body in scripts.items():
            with open(f"{eng}/{name}", "w", encoding="utf-8") as f:
                f.write(body)
        with open(f"{run_dir}/dialogue.json", "w", encoding="utf-8") as f:
            json.dump(dialogue, f, ensure_ascii=False)
        for i, url in enumerate(voices, 1):
            ext = "wav" if url.split("?")[0].endswith(".wav") else "mp3"
            fetch(url, f"{run_dir}/voice/{i:02d}.{ext}")
        threads = str(os.cpu_count() or 4)
        env = {**os.environ, "CARTOON_NO_BLEND": "1", "CARTOON_CPU_THREADS": threads, "CARTOON_FFMPEG_THREADS": threads}

        runpod.serverless.progress_update(job, "textures and timing")
        run(["python", f"{eng}/faces.py", run_dir], eng, env, 300)
        run(["python", f"{eng}/plan.py", run_dir], eng, env, 900)
        frames = json.load(open(f"{run_dir}/timeline.json"))["frames"]

        t1 = time.time()
        try:
            render(job, run_dir, eng, env, engine, frames)
        except JobError as err:
            if engine != "eevee":
                raise
            # Eevee needs EGL from the host driver; where it has none, Cycles on the GPU renders the same scene.
            print(f"eevee failed, falling back to cycles: {err}", flush=True)
            for f in os.listdir(f"{run_dir}/frames") if os.path.isdir(f"{run_dir}/frames") else []:
                os.remove(f"{run_dir}/frames/{f}")
            engine = "cycles"
            render(job, run_dir, eng, env, engine, frames)
        render_sec = time.time() - t1

        runpod.serverless.progress_update(job, "assembly")
        out = f"{tmp}/out.mp4"
        run(["python", f"{eng}/assemble.py", run_dir, out], eng, env, 900)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                                   capture_output=True, text=True, timeout=60).stdout.strip() or 0)
        up = subprocess.run(["node", "/app/upload.mjs", out, upload["pathname"]], capture_output=True, text=True, timeout=900,
                            env={**os.environ, "BLOB_TOKEN": upload["token"]})
        if up.returncode != 0:
            raise JobError("upload failed: " + up.stderr[-400:])
        result = json.loads(up.stdout.strip().splitlines()[-1])
    return {**result, "durationSec": round(dur, 2), "renderSec": round(render_sec, 1), "totalSec": round(time.time() - t0, 1),
            "frames": frames, "engine": engine, "gpu": gpu_name()}


def handler(job):
    try:
        return probe(job) if (job.get("input") or {}).get("kind") == "probe" else cartoon(job)
    except JobError as err:
        return {"error": str(err)[:1500]}


runpod.serverless.start({"handler": handler})
