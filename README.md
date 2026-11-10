# dayoner-render-worker

RunPod Serverless worker that renders Dayoner's 3D voxel cartoons: Blender 5.2 on the GPU (Eevee, falling back to
Cycles where the container has no EGL), Whisper word timings for lip sync, ffmpeg assembly, and an upload to Vercel
Blob with a one-file token from the job.

The image holds tools only. The render scripts arrive with every job (`input.scripts`), so this repository contains
no product code and no keys. See `handler.py` for the job format; `{"input": {"kind": "probe"}}` checks the GPU and
both render engines without uploading anything.

The image is built by GitHub Actions on every push to `main`: `ghcr.io/pajeetos/dayoner-render-worker:latest`.

## One scene on two GPUs

A job may carry `split`: the lead makes the timing, faces and models, shares them through a one-file Blob token,
renders the first half and assembles; a helper job renders the second half and shares its frames. They read each
other's progress from a link the app signs. A helper that is late, failed or never got a GPU costs nothing but its
own time: the lead renders that half itself. See `handler.py` for the format.
