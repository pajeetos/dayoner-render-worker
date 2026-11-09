# dayoner-render-worker

RunPod Serverless worker that renders Dayoner's 3D voxel cartoons: Blender 5.2 on the GPU (Eevee, falling back to
Cycles where the container has no EGL), Whisper word timings for lip sync, ffmpeg assembly, and an upload to Vercel
Blob with a one-file token from the job.

The image holds tools only. The render scripts arrive with every job (`input.scripts`), so this repository contains
no product code and no keys. See `handler.py` for the job format; `{"input": {"kind": "probe"}}` checks the GPU and
both render engines without uploading anything.

The image is built by GitHub Actions on every push to `main`: `ghcr.io/pajeetos/dayoner-render-worker:latest`.
