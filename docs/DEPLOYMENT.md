# Deploying the Phone MVP

Ishaara is not a static website. The browser interface calls a Python service that performs MediaPipe landmark extraction and ONNX inference. Deploy the repository as a **Python web service**, not as static hosting.

## Service configuration

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT local_recognition_server:app`
- Health-check path: `/api/health`
- Recommended memory: at least 1 GB for MediaPipe and video processing
- HTTPS: required for live mobile `getUserMedia` camera access

The included `Procfile` supplies the start command on platforms that support it. `local_recognition_server.py` also reads the standard `PORT` environment variable automatically.

## What is included

- The quantized two-label ONNX model
- Model labels and aggregate evaluation metrics
- Self-contained inference architecture and feature extraction
- The phone interface and native text-to-speech controls

The ignored `INCLUDE` research checkout is not needed by the deployed recognition service.

## Mobile test

1. Wait until `https://<your-domain>/api/health` returns `{"status":"ready", ...}`.
2. Open the HTTPS site on the phone.
3. Press **Start camera** and allow camera permission.
4. Press **Recognize one sign**.
5. Perform `hello` or `thankyou` after the countdown.
6. Confirm that an accepted label is displayed and spoken automatically.
7. Press the speaker button to replay it.

The **Use phone camera** button remains available as a native record/upload fallback.

## Storage warning

Recognition clips use temporary files and are deleted after inference. The **Collect samples** feature writes consented recordings to `data/isl-field-captures`; most cloud services use ephemeral filesystems, so do not rely on cloud collection unless a private persistent disk is configured.

## Deployment limitations

- A static-only host cannot execute this model backend.
- Free services may sleep, causing a slow first request.
- One web worker is intentional: multiple workers duplicate the model and MediaPipe memory.
- Do not describe the two-label checkpoint as unrestricted ISL translation.
