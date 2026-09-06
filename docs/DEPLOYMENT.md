# Deploy Ishaara on Vercel and Render

The repository has two deployable parts:

- **Vercel:** the responsive mobile web app in `review-demo/`
- **Render:** the Flask, MediaPipe, and ONNX recognition API

The frontend sends each short camera recording to the Render service over HTTPS.

## 1. Deploy the backend to Render

1. In Render, choose **New → Blueprint** and connect this GitHub repository.
2. Select the branch containing these changes.
3. Render reads `render.yaml` from the repository root.
4. For `ISHAARA_ALLOWED_ORIGINS`, enter the Vercel origin if known, such as `https://ishaara.vercel.app`. It can be updated after the Vercel project is created.
5. Deploy and copy the service URL, such as `https://ishaara-api.onrender.com`.
6. Open `https://YOUR-RENDER-URL/api/health`. It must return JSON containing `"status": "ready"`.

The Blueprint defines the Python build command, Gunicorn start command, and health check. It intentionally runs one Gunicorn worker because extra workers duplicate the model and MediaPipe memory. If the service runs out of memory, select a larger Render instance.

## 2. Deploy the frontend to Vercel

1. In Vercel, import the same GitHub repository.
2. Leave the repository root as the project root.
3. Add this environment variable:

   ```text
   ISHAARA_API_BASE=https://YOUR-RENDER-URL
   ```

   Do not add a trailing slash. Enable it for Production and Preview if both are used.
4. Deploy. `vercel.json` builds and publishes `review-demo/`.
5. Copy the final Vercel origin.

## 3. Allow the Vercel frontend

In Render, set `ISHAARA_ALLOWED_ORIGINS` to the exact Vercel origin. Multiple origins can be comma-separated:

```text
https://ishaara.vercel.app,https://www.example.com
```

Save the variable and restart or redeploy the Render service.

## 4. Test on a phone

1. Open the HTTPS Vercel URL.
2. Press **Get Started**.
3. Press **Start camera** and allow camera permission.
4. Press **Recognize one sign** and perform `hello` or `thankyou` after the countdown.
5. Tap the speaker button to hear the recognized text.
6. **Use phone camera** remains available as a native record/upload fallback.

HTTPS is required for mobile camera access. A sleeping backend can make the first request slower.

## Local development

Run from the repository root:

```powershell
.\INCLUDE\.venv\Scripts\python.exe local_recognition_server.py
```

Then open `http://127.0.0.1:4173/`. The checked-in `review-demo/config.js` uses same-origin API calls locally.

## Important limitations

- Recognition clips are temporary and deleted after inference.
- Cloud sample collection is not durable without persistent storage.
- The deployed model currently recognizes only its tested `hello` and `thankyou` labels. It is not unrestricted ISL translation.
