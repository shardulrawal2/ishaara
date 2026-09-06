# Connect the ESP32-CAM to Ishaara

Ishaara includes a complete ESP32-CAM firmware sketch in `firmware/IshaaraCameraWebServer/`. The camera sends a low-rate preview outward over HTTPS to Render, so the deployed Vercel app can display it without trying to open a private `192.168.x.x` or `10.x.x.x` address. The phone camera remains available as a fallback.

## How the connection works

1. The ESP32-CAM posts its latest JPEG to `POST /api/devices/<device-id>/preview` about once every 750 ms.
2. The browser fetches `GET /api/devices/<device-id>/preview.jpg` and displays that image in the **Show one sign** panel.
3. Pressing **Recognize one sign** queues a command with `POST /api/devices/<device-id>/capture`.
4. The next preview response tells the ESP32 to capture a sign.
5. The ESP32 captures 20 frames at roughly 10 fps, then uploads them to `/api/frames/raw` as one sequence.
6. Render extracts landmarks, runs the model, and stores the finalized result for the paired browser to display and speak.

This is a cloud-relayed preview, not a 30 fps video stream. It is intentionally lightweight enough for the ESP32-CAM and Render MVP.

## 1. Configure Render

Add an environment variable to the Render service:

```text
ISHAARA_DEVICE_KEY=choose-a-long-random-secret
```

Use the same value in the firmware and in the app's ESP32 settings. Redeploy Render after saving it. If this variable is omitted, the endpoints remain unauthenticated for local development only.

## 2. Configure the firmware

1. Copy `firmware/IshaaraCameraWebServer/secrets.example.h` to `firmware/IshaaraCameraWebServer/secrets.h`.
2. Fill in the Wi-Fi name, Wi-Fi password, device ID, and the same Render device key:

```cpp
#define WIFI_SSID "your-wifi-name"
#define WIFI_PASSWORD "your-wifi-password"
#define ISHAARA_API_BASE "https://ishaara-api.onrender.com"
#define ISHAARA_DEVICE_ID "ishaara-01"
#define ISHAARA_DEVICE_KEY "the-same-random-secret"
```

`secrets.h` is ignored by Git and must never be committed.

3. Open `firmware/IshaaraCameraWebServer/IshaaraCameraWebServer.ino` in Arduino IDE.
4. Select **AI Thinker ESP32-CAM** as the board, enable PSRAM, and use a partition scheme with at least 3 MB application space (usually **Huge APP**).
5. Compile and upload. Use the board's normal GPIO0-to-GND flashing procedure if your programmer requires it, then remove GPIO0 from GND and reset.
6. Open Serial Monitor at 115200 baud. Wait for both the local camera URL and `Ishaara cloud bridge started`.

The supplied sketch uses the same working AI Thinker OV2640 pin map as the original CameraWebServer folder. It starts at VGA, JPEG quality 10, uses PSRAM, and keeps the original local `/capture` and port-81 `/stream` routes.

## 3. Pair the app

1. Deploy the latest branch to Render and Vercel.
2. Open the Vercel app and go to **Settings → ESP32-CAM**.
3. Enter the firmware's exact device ID and device key.
4. Press **Connect ESP32 camera**.
5. Return home and keep **ESP32 camera** selected.

Once the first preview arrives, the actual ESP32 view replaces the camera placeholder. Pressing **Recognize one sign** asks that ESP32—not the phone camera—to record the recognition sequence.

Choose **Phone camera** in the source switch whenever hardware is unavailable.

## Important operational notes

- The ESP32 and phone can be on different networks, but both need internet access.
- Do not leave the ESP32's local `:81/stream` page open while using cloud recognition. A continuous local stream owns camera buffers, so the firmware pauses cloud capture until that stream closes.
- Render's free service may sleep. Open `/api/health` first and wait for it to wake if pairing initially appears offline.
- The preview is relayed as individual JPEGs. Browsers cannot directly embed the ESP32's private HTTP stream inside an HTTPS Vercel page because of private-network reachability and mixed-content security.
- The prototype uses TLS encryption plus a shared device key, but `setInsecure()` does not validate the server certificate. Pin a trusted CA certificate and provision a unique key per device before production.
