# Connect the working ESP32-CAM to Ishaara

The deployed recognition service accepts the ESP32 camera's JPEG frame buffer directly. This does not replace the existing phone-camera path or the existing multipart `/api/frames` contract.

## Data contract

- API: `https://ishaara-api.onrender.com/api/frames/raw`
- Method: `POST`
- Body: one JPEG (`camera_fb_t::buf`)
- Header: `Content-Type: image/jpeg`
- Query parameter `device_id`: the ID entered in the phone app, for example `ishaara-01`
- Query parameter `sequence_id`: one new ID for each complete isolated sign
- Query parameter `final`: `false` for the first frames and `true` for the last frame
- Capture at least 16 frames. The recommended MVP burst is 20 frames, 100 ms apart (approximately 10 fps for two seconds).

The response to non-final frames has HTTP status 202 and `status: collecting`. The final response contains either `status: recognized` and candidates, or `status: no_confident_match`. The phone polls the paired device result and speaks a newly finalized recognition automatically.

## Add this transport to the existing camera sketch

Keep the camera initialization and live-feed code that already works. Add the includes, constants, and functions below. Replace the Wi-Fi credentials if they are not already defined in the sketch.

```cpp
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char *ISHAARA_API = "https://ishaara-api.onrender.com/api/frames/raw";
const char *ISHAARA_DEVICE_ID = "ishaara-01";

bool postIshaaraFrame(camera_fb_t *frame, const String &sequenceId, bool finalFrame) {
  WiFiClientSecure secureClient;
  secureClient.setInsecure(); // MVP only; install the Render CA certificate for production.

  String url = String(ISHAARA_API)
    + "?device_id=" + ISHAARA_DEVICE_ID
    + "&sequence_id=" + sequenceId
    + "&final=" + (finalFrame ? "true" : "false");

  HTTPClient http;
  http.setTimeout(30000);
  if (!http.begin(secureClient, url)) return false;
  http.addHeader("Content-Type", "image/jpeg");
  int status = http.POST(frame->buf, frame->len);
  String response = http.getString();
  Serial.printf("Ishaara frame: HTTP %d %s\n", status, response.c_str());
  http.end();
  return status == 200 || status == 202;
}

bool captureOneIshaaraSign() {
  if (WiFi.status() != WL_CONNECTED) return false;
  String sequenceId = String(ISHAARA_DEVICE_ID) + "-" + String(millis());

  for (int index = 0; index < 20; index++) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) return false;
    bool sent = postIshaaraFrame(frame, sequenceId, index == 19);
    esp_camera_fb_return(frame);
    if (!sent) return false;
    if (index < 19) delay(100);
  }
  return true;
}
```

Call `captureOneIshaaraSign()` from a button handler or a non-blocking application state after the signer is ready. Do not invoke HTTPS or camera capture directly inside a GPIO interrupt service routine; the ISR should only set a volatile flag, and `loop()` should consume that flag.

Example:

```cpp
volatile bool signRequested = false;

void IRAM_ATTR onSignButton() {
  signRequested = true;
}

void loop() {
  // Keep the existing live-feed server handling here.
  if (signRequested) {
    signRequested = false;
    captureOneIshaaraSign();
  }
}
```

If the current live-stream client owns camera buffers continuously, pause that stream during the two-second recognition burst and resume it afterward. Only one component should own a `camera_fb_t` buffer at a time.

## Pair it with the phone

1. Deploy the latest `codex/work` branch to Render and Vercel.
2. Open Ishaara on the phone.
3. Open **Settings → ESP32-CAM**.
4. Enter `ishaara-01`, exactly matching `ISHAARA_DEVICE_ID` in the sketch.
5. Tap **Connect ESP32 camera**.
6. Trigger `captureOneIshaaraSign()` on the ESP32.
7. The phone receives the finalized result, displays it, and speaks it in the selected language.

For the prototype, the ESP32 and phone do not need to be on the same Wi-Fi network because both communicate through Render. They both need internet access. A production build should authenticate devices and validate TLS certificates rather than using `setInsecure()`.
