#include <Arduino.h>
#include <cstdlib>
#include <cstring>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "esp_camera.h"
#include "ishaara_cloud.h"
#include "secrets.h"

namespace {
const char *API_BASE = ISHAARA_API_BASE;
constexpr unsigned long PREVIEW_INTERVAL_MS = 750;
constexpr int RECOGNITION_FRAMES = 20;
constexpr int RECOGNITION_FRAME_INTERVAL_MS = 100;
unsigned long lastPreviewAt = 0;

struct BufferedFrame {
  uint8_t *data = nullptr;
  size_t length = 0;
};

int postJpeg(const String &url, uint8_t *data, size_t length, String *responseBody = nullptr) {
  WiFiClientSecure client;
  client.setInsecure();  // Prototype only. Pin Render's CA certificate before production.
  HTTPClient http;
  http.setConnectTimeout(10000);
  http.setTimeout(30000);
  if (!http.begin(client, url)) return -1;
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Ishaara-Device-Key", ISHAARA_DEVICE_KEY);
  int status = http.POST(data, length);
  if (responseBody != nullptr) *responseBody = http.getString();
  http.end();
  return status;
}

bool captureRecognitionBurst() {
  ishaaraCloudBusy = true;
  String sequenceId = String(ISHAARA_DEVICE_ID) + "-" + String(millis());
  BufferedFrame frames[RECOGNITION_FRAMES];
  bool completed = true;

  // Capture the temporal sequence first. Network uploads happen afterward so
  // TLS latency cannot stretch a two-second sign into a much longer sequence.
  for (int index = 0; index < RECOGNITION_FRAMES; index++) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      completed = false;
      break;
    }
    frames[index].data = static_cast<uint8_t *>(ps_malloc(frame->len));
    frames[index].length = frame->len;
    if (frames[index].data != nullptr) memcpy(frames[index].data, frame->buf, frame->len);
    else completed = false;
    esp_camera_fb_return(frame);
    if (!completed) break;
    if (index < RECOGNITION_FRAMES - 1) delay(RECOGNITION_FRAME_INTERVAL_MS);
  }

  for (int index = 0; completed && index < RECOGNITION_FRAMES; index++) {
    String url = String(API_BASE) + "/api/frames/raw?device_id=" + ISHAARA_DEVICE_ID
      + "&sequence_id=" + sequenceId
      + "&final=" + (index == RECOGNITION_FRAMES - 1 ? "true" : "false");
    String response;
    int status = postJpeg(url, frames[index].data, frames[index].length, &response);
    Serial.printf("Ishaara recognition frame %d/%d: HTTP %d\n", index + 1, RECOGNITION_FRAMES, status);
    if (index == RECOGNITION_FRAMES - 1) Serial.println(response);
    if (status != 200 && status != 202) {
      completed = false;
      break;
    }
  }

  for (int index = 0; index < RECOGNITION_FRAMES; index++) free(frames[index].data);

  ishaaraCloudBusy = false;
  return completed;
}
}  // namespace

volatile bool ishaaraCloudBusy = false;

void startIshaaraCloud() {
  Serial.printf("Ishaara cloud device: %s\n", ISHAARA_DEVICE_ID);
  Serial.println("Ishaara preview starts when no local stream is open.");
  Serial.println("Ishaara cloud bridge started");
}

void serviceIshaaraCloud() {
  if (WiFi.status() != WL_CONNECTED || ishaaraStreamActive || ishaaraCloudBusy) return;
  unsigned long now = millis();
  if (now - lastPreviewAt < PREVIEW_INTERVAL_MS) return;
  lastPreviewAt = now;

  camera_fb_t *frame = esp_camera_fb_get();
  if (frame == nullptr) {
    Serial.println("Ishaara preview capture failed");
    return;
  }
  String response;
  String url = String(API_BASE) + "/api/devices/" + ISHAARA_DEVICE_ID + "/preview";
  int status = postJpeg(url, frame->buf, frame->len, &response);
  esp_camera_fb_return(frame);
  if (status != 200) {
    Serial.printf("Ishaara preview upload failed: HTTP %d\n", status);
    return;
  }
  if (response.indexOf("\"capture_requested\":true") >= 0) {
    Serial.println("Ishaara recognition requested");
    captureRecognitionBurst();
  }
}
