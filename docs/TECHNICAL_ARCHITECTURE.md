# Ishaara Technical Architecture

## Document purpose

This document describes the engineering state of the Ishaara repository as implemented on the `codex/work` branch. It separates executable behavior from proposed production architecture. This distinction is essential for a technical review: the repository currently implements isolated-sign temporal classification over mobile video uploads and ordered ESP32-compatible JPEG sequences, but it does not yet implement continuous WebSocket streaming, a 30-frame `deque`, a 1,662-feature MediaPipe Holistic representation, prediction majority voting, or an ESP32 emergency interrupt endpoint.

The current implementation is an ML integration prototype. It proves capture, landmark extraction, tensor preparation, quantized ONNX Runtime inference, explicit uncertainty rejection, local training-data collection, signer-separated dataset preparation, and transfer learning. It does not yet prove production latency, continuous-sign segmentation, unrestricted ISL recognition, or emergency-path latency.

## 1. Implementation status

| Capability | Repository status | Engineering detail |
|---|---|---|
| Mobile camera capture | Implemented | Browser `getUserMedia` plus a 2.5-second `MediaRecorder` clip |
| Mobile WebSocket streaming | Not implemented | Mobile clips are sent as one multipart HTTP request after recording |
| ESP32-CAM ingestion contract | Implemented server-side | Ordered JPEGs accepted at `POST /api/frames` |
| ESP32 firmware | Not present | Physical capture, Wi-Fi transport, retry, and power-management code remain external work |
| MediaPipe extraction | Implemented | Separate MediaPipe Hands and Pose graphs |
| MediaPipe Holistic 1,662-float frame | Not implemented | Current frame schema is 134 floats and excludes face landmarks |
| Temporal sequence classification | Implemented | Fixed sequence of up to 169 frames, classified by a small transformer |
| 30-frame rolling `deque` | Not implemented | Snapshot sequences are accumulated in a Python list and finalized explicitly |
| ONNX Runtime inference | Implemented | Quantized ORT model, CPU execution provider, cached session |
| LSTM inference | Not active | LSTM configuration exists upstream, but the exported runtime model is a transformer |
| Confidence rejection | Implemented | Top probability below 0.75 returns `no_confident_match` |
| Majority-vote smoothing/debouncing | Not implemented | One inference is emitted per finalized clip or snapshot sequence |
| Labelled mobile data collection | Implemented | Consented clips and JSONL manifest remain local and Git-ignored |
| Signer-separated fine-tuning | Implemented as an offline workflow | Dataset preparation and PyTorch transfer-learning scripts are present |
| Refined-model runtime promotion | Not implemented | Fine-tuned checkpoints require validation, export, and explicit runtime wiring |
| Emergency HTTP endpoint | Not implemented | README identifies emergency phrases as planned |
| Hardware interrupt integration | Not implemented | No ESP32 source or GPIO interrupt-service routine exists in this repository |

## 2. Runtime component boundaries

### 2.1 Browser review client

The browser client in `review-demo/app.js` performs four distinct functions:

1. It acquires a user-facing camera with `navigator.mediaDevices.getUserMedia`.
2. It records a short mobile-backup clip with `MediaRecorder` and posts it to `POST /api/mobile/recognize`.
3. It simulates the ESP32 snapshot contract by capturing 20 JPEG images at approximately 100 ms intervals and posting them individually to `POST /api/frames`.
4. It records explicitly labelled local training clips and posts them to `POST /api/training/captures` with a sign label and non-identifying signer code.

The requested camera constraints are 640 × 480 ideal resolution, 24 frames per second ideal, 30 frames per second maximum, front-facing camera, and no audio. Browser and device implementations may negotiate different actual values. The code does not query and persist the negotiated `MediaStreamTrack` settings, so actual capture resolution and frame rate are not currently part of the sample manifest.

### 2.2 Local recognition service

`local_recognition_server.py` is a Flask application bound to `127.0.0.1:4173` for local development. It honors standard deployment `PORT` settings and binds to `0.0.0.0` in that environment. It serves the phone application and ML-facing endpoints:

- `POST /api/recognize`: manually uploaded short video.
- `POST /api/mobile/recognize`: camera-recorded mobile-backup video.
- `POST /api/frames`: ordered JPEG sequence compatible with the intended ESP32-CAM transport.
- `POST /api/training/captures`: explicitly labelled local training recording.

The Flask application has a 50 MB request-body limit. A production Gunicorn command is supplied, but authentication, TLS termination, device registration, API versioning, persistent job queues, and an observability pipeline remain hosting responsibilities.

### 2.3 Landmark processing layer

Video ingestion uses the self-contained `ishaara_runtime.py` preprocessing implementation. Its output was compared against the original INCLUDE path with zero numerical difference. Snapshot ingestion uses `SnapshotSequence` in `photo_stream_recognition.py`, which reproduces the same 134-value landmark schema.

### 2.4 Inference layer

`recognize_refined.py` owns the deployed label metadata, ONNX Runtime session, confidence gate, and top-k decoding. It loads the checked-in 3.68 MB two-label quantized ONNX model with the CPU execution provider. The session is memoized with `lru_cache(maxsize=1)`, preventing model reconstruction for every request.

## 3. Implemented end-to-end data flow

### 3.1 Mobile-video backup ingestion

The mobile backup is clip-oriented rather than stream-oriented.

1. The browser opens the selected camera with `getUserMedia`.
2. When recognition is requested, it selects the first supported recording MIME type from VP8 WebM, generic WebM, and MP4.
3. A `MediaRecorder` runs for 2.5 seconds and requests encoded chunks every 150 ms.
4. The chunks are combined into one browser `Blob` and wrapped as a `File` named `mobile-sign.webm` or `mobile-sign.mp4`.
5. The browser posts that file as multipart form data under the `clip` field to `POST /api/mobile/recognize`.
6. The Flask service writes the upload to a uniquely named temporary file.
7. OpenCV and MediaPipe process the temporary file sequentially.
8. The service runs one sequence inference, serializes either a recognized result or an uncertainty rejection, and deletes the temporary clip in a `finally` block.

This path does not provide asynchronous frame-level server ingestion. Capture completes before network upload begins. Network bandwidth is therefore bursty: the request size is the complete encoded 2.5-second clip. The browser does not specify `videoBitsPerSecond`, so the encoder controls the bitrate. A production bandwidth budget cannot be claimed from the present code; it must be measured from actual `Blob.size`, negotiated resolution, codec, device, and network traces.

### 3.2 ESP32-compatible JPEG ingestion

The intended ESP32-CAM contract is explicit sequence finalization over HTTP:

- Form field `sequence_id`: stable identifier shared by every image belonging to one isolated sign; maximum accepted length is 64 characters.
- Form file `frame`: one JPEG image with MIME type `image/jpeg` or `image/jpg`.
- Form field `final`: string `true` only on the final image.

For a non-final request, the server returns HTTP 202 with collection diagnostics: received-frame count, whether the current frame contained any detected landmark, accumulated pose-frame count, accumulated signer-hand-frame count, and the minimum sequence length.

For a final request, the server requires at least 16 snapshots, prepares a fixed tensor, runs inference, returns the result, removes the sequence from the active dictionary, and closes its MediaPipe graph instances.

At most eight active sequence IDs are allowed. Each sequence records a monotonic last-update time. Sequences idle for more than 120 seconds are closed and removed when a subsequent frame request triggers cleanup. There is no independent cleanup scheduler.

The browser's ESP simulation captures 20 JPEGs at 100 ms spacing, yielding an approximate two-second, 10-fps sequence. The server itself does not enforce a target frame interval, timestamp monotonicity, duplicate detection, sequence ordering number, source identity, or maximum collection length before finalization.

### 3.3 Training-capture ingestion

The local collection endpoint accepts a label, a non-identifying signer code, and a video. Labels are restricted to 1–49 lowercase-normalized characters from letters, digits, spaces, and hyphens. Signer codes are restricted to 1–32 lowercase-normalized letters, digits, and hyphens.

Each accepted video receives a UUID-based filename. The server appends one JSON object to `data/isl-field-captures/manifest.jsonl` containing capture ID, label, signer code, filename, UTC capture time, and source type. The capture directory is excluded from Git. The data is retained because the user explicitly selected the training-save action; this differs from recognition uploads, which are deleted after inference.

## 4. Landmark feature extraction

### 4.1 Actual frame schema

The implemented schema contains 67 two-dimensional landmarks:

- Upper-body pose: first 25 MediaPipe Pose landmarks × `(x, y)` = 50 values.
- First hand: 21 MediaPipe Hand landmarks × `(x, y)` = 42 values.
- Second hand: 21 MediaPipe Hand landmarks × `(x, y)` = 42 values.
- Total: `50 + 42 + 42 = 134` float values per frame.

Coordinates are normalized image coordinates when emitted by MediaPipe. Face mesh coordinates, pose depth, hand depth, visibility, and presence scores are not included in the runtime model input.

### 4.2 Detector configuration

Snapshot processing creates one MediaPipe Hands graph and one MediaPipe Pose graph per active `SnapshotSequence`:

- Hands: temporal tracking mode, maximum two hands, minimum detection confidence 0.5, minimum tracking confidence 0.5.
- Pose: temporal tracking mode, minimum detection confidence 0.5, minimum tracking confidence 0.5.

JPEG bytes are decoded by OpenCV into BGR pixels and converted to RGB before MediaPipe execution.

### 4.3 Missing-landmark representation

Missing landmarks are initially represented as IEEE-754 `NaN`, not zero. This distinction allows the preprocessing stage to differentiate an absent detection from a real coordinate at the image boundary.

The pose and two hand groups are interpolated independently with Pandas linear interpolation using `limit_direction="both"`. Interior gaps are interpolated, and leading/trailing gaps are filled from the nearest available observation. If an entire coordinate group has no finite values, the implementation replaces that group with zeros. After interpolation, normalized x coordinates are multiplied by 1920 and normalized y coordinates by 1080 to reproduce INCLUDE's training-time scale.

There is an edge case in the current snapshot implementation: if a group contains some finite landmarks but an individual landmark column is missing for the entire sequence, that column can remain `NaN`. The video path verifies that its final tensor is finite; the snapshot path does not currently perform the equivalent final finite-value assertion. This should be corrected before production promotion.

### 4.4 Signer-hand association

The hand and pose detectors are independent, so the implementation applies a spatial association gate. When shoulders and wrists are finite, it calculates shoulder width and the Euclidean distance from the hand's wrist-origin landmark to each pose wrist. A hand is retained only when the nearest-wrist distance is at most the larger of 0.08 normalized units or 1.5 times shoulder width.

For a frame containing only one detected hand, the implementation compares that hand's origin with pose landmarks 15 and 16. It swaps first-hand and second-hand storage when necessary to reproduce the INCLUDE checkpoint's wrist-relative convention. When both hands or neither hand are detected, their MediaPipe result order is retained.

This logic reduces contamination from unrelated hands but is not persistent multi-person tracking. It assumes one primary signer and contains no person ID, tracking filter, or face/torso identity association.

### 4.5 Temporal tensor construction

The current model's fixed input shape is `(1, 169, 134)`:

- Axis 0: batch size of one.
- Axis 1: 169 temporal positions.
- Axis 2: 134 landmark features per position.

Sequences shorter than 169 frames are padded with all-zero frames at the end. Sequences longer than 169 are rejected rather than sampled or truncated. Snapshot sequences must contain at least 16 frames.

The numeric payload of one fully populated float32 tensor is approximately `1 × 169 × 134 × 4 = 90,584` bytes. This excludes NumPy container overhead, Pandas interpolation allocations, decoded image buffers, and MediaPipe graph memory.

## 5. Why the requested 1,662-feature contract is different

A conventional MediaPipe Holistic vector of 1,662 values is commonly constructed as:

- Pose: 33 landmarks × `(x, y, z, visibility)` = 132.
- Face: 468 landmarks × `(x, y, z)` = 1,404.
- Left hand: 21 landmarks × `(x, y, z)` = 63.
- Right hand: 21 landmarks × `(x, y, z)` = 63.
- Total: `132 + 1,404 + 63 + 63 = 1,662`.

That schema is not interchangeable with the implemented 134-feature checkpoint. Changing the input from `(169, 134)` to `(30, 1662)` changes both temporal resolution and feature semantics. It requires a new training dataset, preprocessing contract, model input projection, export, accuracy evaluation, and runtime integration. Zero-padding a 1,662-vector and passing it to the current ORT graph would fail its fixed input-shape validation.

Face landmarks also raise a cost/benefit question. Non-manual facial markers can be linguistically important, but a full 468-point mesh contributes 84.5% of the 1,662-value vector. Production design should measure whether a reduced face subset provides comparable accuracy with lower bandwidth, memory, and compute cost.

## 6. Sequence model and ONNX Runtime execution

### 6.1 Active model architecture

The active checkpoint is a small no-CNN transformer fine-tuned on the team's scoped data, not an LSTM. Its principal layers are:

- Linear projection from 134 input features to hidden width 256.
- Learned absolute positional embeddings with maximum capacity 256 positions.
- Layer normalization and embedding dropout.
- Two BERT-style transformer encoder layers.
- Four attention heads per transformer layer.
- Temporal max pooling across all 169 positions.
- Dropout with probability 0.2 during training only.
- Linear classification head from 256 hidden values to two class logits.
- Numerically stable softmax applied by the Python runtime.

The fixed ONNX input is named `input`; the output is named `logits`. Export uses opset 17 with no dynamic axes.

### 6.2 Model packaging

The verified PyTorch checkpoint is exported by `scripts/export_refined_onnx.py`, then receives dynamic QInt8 weight quantization. The quantized graph runs directly through ONNX Runtime.

The generated artifacts observed during implementation are approximately:

- PyTorch training checkpoint: approximately 14.4 MB.
- Quantized deployment ONNX: approximately 3.68 MB.

The approved deployment ONNX, aggregate metrics, and training checkpoint are checked into the current prototype repository; raw recordings remain excluded.

### 6.3 Inference mechanics

The runtime executes exact shape `(1, 169, 134)` with `CPUExecutionProvider`, extracts the first batch row, applies softmax, sorts probabilities in descending order, maps indices through the two-label metadata, and returns the requested top-k candidates.

If the first probability is below 0.75, the API returns `no_confident_match` rather than presenting a translation. Otherwise, it returns `recognized` and ranked candidates.

The 0.75 threshold is a measured prototype operating point, not a universal guarantee. Softmax confidence is not equivalent to correctness, especially under capture-domain shift, so broader promotion still requires held-out calibration, per-class precision/recall, confusion matrices, and an explicit unknown/no-sign class.

## 7. Output stabilization: implemented behavior and required refactor

The current system performs one inference per explicitly bounded sign. It does not continuously slide across a live stream; therefore it has no repeated frame-level predictions to smooth and no UI flicker debounce state.

The requested `confidence > 0.80` gate and majority voting are not present. Claiming them as implemented would be inaccurate.

A defensible continuous-recognition target should maintain these separate controls:

1. A bounded feature buffer containing a fixed number of temporally ordered frames.
2. A stride controlling how often a new inference is scheduled rather than inferring on every frame.
3. A no-sign/activity detector that prevents classification during rest or transition motion.
4. Confidence calibration derived from held-out signers, not a globally guessed threshold.
5. A history of recent top labels and probabilities.
6. A stability rule such as the same label winning at least four of the last five eligible windows.
7. A minimum stable duration before committing text.
8. A cooldown or release condition requiring a return to no-sign before emitting the same word again.

Majority voting alone is insufficient because adjacent sliding windows overlap heavily and are statistically correlated. The production state machine must combine activity segmentation, calibrated probability, temporal consensus, and duplicate suppression.

## 8. Sliding-window and asynchronous ingestion target

No `collections.deque(maxlen=30)` exists in the current repository. The nearest implemented equivalent is a per-sequence Python list finalized by the client. A production continuous pipeline may use a bounded deque, but its length must match the newly trained model contract.

For a proposed `(30, 1662)` float32 feature buffer, the raw numeric payload would be `30 × 1662 × 4 = 199,440` bytes per active signer, excluding Python and runtime overhead. `deque(maxlen=30)` provides constant-time append and automatic eviction of the oldest frame when full; it bounds history length but does not itself provide thread safety for a multi-stage application, timestamps, gap detection, backpressure, or consistent sampling.

A production WebSocket protocol should define at least:

- Protocol version and authenticated device/session identity.
- Binary frame envelope with sequence number and monotonic capture timestamp.
- Encoded image format, width, height, rotation, and camera-facing metadata.
- Maximum message size and negotiated target frame rate.
- Heartbeat, reconnect, duplicate suppression, and resume semantics.
- Server acknowledgements or explicit lossy behavior.
- Bounded decode and inference queues.
- Backpressure policy; for real-time vision, dropping stale frames is generally preferable to accumulating latency.
- Metrics for capture-to-server, decode, landmark, queue, inference, stabilization, and output latency.

Asynchronous network handling does not make MediaPipe or ONNX execution non-blocking. CPU-heavy decode, landmark extraction, and inference must run in bounded worker executors or dedicated processes so they do not stall the socket event loop. Per-session state should be owned by one ordered processing actor or protected against concurrent mutation.

## 9. Dual capture-node and emergency architecture

### 9.1 Implemented dual capture concept

The current system supports two alternative visual sources that converge on the same landmark-model family:

- Node 1: smartphone or laptop browser records a short encoded video, currently the more compatible route for the video-trained checkpoint.
- Node 2: ESP32-CAM is intended to send a sequence of JPEG snapshots through `/api/frames`.

These are fallback capture paths, not concurrently fused camera nodes. There is no active WebSocket session and no ESP32 firmware in this repository.

### 9.2 Emergency control status

Emergency phrases are roadmap behavior. The repository contains no emergency API route, no GPIO mapping, no debounce implementation, no ESP32 interrupt-service routine, and no measured emergency latency.

An ESP32 GPIO interrupt cannot directly interrupt a remote Python inference event loop. The accurate technical sequence would be:

1. A physical input changes state on the ESP32.
2. A minimal interrupt-service routine records the event or notifies a high-priority firmware task; it should not perform Wi-Fi or HTTP operations inside the ISR.
3. The firmware task debounces the input and transmits an authenticated emergency event.
4. A dedicated server endpoint or message channel validates the event and writes an emergency state with priority over recognition output.
5. The client immediately presents a preconfigured emergency phrase through text and/or speech.

Near-zero latency cannot be claimed for an HTTP path because it includes ISR scheduling, firmware task wake-up, Wi-Fi association state, network transmission, server scheduling, and client delivery. A credible target requires measurement of p50, p95, and p99 end-to-end latency. For resiliency, the ESP32 or phone should also hold the emergency phrase locally so loss of the ML server does not disable it.

To prevent heavy ML work from delaying the emergency path, production deployment should isolate emergency control from the inference worker pool. Options include a separate lightweight service, a priority message channel, or a phone-local BLE/GATT control path. Merely adding another route to a single blocking inference process would not guarantee priority.

## 10. Offline ISL refinement workflow

### 10.1 Dataset preparation

`scripts/prepare_isl_training_data.py` reads the local JSONL manifest, skips missing referenced recordings with warnings, normalizes labels, and requires at least three signer codes globally and for every label. It assigns entire signer identities to train, validation, or test splits. This prevents the same person from appearing in both training and held-out evaluation.

Each video is processed through the same MediaPipe extraction path as the base checkpoint. Generated JSON landmark samples are written into train, validation, and test directories with a scoped label map and a persisted signer-split assignment.

### 10.2 Transfer learning

`scripts/train_isl_refinement.py` requires at least two labels. It initializes the small transformer with the INCLUDE checkpoint's feature projection, position embeddings, and encoder weights while excluding the original 263-class final layer. A new output head is created for the scoped Ishaara vocabulary.

The entire model is optimized with AdamW, default learning rate `2e-5`, weight decay 0.01, gradient-norm clipping at 1.0, and cross-entropy loss. Validation is computed every epoch. The checkpoint selected for final testing is the epoch with best validation macro-F1. Final output includes held-out-signer accuracy and macro-F1.

This workflow reduces but does not eliminate evaluation risk. With only a few signers, each split is small and identity-specific. Production evaluation should add more signers, class balance checks, per-class precision/recall/F1, confusion matrices, camera-domain stratification, repeated group cross-validation, and a dedicated no-sign/unknown class.

### 10.3 Promotion boundary

The fine-tuning script writes a PyTorch `.pth` artifact and metrics. After review, `scripts/export_refined_onnx.py` creates the quantized deployment model. The service can promote it with `ISHAARA_REFINED_MODEL` and `ISHAARA_REFINED_METADATA` without deleting the prior checkpoint.

## 11. Engineering rationale for temporal classification

Static image classification observes a single pose and cannot reliably represent movement direction, speed, repetition, trajectory, or the transition between hand configurations. Many signs can share a similar instantaneous handshape while differing in motion or ordering. A temporal model receives an ordered landmark sequence and can learn dependencies across frames.

Landmark-based temporal classification also reduces the input dimensionality relative to raw RGB video and removes some background appearance variation. It does not remove all domain shift: landmark quality remains sensitive to framing, occlusion, motion blur, illumination, camera viewpoint, skin/background contrast, and detector version.

The implemented fixed-clip design simplifies segmentation because the user explicitly starts and ends one sign. Continuous recognition is substantially harder. The system must simultaneously preserve ordering, reject stale frames, normalize time, detect sign boundaries, manage transition motion, schedule inference without backlog, stabilize correlated window outputs, suppress duplicate words, and keep high-priority control events responsive.

The non-trivial engineering value is not the presence of a deque or an asynchronous keyword. It is maintaining explicit contracts and bounded latency across capture, transport, decode, landmark extraction, temporal buffering, inference, decision stabilization, and accessible output under real device variability.

## 12. Production end-state

The intended production evolution is:

1. Validate a small ISL vocabulary using representative signers and the same camera geometry as the product.
2. Export and integrate the validated scoped model with calibrated rejection behavior.
3. Add a true unknown/no-sign class and activity segmentation.
4. Define and implement a versioned, authenticated stream protocol with bounded queues and timestamps.
5. Implement continuous temporal windows and a measured stabilization state machine.
6. Add the ESP32 emergency firmware path and an inference-independent emergency service or phone-local control path.
7. Transition the primary capture source to the OV2640 ESP32-CAM after validating frame rate, JPEG quality, power consumption, thermal behavior, Wi-Fi reliability, and wearable ergonomics.
8. Retain smartphone capture as a supported fallback and data-collection path.
9. Move inference to the phone where feasible, reducing round-trip latency and improving privacy and offline operation.
10. Expand vocabulary only when each added class has representative data and held-out evaluation.

## 13. Claims suitable for technical review

The team can accurately state that the repository implements:

- Temporal isolated-sign classification over a fixed 169-frame landmark tensor.
- A 134-feature representation combining upper-body pose and both hands.
- Separate mobile-video and ESP32-compatible JPEG ingestion routes.
- Signer-hand spatial association and wrist-based hand ordering.
- Missing-detection interpolation and fixed-length padding.
- A quantized transformer deployed through ONNX Runtime.
- Explicit low-confidence rejection rather than unconditional translation.
- Local, consent-aware data capture and signer-separated transfer learning.
- Verified control-clip tests demonstrating that the base checkpoint executes correctly while also exposing camera-domain mismatch.

The team should not state that the repository currently implements:

- WebSocket video streaming.
- A 30-frame rolling deque.
- 1,662-value Holistic features or face mesh input.
- Active LSTM inference.
- An 80% calibrated threshold.
- Majority voting or output debounce.
- ESP32 GPIO interrupt firmware or a near-zero-latency emergency override.
- Production-ready or unrestricted ISL accuracy.

These are valid target-architecture items, but each should be presented as planned engineering work with an explicit validation method rather than as completed code.
