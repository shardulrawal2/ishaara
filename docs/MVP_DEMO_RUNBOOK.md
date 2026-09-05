# Ishaara Three-Hour MVP Runbook

## Honest demo scope

The working MVP recognizes exactly two isolated ISL signs from a laptop or phone camera:

- `hello`
- `thankyou`

It uses a transformer initialized from the INCLUDE checkpoint and fine-tuned on the team's own camera recordings. The evaluation split excludes one teammate from training.

Measured offline results:

- Held-out signer set: 63 clips.
- Overall held-out accuracy: 85.7%.
- Held-out macro-F1: 0.855.
- At the 75% confidence gate: 74.6% coverage and 91.5% accepted-prediction accuracy.
- Four original held-out videos tested end to end: four accepted and four correct.

These figures apply only to the two-word dataset and recorded conditions. They are not a claim of unrestricted ISL translation.

## Start the MVP

From PowerShell in the repository root:

```powershell
cd C:\Users\SHARDUL\Downloads\dnd
.\INCLUDE\.venv\Scripts\python.exe local_recognition_server.py
```

Open `http://127.0.0.1:4173/` and refresh the page with `Ctrl + F5`.

## Perform the demo

1. Press **Start camera** and allow camera access.
2. Position one signer with both hands and the upper body visible.
3. Press **Record and recognize**.
4. Wait through the visible 3–2–1 countdown.
5. Perform one complete `hello` or `thankyou` sign while the button says **Recording now**.
6. Wait for landmark extraction and inference.
7. Read the accepted label and confidence, or explain that the confidence gate rejected an uncertain sample.

For the most reliable result, use the same framing and sign execution used during data collection. Do not demonstrate untrained words or random gestures as if the two-class model could identify them.

## Technical statement for reviewers

The browser records a temporary 2.5-second clip. The local service extracts 25 upper-body pose landmarks and 21 landmarks for each hand, producing 134 values per frame. It interpolates short detection gaps, pads the sequence to 169 frames, and runs the fine-tuned temporal transformer. Predictions below 75% confidence are not presented as translations. The temporary recognition video is deleted after processing.

The ESP32 endpoint remains in the repository but is outside this MVP demonstration because the hardware is currently unavailable.

## Fast recovery

- Camera unavailable: close other applications using the webcam, refresh, and allow permission again.
- Model unavailable: confirm `artifacts\hello-thankyou-v1\ishaara_isl_refinement.pth` exists.
- No hands detected: improve lighting, move closer, and keep the hands in the frame for the complete sign.
- Incorrect result: return to neutral, wait one second, repeat the sign once during the recording window.
- Stale interface: press `Ctrl + F5` after restarting the server.
