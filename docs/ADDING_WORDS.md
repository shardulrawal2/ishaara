# Adding More Words to Ishaara

Yes, the current classifier supports more than two output labels. A word becomes available in the app only after a new checkpoint containing that label passes held-out-signer testing.

## One rule that matters most

Every teammate records **the same set of words**. Do not assign one word to each person. If a model sees `welcome` from only one person, it can learn that person instead of learning the sign.

For every new label:

- Use at least three non-identifying signer codes, such as `signer-01`, `signer-02`, and `signer-03`.
- Record 25–40 clips per signer as a practical first pass.
- Keep one complete signer out of training.
- Vary distance, angle, background, clothing, and lighting.
- Record one complete isolated sign per 2.5-second clip.
- Use exactly the same label spelling for every recording.

## Collection

Open **Vocabulary → Collect samples** in the app. Enter the label and signer code, then record the sign repeatedly. The clips remain under `data\isl-field-captures` and are ignored by Git.

For an initial expansion, add only three visually distinct words. A sensible sequence is `welcome`, `help`, and `yes`, but an ISL user or interpreter must verify the signs and collection protocol before the team treats those labels as ISL ground truth.

## Prepare a fresh dataset

Never overwrite an earlier prepared dataset. From the repository root, choose a new output folder:

```powershell
.\INCLUDE\.venv\Scripts\python.exe scripts\prepare_isl_training_data.py --captures data\isl-field-captures --output data\isl-training-keypoints-v3
```

The preparation script rejects labels represented by fewer than three signer codes and assigns entire signers—not individual recordings—to train, validation, and test splits.

## Train a candidate checkpoint

Use a new artifact directory:

```powershell
.\INCLUDE\.venv\Scripts\python.exe scripts\train_isl_refinement.py --data-dir data\isl-training-keypoints-v3 --output artifacts\ishaara-v3 --epochs 35
```

Do not replace the running checkpoint just because training completed. Review validation macro-F1 and held-out signer accuracy first.

## Evaluate before promotion

```powershell
.\INCLUDE\.venv\Scripts\python.exe scripts\evaluate_refined_mvp.py --data-dir data\isl-training-keypoints-v3 --model artifacts\ishaara-v3\ishaara_isl_refinement.pth
```

Also test original videos from the held-out signer through the complete video-to-landmark-to-model path. Inspect a confusion matrix for every label; overall accuracy can hide one word that never works.

## Run the approved candidate

Point the server at the candidate without deleting the current two-word model:

```powershell
$env:ISHAARA_REFINED_MODEL="artifacts\ishaara-v3\ishaara_isl_refinement.pth"
.\INCLUDE\.venv\Scripts\python.exe local_recognition_server.py
```

The app reads labels dynamically from `/api/model` and `/api/vocabulary`, so approved new words appear automatically after the server starts with the new checkpoint. No frontend hard-coding is required.

## Promotion checklist

- Every label has at least three signers.
- No held-out signer appears in training.
- Per-label precision and recall are acceptable.
- Random gestures and unsupported signs are rejected at the configured confidence gate.
- Live camera tests work for multiple people in the intended lighting.
- The vocabulary page shows exactly the labels contained in the checkpoint.

Only after all six checks pass should the new checkpoint replace the production artifact.
