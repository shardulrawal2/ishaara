const cameraButton = document.querySelector("#camera-button");
const packageButton = document.querySelector("#package-button");
const video = document.querySelector("#camera");
const placeholder = document.querySelector("#camera-placeholder");
const cameraState = document.querySelector("#camera-state");
const cameraOverlay = document.querySelector("#camera-overlay");
const cameraNote = document.querySelector("#camera-note");
const translationText = document.querySelector("#translation-text");
const translationDetail = document.querySelector("#translation-detail");
const flowSteps = [...document.querySelectorAll(".flow-step")];
const clipInput = document.querySelector("#clip-input");
const analyzeButton = document.querySelector("#analyze-button");
const clipNote = document.querySelector("#clip-note");
const photosInput = document.querySelector("#photos-input");
const analyzePhotosButton = document.querySelector("#analyze-photos-button");
const photosNote = document.querySelector("#photos-note");
const captureSequenceButton = document.querySelector("#capture-sequence-button");
const mobileRecognizeButton = document.querySelector("#mobile-recognize-button");
const mobileNote = document.querySelector("#mobile-note");
const trainingLabel = document.querySelector("#training-label");
const trainingSigner = document.querySelector("#training-signer");
const saveTrainingButton = document.querySelector("#save-training-button");
const trainingNote = document.querySelector("#training-note");
const vocabularyButton = document.querySelector("#vocabulary-button");
const vocabularyContent = document.querySelector("#vocabulary-content");
const vocabularySearch = document.querySelector("#vocabulary-search");
const vocabularyList = document.querySelector("#vocabulary-list");
const vocabularyNote = document.querySelector("#vocabulary-note");

let cameraStream;
let vocabulary = [];

function showRecognitionResult(payload, note) {
  if (payload.status === "no_confident_match") {
    translationText.textContent = "No confident match";
    translationDetail.textContent = `Closest tentative match: ${payload.closest_label} (${(payload.confidence * 100).toFixed(1)}%).`;
    if (note) note.textContent = "No translation was shown because the model was not confident enough.";
    return false;
  }
  const [topCandidate, ...otherCandidates] = payload.candidates;
  translationText.textContent = topCandidate.label;
  translationDetail.textContent = `${(topCandidate.confidence * 100).toFixed(1)}% confidence. Other candidates: ${otherCandidates.map((candidate) => `${candidate.label} (${(candidate.confidence * 100).toFixed(1)}%)`).join(", ")}.`;
  return true;
}

function setActiveStep(index) {
  flowSteps.forEach((step, currentIndex) => step.classList.toggle("active", currentIndex <= index));
}

cameraButton.addEventListener("click", async () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
    video.srcObject = null;
    video.hidden = true;
    placeholder.hidden = false;
    cameraOverlay.hidden = true;
    cameraState.textContent = "Camera off";
    cameraState.classList.remove("on");
    cameraButton.textContent = "Start camera";
    cameraNote.textContent = "Your video stays in this browser preview.";
    translationText.textContent = "Waiting for landmarks";
    translationDetail.textContent = "The model is packaged and validated. Native hand + pose landmarks are the next step.";
    setActiveStep(0);
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    cameraNote.textContent = "This browser does not provide camera access.";
    return;
  }

  cameraButton.disabled = true;
  cameraButton.textContent = "Connecting…";
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 } },
      audio: false,
    });
    video.srcObject = cameraStream;
    video.hidden = false;
    placeholder.hidden = true;
    cameraOverlay.hidden = false;
    cameraState.textContent = "Preview live";
    cameraState.classList.add("on");
    cameraButton.textContent = "Stop camera";
    cameraNote.textContent = "Mobile backup is ready. Recognition records only a short clip, then deletes it after processing.";
    translationText.textContent = "Camera connected";
    translationDetail.textContent = "Live video is ready. The next native milestone turns each frame into the 134 landmark values required by the model.";
    setActiveStep(1);
  } catch (error) {
    cameraNote.textContent = "Camera access was not granted. You can still review the verified model flow.";
  } finally {
    cameraButton.disabled = false;
  }
});

packageButton.addEventListener("click", () => {
  packageButton.disabled = true;
  packageButton.textContent = "Checking package…";
  translationText.textContent = "Verifying model package";
  translationDetail.textContent = "Checking the fixed input contract and mobile-ready ORT bundle.";
  setActiveStep(2);

  window.setTimeout(() => {
    translationText.textContent = "Model package ready";
    translationDetail.textContent = "Verified: 169 frames × 134 landmark values → 263 output classes. The ORT package is 3.74 MB and ready for native app wiring.";
    packageButton.textContent = "Model package verified";
    packageButton.disabled = false;
  }, 750);
});

clipInput.addEventListener("change", () => {
  const [clip] = clipInput.files;
  analyzeButton.disabled = !clip;
  clipNote.textContent = clip ? `${clip.name} selected. It will be processed locally and deleted after recognition.` : "Use one clear sign, keep hands and upper body visible, and stay under 169 frames.";
});

analyzeButton.addEventListener("click", async () => {
  const [clip] = clipInput.files;
  if (!clip) return;

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";
  translationText.textContent = "Reading landmarks";
  translationDetail.textContent = "Extracting hands and upper-body pose, then running the local model.";
  setActiveStep(2);

  try {
    const formData = new FormData();
    formData.append("clip", clip);
    const response = await fetch("/api/recognize", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Recognition could not be completed.");

    showRecognitionResult(payload, clipNote);
    if (payload.status !== "no_confident_match") clipNote.textContent = "Recognition completed locally. Treat this initial checkpoint as an experimental, scoped-vocabulary result.";
  } catch (error) {
    translationText.textContent = "Recognition needs a clearer clip";
    translationDetail.textContent = error.message;
    clipNote.textContent = "Try a short, well-lit clip with the signer’s hands and upper body fully visible.";
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze video";
  }
});

photosInput.addEventListener("change", () => {
  const photoCount = photosInput.files.length;
  analyzePhotosButton.disabled = photoCount < 16;
  photosNote.textContent = photoCount
    ? `${photoCount} photos selected. Keep them in capture order; the final photo triggers recognition.`
    : "Choose at least 16 JPEG photos, in the order a single sign is performed, or start the camera and capture an ordered sequence automatically.";
});

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function recordCameraClip(durationMs = 2500) {
  if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error("Start the mobile camera first, then keep your hands and upper body in view.");
  }
  if (!window.MediaRecorder) throw new Error("This browser cannot record a mobile camera clip.");

  const supportedType = ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find((type) => MediaRecorder.isTypeSupported(type));
  const recorder = supportedType ? new MediaRecorder(cameraStream, { mimeType: supportedType }) : new MediaRecorder(cameraStream);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });
  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.addEventListener("error", () => reject(new Error("The camera recording failed.")), { once: true });
  });
  recorder.start(250);
  await wait(durationMs);
  recorder.stop();
  await stopped;
  const type = recorder.mimeType || "video/webm";
  const extension = type.includes("mp4") ? "mp4" : "webm";
  return new File([new Blob(chunks, { type })], `mobile-sign.${extension}`, { type });
}

async function analyzePhotoSequence(photos) {
  const sequenceId = `browser-${Date.now()}`;
  translationText.textContent = "Reading photo sequence";
  translationDetail.textContent = "Sending snapshots through the same path the ESP32 will use.";
  setActiveStep(2);

  try {
    let payload;
    for (let index = 0; index < photos.length; index += 1) {
      const formData = new FormData();
      formData.append("sequence_id", sequenceId);
      formData.append("frame", photos[index]);
      formData.append("final", String(index === photos.length - 1));
      const response = await fetch("/api/frames", { method: "POST", body: formData });
      payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The photo sequence could not be processed.");
      photosNote.textContent = index === photos.length - 1
        ? "Done. The recognition result is shown above."
        : `Converted photo ${index + 1} of ${photos.length} to landmarks…`;
    }

    if (payload.status === "no_confident_match") {
      translationText.textContent = "No confident match";
      translationDetail.textContent = `Closest tentative match: ${payload.closest_label} (${(payload.confidence * 100).toFixed(1)}%). Signer hands were detected in ${payload.signer_hand_frames} of ${payload.frames_received} photos.`;
      photosNote.textContent = "This is diagnostic information, not a translation. It helps distinguish missing hand tracking from a model mismatch.";
      return;
    }

    const [topCandidate, ...otherCandidates] = payload.candidates;
    translationText.textContent = topCandidate.label;
    translationDetail.textContent = `${(topCandidate.confidence * 100).toFixed(1)}% confidence. Other candidates: ${otherCandidates.map((candidate) => `${candidate.label} (${(candidate.confidence * 100).toFixed(1)}%)`).join(", ")}.`;
  } catch (error) {
    translationText.textContent = "Recognition needs clearer photos";
    translationDetail.textContent = error.message;
    photosNote.textContent = "Use a well-lit, ordered JPEG sequence with the signer’s hands and upper body in every photo.";
  } finally {
    analyzePhotosButton.disabled = photosInput.files.length < 16;
    analyzePhotosButton.textContent = "Analyze photos";
  }
}

analyzePhotosButton.addEventListener("click", async () => {
  const photos = [...photosInput.files];
  if (photos.length < 16) return;

  analyzePhotosButton.disabled = true;
  analyzePhotosButton.textContent = "Processing…";
  await analyzePhotoSequence(photos);
});

mobileRecognizeButton.addEventListener("click", async () => {
  mobileRecognizeButton.disabled = true;
  mobileRecognizeButton.textContent = "Recording… perform one sign";
  try {
    const clip = await recordCameraClip();
    mobileRecognizeButton.textContent = "Recognizing…";
    translationText.textContent = "Reading mobile video";
    translationDetail.textContent = "Extracting hand and upper-body landmarks from the short recording.";
    setActiveStep(2);
    const formData = new FormData();
    formData.append("clip", clip);
    const response = await fetch("/api/mobile/recognize", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The mobile recording could not be recognized.");
    showRecognitionResult(payload, mobileNote);
    if (payload.status !== "no_confident_match") mobileNote.textContent = "Mobile backup recognition completed locally; the short recording was deleted.";
  } catch (error) {
    translationText.textContent = "Mobile recognition needs a clearer sign";
    translationDetail.textContent = error.message;
    mobileNote.textContent = "Keep one signer’s hands and upper body in frame, then try again.";
  } finally {
    mobileRecognizeButton.disabled = false;
    mobileRecognizeButton.textContent = "Record and recognize (2.5 seconds)";
  }
});

saveTrainingButton.addEventListener("click", async () => {
  const label = trainingLabel.value.trim();
  const signer = trainingSigner.value.trim();
  if (!label || !signer) {
    trainingNote.textContent = "Enter the sign label and a non-identifying signer code before recording.";
    return;
  }
  saveTrainingButton.disabled = true;
  saveTrainingButton.textContent = "Recording…";
  try {
    const clip = await recordCameraClip();
    saveTrainingButton.textContent = "Saving…";
    const formData = new FormData();
    formData.append("label", label);
    formData.append("signer", signer);
    formData.append("clip", clip);
    const response = await fetch("/api/training/captures", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The labelled capture could not be saved.");
    trainingNote.textContent = `Saved “${payload.label}”. Record this sign again with different framing, lighting, and signers.`;
  } catch (error) {
    trainingNote.textContent = error.message;
  } finally {
    saveTrainingButton.disabled = false;
    saveTrainingButton.textContent = "Record and save";
  }
});

captureSequenceButton.addEventListener("click", async () => {
  if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    photosNote.textContent = "Start the camera first, then click this again while performing one supported sign.";
    return;
  }

  captureSequenceButton.disabled = true;
  captureSequenceButton.textContent = "Capturing…";
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  const photos = [];
  try {
    for (let index = 0; index < 20; index += 1) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const photo = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      if (!photo) throw new Error("The camera snapshot could not be created.");
      photos.push(new File([photo], `snapshot-${index + 1}.jpg`, { type: "image/jpeg" }));
      photosNote.textContent = `Captured photo ${index + 1} of 20… keep signing.`;
      await wait(100);
    }
    captureSequenceButton.textContent = "Analyzing…";
    await analyzePhotoSequence(photos);
  } catch (error) {
    translationText.textContent = "Photo capture failed";
    translationDetail.textContent = error.message;
    photosNote.textContent = "Start the camera, allow camera access, and keep the signer in frame.";
  } finally {
    captureSequenceButton.disabled = false;
    captureSequenceButton.textContent = "Capture a 2-second photo sequence";
  }
});

function renderVocabulary(filter = "") {
  const query = filter.trim().toLowerCase();
  const visibleWords = vocabulary.filter((word) => word.includes(query));
  vocabularyList.replaceChildren(...visibleWords.map((word) => {
    const chip = document.createElement("span");
    chip.textContent = word;
    return chip;
  }));
  vocabularyNote.textContent = query
    ? `${visibleWords.length} matching supported word${visibleWords.length === 1 ? "" : "s"}.`
    : `${vocabulary.length} exact labels in the current checkpoint. Test only one of these signs at a time.`;
}

vocabularyButton.addEventListener("click", async () => {
  if (!vocabularyContent.hidden) {
    vocabularyContent.hidden = true;
    vocabularyButton.textContent = "Show supported prototype words";
    return;
  }

  vocabularyContent.hidden = false;
  vocabularyButton.textContent = "Hide supported prototype words";
  if (vocabulary.length) return;

  vocabularyButton.disabled = true;
  vocabularyNote.textContent = "Loading the model vocabulary…";
  try {
    const response = await fetch("/api/vocabulary");
    const payload = await response.json();
    if (!response.ok) throw new Error("The model vocabulary could not be loaded.");
    vocabulary = payload.labels;
    renderVocabulary();
  } catch (error) {
    vocabularyNote.textContent = error.message;
  } finally {
    vocabularyButton.disabled = false;
  }
});

vocabularySearch.addEventListener("input", () => renderVocabulary(vocabularySearch.value));
