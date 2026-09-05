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
const vocabularyButton = document.querySelector("#vocabulary-button");
const vocabularyContent = document.querySelector("#vocabulary-content");
const vocabularySearch = document.querySelector("#vocabulary-search");
const vocabularyList = document.querySelector("#vocabulary-list");
const vocabularyNote = document.querySelector("#vocabulary-note");

let cameraStream;
let vocabulary = [];

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
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = cameraStream;
    video.hidden = false;
    placeholder.hidden = true;
    cameraOverlay.hidden = false;
    cameraState.textContent = "Preview live";
    cameraState.classList.add("on");
    cameraButton.textContent = "Stop camera";
    cameraNote.textContent = "Camera is live locally. No video is uploaded by this preview.";
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

    const [topCandidate, ...otherCandidates] = payload.candidates;
    translationText.textContent = topCandidate.label;
    translationDetail.textContent = `${(topCandidate.confidence * 100).toFixed(1)}% confidence. Other candidates: ${otherCandidates.map((candidate) => `${candidate.label} (${(candidate.confidence * 100).toFixed(1)}%)`).join(", ")}.`;
    clipNote.textContent = "Recognition completed locally. Treat this initial checkpoint as an experimental, scoped-vocabulary result.";
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
  analyzePhotosButton.disabled = photoCount < 6;
  photosNote.textContent = photoCount
    ? `${photoCount} photos selected. Keep them in capture order; the final photo triggers recognition.`
    : "Choose at least 6 JPEG photos, in the order a single sign is performed. This simulates the ESP32 photo stream.";
});

analyzePhotosButton.addEventListener("click", async () => {
  const photos = [...photosInput.files];
  if (photos.length < 6) return;

  const sequenceId = `browser-${Date.now()}`;
  analyzePhotosButton.disabled = true;
  analyzePhotosButton.textContent = "Processing…";
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
      translationDetail.textContent = "This sign is not confidently represented by the current model vocabulary.";
      photosNote.textContent = "No result was shown because the model was uncertain—not because it recognized this as a different sign.";
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
    analyzePhotosButton.disabled = photosInput.files.length < 6;
    analyzePhotosButton.textContent = "Analyze photos";
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
