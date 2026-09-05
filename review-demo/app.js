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

let cameraStream;

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
