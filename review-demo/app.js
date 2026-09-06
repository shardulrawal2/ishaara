const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const cameraPanel = $("#camera-panel");
const cameraPreview = $("#camera-preview");
const cameraStatus = $("#camera-status");
const cameraToggle = $("#camera-toggle");
const cameraSelect = $("#camera-select");
const countdown = $("#countdown");
const recognizeButton = $("#recognize-now");
const mobileCaptureButton = $("#mobile-capture-button");
const mobileCaptureInput = $("#mobile-capture-input");
const actionNote = $("#action-note");
const recognizedWord = $("#recognized-word");
const recognizedDetail = $("#recognized-detail");
const confidenceBadge = $("#confidence-badge");
const confidenceFill = $("#confidence-fill");
const speakResultButton = $("#speak-result");
const conversationSign = $("#conversation-sign");
const conversationLog = $("#conversation-log");
const modelPill = $("#model-pill");
const modelPillText = $("#model-pill-text");
const settingsModelDetail = $("#settings-model-detail");
const wordList = $("#word-list");
const vocabularyCount = $("#vocabulary-count");
const vocabularySearch = $("#vocabulary-search");
const voiceSelect = $("#voice-select");
const autoSpeakToggle = $("#auto-speak");
const trainingDialog = $("#training-dialog");
const trainingLabel = $("#training-label");
const trainingSigner = $("#training-signer");
const trainingButton = $("#record-training");
const trainingStatus = $("#training-status");
const toast = $("#toast");

let cameraStream = null;
let currentResult = "";
let modelLabels = [];
let toastTimer = null;
let speechPrimed = false;

function formatLabel(label) {
  if (label === "thankyou") return "Thank you";
  return label.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 2600);
}

function setRoute(route) {
  $$(".app-view").forEach((view) => view.classList.toggle("is-active", view.dataset.view === route));
  $$(".bottom-nav button").forEach((button) => button.classList.toggle("is-active", button.dataset.route === route));
  $(".app-main").scrollTo({ top: 0, behavior: "smooth" });
}

function updateRecognitionState(title, detail, confidence = 0, accepted = false) {
  recognizedWord.textContent = title;
  recognizedDetail.textContent = detail;
  confidenceBadge.textContent = accepted ? `${Math.round(confidence * 100)}% confident` : confidence ? `${Math.round(confidence * 100)}% closest` : "Waiting";
  confidenceBadge.style.color = accepted ? "var(--green)" : "var(--muted)";
  confidenceFill.style.width = `${Math.max(0, Math.min(1, confidence)) * 100}%`;
}

function speak(text) {
  if (!("speechSynthesis" in window) || !text) {
    showToast("Speech output is not available in this browser.");
    return;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.resume();
  const utterance = new SpeechSynthesisUtterance(text);
  const selectedVoice = window.speechSynthesis.getVoices().find((voice) => voice.name === voiceSelect.value);
  if (selectedVoice) utterance.voice = selectedVoice;
  utterance.lang = "en-IN";
  utterance.rate = 1.06;
  window.speechSynthesis.speak(utterance);
}

function primeSpeechEngine() {
  if (speechPrimed || !("speechSynthesis" in window)) return;
  const warmup = new SpeechSynthesisUtterance(" ");
  warmup.volume = 0;
  window.speechSynthesis.speak(warmup);
  speechPrimed = true;
}

function loadVoices() {
  if (!("speechSynthesis" in window)) return;
  const previous = voiceSelect.value;
  const voices = window.speechSynthesis.getVoices().filter((voice) => voice.lang.toLowerCase().startsWith("en"));
  voiceSelect.innerHTML = '<option value="">System default</option>';
  voices.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice.name;
    option.textContent = `${voice.name} (${voice.lang})`;
    voiceSelect.append(option);
  });
  if (voices.some((voice) => voice.name === previous)) voiceSelect.value = previous;
}

async function loadModel() {
  try {
    const response = await fetch("/api/model", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Model unavailable");
    modelLabels = payload.labels;
    modelPill.classList.add("is-ready");
    modelPillText.textContent = `${payload.labels.length} signs ready`;
    settingsModelDetail.textContent = `${payload.model} · ${(payload.metrics.held_out_signer_accuracy * 100).toFixed(1)}% held-out accuracy`;
    renderVocabulary();
  } catch (error) {
    modelPill.classList.add("is-error");
    modelPillText.textContent = "Model unavailable";
    settingsModelDetail.textContent = error.message;
    actionNote.textContent = "Start the local Python service and refresh this page.";
  }
}

function renderVocabulary(query = "") {
  const normalized = query.trim().toLowerCase();
  const labels = modelLabels.filter((label) => label.includes(normalized));
  vocabularyCount.textContent = `${modelLabels.length} trained`;
  wordList.replaceChildren();
  labels.forEach((label) => {
    const row = document.createElement("article");
    row.className = "word-row";
    row.innerHTML = `<span class="word-letter">${formatLabel(label).charAt(0)}</span><div><b></b><small>Trained and enabled</small></div><button type="button" aria-label="Speak ${formatLabel(label)}">◖))</button>`;
    $("b", row).textContent = formatLabel(label);
    $("button", row).addEventListener("click", () => speak(formatLabel(label)));
    wordList.append(row);
  });
  if (!labels.length) {
    const empty = document.createElement("p");
    empty.className = "action-note";
    empty.textContent = "That word is not trained yet. Collect examples before adding it to the model.";
    wordList.append(empty);
  }
}

async function refreshCameraChoices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
  const current = cameraSelect.value;
  cameraSelect.innerHTML = '<option value="">Default camera</option>';
  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    cameraSelect.append(option);
  });
  if (cameras.some((camera) => camera.deviceId === current)) cameraSelect.value = current;
}

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraPanel.classList.remove("is-on");
  cameraStatus.textContent = "Camera off";
  cameraToggle.textContent = "Start camera";
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Live camera needs localhost or HTTPS. Use the phone-camera upload button instead.");
  stopCamera();
  const deviceId = cameraSelect.value;
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 720 }, height: { ideal: 720 } } : { facingMode: "user", width: { ideal: 720 }, height: { ideal: 720 } },
  });
  cameraPreview.srcObject = cameraStream;
  await cameraPreview.play();
  cameraPanel.classList.add("is-on");
  cameraStatus.textContent = "Camera live";
  cameraToggle.textContent = "Stop camera";
  await refreshCameraChoices();
  return cameraStream;
}

async function runCountdown() {
  countdown.hidden = false;
  for (let count = 3; count >= 1; count -= 1) {
    countdown.textContent = count;
    actionNote.textContent = count === 1 ? "Get ready to perform the complete sign." : "Keep your hands inside the guide.";
    await wait(850);
  }
  countdown.textContent = "SIGN";
  await wait(350);
  countdown.hidden = true;
}

function supportedMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported(type)) || "";
}

async function recordCameraClip(duration = 2500) {
  if (!cameraStream) await startCamera();
  if (!("MediaRecorder" in window)) throw new Error("This browser cannot record a camera clip. Use the phone-camera upload button.");
  const chunks = [];
  const mimeType = supportedMimeType();
  const recorder = mimeType ? new MediaRecorder(cameraStream, { mimeType }) : new MediaRecorder(cameraStream);
  recorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.addEventListener("error", () => reject(new Error("The camera recording failed.")), { once: true });
  });
  recorder.start(150);
  await wait(duration);
  recorder.stop();
  await stopped;
  if (!chunks.length) throw new Error("The camera did not return a recording.");
  return new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" });
}

async function sendRecognitionClip(blob, filename = "ishaara-camera.webm") {
  const form = new FormData();
  form.append("clip", blob, filename);
  const response = await fetch("/api/mobile/recognize", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Recognition failed.");
  return payload;
}

function presentRecognition(payload) {
  if (payload.status === "no_confident_match") {
    currentResult = "";
    speakResultButton.disabled = true;
    updateRecognitionState("No confident match", "Return to neutral and repeat one trained sign with both hands visible.", payload.confidence || 0, false);
    actionNote.textContent = "The confidence gate rejected this sample instead of guessing.";
    return;
  }
  const winner = payload.candidates[0];
  currentResult = formatLabel(winner.label);
  speakResultButton.disabled = false;
  updateRecognitionState(currentResult, "Accepted by the scoped local model. Tap the speaker to say it aloud.", winner.confidence, true);
  conversationSign.textContent = currentResult;
  actionNote.textContent = "Recognition completed locally; the temporary clip was deleted.";
  if (autoSpeakToggle.checked) speak(currentResult);
}

async function recognizeFromLiveCamera() {
  recognizeButton.disabled = true;
  mobileCaptureButton.disabled = true;
  try {
    if (!cameraStream) await startCamera();
    await runCountdown();
    actionNote.textContent = "Recording now — perform one complete sign.";
    recognizeButton.textContent = "Recording…";
    const clip = await recordCameraClip();
    recognizeButton.textContent = "Reading landmarks…";
    updateRecognitionState("Reading your sign", "Extracting hand and upper-body landmarks on this computer.");
    presentRecognition(await sendRecognitionClip(clip));
  } catch (error) {
    updateRecognitionState("Could not recognize", error.message);
    actionNote.textContent = "Improve lighting, keep both hands visible, and try again.";
  } finally {
    recognizeButton.disabled = false;
    mobileCaptureButton.disabled = false;
    recognizeButton.textContent = "Recognize one sign";
    countdown.hidden = true;
  }
}

async function recognizeUploadedClip(file) {
  recognizeButton.disabled = true;
  mobileCaptureButton.disabled = true;
  updateRecognitionState("Reading phone clip", "Uploading to the local Ishaara service for landmark extraction.");
  actionNote.textContent = "Processing the selected camera recording…";
  try {
    presentRecognition(await sendRecognitionClip(file, file.name || "phone-sign.mp4"));
  } catch (error) {
    updateRecognitionState("Could not recognize", error.message);
    actionNote.textContent = "Use one short, well-lit clip with hands and upper body visible.";
  } finally {
    recognizeButton.disabled = false;
    mobileCaptureButton.disabled = false;
    mobileCaptureInput.value = "";
  }
}

async function saveTrainingCapture() {
  const label = trainingLabel.value.trim().toLowerCase();
  const signer = trainingSigner.value.trim().toLowerCase();
  if (!label || !signer) {
    trainingStatus.textContent = "Enter both the exact sign label and a non-identifying signer code.";
    return;
  }
  trainingButton.disabled = true;
  trainingButton.textContent = "Preparing camera…";
  try {
    if (!cameraStream) await startCamera();
    trainingStatus.textContent = "Perform the sign now for 2.5 seconds.";
    trainingButton.textContent = "Recording…";
    const clip = await recordCameraClip();
    const form = new FormData();
    form.append("label", label);
    form.append("signer", signer);
    form.append("clip", clip, "training-example.webm");
    trainingButton.textContent = "Saving locally…";
    const response = await fetch("/api/training/captures", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save the recording.");
    trainingStatus.textContent = `Saved one ${formatLabel(label)} example for ${signer}. Repeat with varied angles and lighting.`;
    showToast("Training example saved locally.");
  } catch (error) {
    trainingStatus.textContent = error.message;
  } finally {
    trainingButton.disabled = false;
    trainingButton.textContent = "Record and save 2.5 seconds";
  }
}

function appendConversationMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message spoken";
  const label = document.createElement("small");
  label.textContent = "They said";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  wrapper.append(label, paragraph);
  conversationLog.append(wrapper);
  conversationLog.scrollTop = conversationLog.scrollHeight;
}

$$('[data-route]').forEach((button) => button.addEventListener("click", () => setRoute(button.dataset.route)));
$("#brand-home").addEventListener("click", () => setRoute("home"));
cameraToggle.addEventListener("click", async () => {
  if (cameraStream) { stopCamera(); return; }
  try { await startCamera(); } catch (error) { actionNote.textContent = error.message; }
});
cameraSelect.addEventListener("change", async () => {
  if (!cameraStream) return;
  try { await startCamera(); } catch (error) { showToast(error.message); }
});
recognizeButton.addEventListener("click", recognizeFromLiveCamera);
mobileCaptureButton.addEventListener("click", () => mobileCaptureInput.click());
mobileCaptureInput.addEventListener("change", () => {
  const [file] = mobileCaptureInput.files;
  if (file) recognizeUploadedClip(file);
});
speakResultButton.addEventListener("click", () => speak(currentResult));
autoSpeakToggle.addEventListener("change", () => {
  window.localStorage.setItem("ishaara-auto-speak", String(autoSpeakToggle.checked));
  showToast(autoSpeakToggle.checked ? "Automatic speech enabled." : "Automatic speech disabled.");
});
voiceSelect.addEventListener("change", () => window.localStorage.setItem("ishaara-voice", voiceSelect.value));
$("#conversation-recognize").addEventListener("click", () => setRoute("home"));
$("#message-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#message-input");
  const text = input.value.trim();
  if (!text) return;
  appendConversationMessage(text);
  speak(text);
  input.value = "";
});
vocabularySearch.addEventListener("input", () => renderVocabulary(vocabularySearch.value));
$("#open-training").addEventListener("click", () => trainingDialog.showModal());
$("#close-training").addEventListener("click", () => trainingDialog.close());
trainingButton.addEventListener("click", saveTrainingCapture);
$("#emergency-list").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  speak(button.textContent.trim());
  showToast("Emergency phrase spoken aloud.");
});
$$('[data-theme-choice]').forEach((button) => button.addEventListener("click", () => {
  const theme = button.dataset.themeChoice;
  document.body.dataset.theme = theme;
  window.localStorage.setItem("ishaara-theme", theme);
  $$('[data-theme-choice]').forEach((choice) => choice.classList.toggle("is-selected", choice === button));
}));

document.addEventListener("pointerdown", primeSpeechEngine, { once: true });
window.addEventListener("beforeunload", stopCamera);
if ("speechSynthesis" in window) window.speechSynthesis.addEventListener("voiceschanged", loadVoices);

const storedTheme = window.localStorage.getItem("ishaara-theme") || "light";
document.body.dataset.theme = storedTheme;
$$('[data-theme-choice]').forEach((choice) => choice.classList.toggle("is-selected", choice.dataset.themeChoice === storedTheme));
autoSpeakToggle.checked = window.localStorage.getItem("ishaara-auto-speak") !== "false";
loadVoices();
window.setTimeout(() => {
  loadVoices();
  const storedVoice = window.localStorage.getItem("ishaara-voice");
  if (storedVoice && [...voiceSelect.options].some((option) => option.value === storedVoice)) voiceSelect.value = storedVoice;
}, 250);
loadModel();
