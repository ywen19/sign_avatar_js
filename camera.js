const videoEl = document.getElementById('camera-video');
const cameraEl = document.querySelector('.camera');
const startBtn = document.getElementById('start-camera-btn');
const stopBtn = document.getElementById('stop-camera-btn');
const statusEl = document.getElementById('camera-status');
const buttonsEl = document.querySelector('.buttons');

let stream = null;

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function setCameraActive(isActive) {
  if (cameraEl) {
    cameraEl.classList.toggle('camera-active', isActive);
  }

  if (buttonsEl) {
    buttonsEl.classList.toggle('camera-running', isActive);
  }
}

async function startCamera() {
  try {
    if (stream) {
      setStatus('Camera is already running.');
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('This browser does not support camera access.');
      return;
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    videoEl.srcObject = stream;
    await videoEl.play();
    setCameraActive(true);
    setStatus('Camera started.');
  } catch (error) {
    console.error('Failed to start camera:', error);
    setCameraActive(false);
    setStatus('Failed to start camera. Please allow camera permission.');
  }
}

async function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoEl.srcObject = null;
    stream = null;
  }

  setCameraActive(false);
  setStatus('Camera stopped.');
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    setCameraActive(false);
  }
});

setCameraActive(false);
setStatus('Camera idle.');