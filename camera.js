const videoEl = document.getElementById('camera-video');
const startBtn = document.getElementById('start-camera-btn');
const stopBtn = document.getElementById('stop-camera-btn');
const statusEl = document.getElementById('camera-status');

let stream = null;

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
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
    setStatus('Camera started.');
  } catch (error) {
    console.error('Failed to start camera:', error);
    setStatus('Failed to start camera. Please allow camera permission.');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoEl.srcObject = null;
    stream = null;
  }

  if (window.switchAvatarAnimation && window.getCurrentAnimationUrl) {
    const current = window.getCurrentAnimationUrl();
    const next = current === "Dancing_mixamo_com_frames.json" ? "Headbutt_mixamo_com_frames.json" : "Dancing_mixamo_com_frames.json";
    window.switchAvatarAnimation(next);
    setStatus(`Switched animation to: ${next}`);
  } else {
    setStatus('Camera stopped.');
  }
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);

setStatus('Camera idle.');