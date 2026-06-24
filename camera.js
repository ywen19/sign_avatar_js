const videoEl = document.getElementById('camera-video');
const cameraEl = document.querySelector('.camera');
const toggleBtn = document.getElementById('camera-toggle-btn');
const toggleIcon = document.getElementById('camera-toggle-icon');
const statusEl = document.getElementById('camera-status');

let stream = null;

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function updateToggleIcon(isHovering = false) {
  if (!toggleIcon) return;

  const isActive = Boolean(stream);
  toggleIcon.src = isHovering
    ? (isActive ? '/static/src/video_off.png' : '/static/src/video_on.png')
    : (isActive ? '/static/src/video_on.png' : '/static/src/video_off.png');
}

function setCameraActive(isActive) {
  if (cameraEl) {
    cameraEl.classList.toggle('camera-active', isActive);
  }

  if (toggleBtn) {
    toggleBtn.setAttribute('aria-label', isActive ? 'Turn camera off' : 'Turn camera on');
  }

  updateToggleIcon(false);
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

async function toggleCamera() {
  if (stream) {
    await stopCamera();
  } else {
    await startCamera();
  }
}

if (toggleBtn) {
  toggleBtn.addEventListener('mouseenter', () => updateToggleIcon(true));
  toggleBtn.addEventListener('mouseleave', () => updateToggleIcon(false));
  toggleBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    toggleCamera();
  });
}

window.addEventListener('beforeunload', () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    setCameraActive(false);
  }
});

setCameraActive(false);
setStatus('Camera idle.');
