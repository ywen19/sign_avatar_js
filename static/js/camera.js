/**
 * Manages the current camera-preview placeholder.
 *
 * This script owns camera permission, preview startup, and media-track cleanup.
 * Future work can extend the camera flow with recording, sign-language
 * recognition, LLM processing, and avatar-response generation.
 */

// Cache the page elements used to display and control the camera preview.
const videoEl = document.getElementById("camera-video");
const cameraEl = document.querySelector(".camera");
const toggleBtn = document.getElementById("camera-toggle-btn");
const toggleIcon = document.getElementById("camera-toggle-icon");
const statusEl = document.getElementById("camera-status");

// Track the owned media stream and prevent overlapping toggle operations.
let stream = null;
let isCameraChanging = false;

/**
 * Publish a camera-status message in the shared frontend status area.
 *
 * @param {string} message - Status text to display.
 */
function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

/**
 * Update the toggle icon for the current stream and pointer-hover state.
 *
 * @param {boolean} isHovering - Whether the pointer is over the toggle.
 */
function updateToggleIcon(isHovering = false) {
  if (!toggleIcon) return;

  const isActive = Boolean(stream);
  const showActiveIcon = isHovering ? !isActive : isActive;
  toggleIcon.src = showActiveIcon
    ? "/static/src/video_on.png"
    : "/static/src/video_off.png";
}

/**
 * Synchronize camera styling and accessible button text with camera state.
 *
 * @param {boolean} isActive - Whether the camera preview is active.
 */
function setCameraActive(isActive) {
  if (cameraEl) {
    cameraEl.classList.toggle("camera-active", isActive);
  }

  if (toggleBtn) {
    const label = isActive ? "Turn camera off" : "Turn camera on";
    toggleBtn.setAttribute("aria-label", label);
  }

  updateToggleIcon(false);
}

/**
 * Stop every owned media track and detach the stream from the video element.
 */
function releaseCameraStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  if (videoEl) {
    videoEl.srcObject = null;
  }

  stream = null;
}

/**
 * Request camera access and attach the resulting stream to the preview.
 *
 * @returns {Promise<void>}
 */
async function startCamera() {
  try {
    if (stream) {
      setStatus("Camera is already running.");
      return;
    }

    // Do not request hardware access if the preview target is unavailable.
    if (!videoEl) {
      setStatus("Camera preview is unavailable.");
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser does not support camera access.");
      return;
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    videoEl.srcObject = stream;
    await videoEl.play();
    setCameraActive(true);
    setStatus("Camera started.");
  } catch (error) {
    console.error("Failed to start camera:", error);

    // Release tracks acquired before a later preview-startup failure.
    releaseCameraStream();
    setCameraActive(false);
    setStatus("Failed to start camera. Please allow camera permission.");
  }
}

/**
 * Stop the active preview and release its camera hardware resources.
 */
function stopCamera() {
  releaseCameraStream();
  setCameraActive(false);
  setStatus("Camera stopped.");
}

/**
 * Start or stop the camera while ignoring overlapping toggle signals.
 *
 * @returns {Promise<void>}
 */
async function toggleCamera() {
  if (isCameraChanging) return;

  isCameraChanging = true;

  try {
    if (stream) {
      stopCamera();
    } else {
      await startCamera();
    }
  } finally {
    isCameraChanging = false;
  }
}

// Connect pointer and click signals when the camera button is available.
if (toggleBtn) {
  toggleBtn.addEventListener("mouseenter", () => updateToggleIcon(true));
  toggleBtn.addEventListener("mouseleave", () => updateToggleIcon(false));
  toggleBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleCamera();
  });
}

// Release camera hardware when the browser page is being closed.
window.addEventListener("beforeunload", () => {
  releaseCameraStream();
  setCameraActive(false);
});

// Initialize the camera controls in their inactive state.
setCameraActive(false);
setStatus("Camera idle.");
