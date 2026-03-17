const statusEl = document.getElementById("status");
const endBtn = document.getElementById("endBtn");

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadStartAnimation() {
  const res = await fetch("/api/start");
  if (!res.ok) throw new Error("Failed to load dancing animation");
  return await res.json();
}

async function loadEndAnimation() {
  const res = await fetch("/api/end", { method: "POST" });
  if (!res.ok) throw new Error("Failed to load headbutt animation");
  return await res.json();
}

function applyCameraMode(cameraMode) {
  if (!window.camera) return;

  if (cameraMode === "start") {
    window.camera.position.set(0, 1.5, 3.5);
    window.camera.lookAt(0, 1, 0);
  }
}

function applyState(payload) {
  setStatus(`Animation: ${payload.animation}`);

  applyCameraMode(payload.camera);

  if (window.loadAvatarAnimationFromJson) {
    window.loadAvatarAnimationFromJson(payload.animation, payload.frames);
  } else {
    console.log(payload);
  }
}

async function initApp() {
  try {
    if (window.initScene) {
      window.initScene();
    }

    if (window.loadAvatarModel) {
      await window.loadAvatarModel("../models/model.glb");
    }

    const payload = await loadStartAnimation();
    applyState(payload);
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

endBtn.addEventListener("click", async () => {
  try {
    const payload = await loadEndAnimation();
    applyState(payload);
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
});

initApp();