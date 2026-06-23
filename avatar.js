console.log("AVATAR JS LOADED");

let scene, camera, renderer, avatar;
let skeleton = null;
const boneNameMap = {};

let poseFrames = {};
let maxFrame = 0;
let fps = 30;
let currentAnimationName = null;
let currentAnimationPayload = null;
let currentModelUrl = null;
let modelLoadRequestId = 0;

const DEFAULT_MODEL_URL = "/models/woman.glb";
const AVATAR_MODEL_STORAGE_KEY = "sign-demo-avatar-model";
const modelAssetCache = new Map();
const modelAssetPromises = new Map();
let avatarManifestPromise = null;

const clock = new THREE.Clock();
const stageEl = document.getElementById("avatar-stage");

function normalizeBoneName(name) {
  if (!name) return name;
  return String(name).trim().replace(/^mixamorig[:_.]?/i, "");
}

function log(...args) {
  console.log("[AVATAR]", ...args);
}

function init() {
  log("stageEl =", stageEl);

  if (!stageEl) {
    console.error("avatar-stage container not found.");
    return;
  }

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202020);

  const width = stageEl.clientWidth || 800;
  const height = stageEl.clientHeight || 600;

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 8.0, 23.0);
  camera.lookAt(0, 8.0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputEncoding = THREE.sRGBEncoding;

  stageEl.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
  scene.add(hemi);

  const dirLight = new THREE.DirectionalLight(0xffffff, 2);
  dirLight.position.set(5, 10, 5);
  scene.add(dirLight);

  window.addEventListener("resize", onWindowResize);
}

function clearBoneNameMap() {
  Object.keys(boneNameMap).forEach((key) => {
    delete boneNameMap[key];
  });
}

function normalizeModelUrl(url) {
  if (!url) return DEFAULT_MODEL_URL;
  return url.startsWith("/") ? url : `/models/${url}`;
}

function getStoredAvatarModelUrl() {
  try {
    const stored = window.localStorage.getItem(AVATAR_MODEL_STORAGE_KEY);
    return stored ? normalizeModelUrl(stored) : null;
  } catch (err) {
    console.warn("Could not read stored avatar model:", err);
    return null;
  }
}

function rememberCurrentAvatarModel() {
  if (!currentModelUrl) return;

  try {
    window.localStorage.setItem(AVATAR_MODEL_STORAGE_KEY, currentModelUrl);
  } catch (err) {
    console.warn("Could not store avatar model:", err);
  }
}

function prepareLoadedAvatar(gltf) {
  const nextAvatar = gltf.scene;
  let nextSkeleton = null;

  nextAvatar.scale.set(10, 10, 10);

  const bbox = new THREE.Box3().setFromObject(nextAvatar);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bbox.getSize(size);
  bbox.getCenter(center);

  log("Model bbox size:", size);
  log("Model bbox center:", center);

  nextAvatar.position.sub(center);

  nextAvatar.traverse((obj) => {
    if (obj.isSkinnedMesh && !nextSkeleton) {
      nextSkeleton = obj.skeleton;
    }
  });

  return { nextAvatar, nextSkeleton };
}

function installLoadedAvatar(nextAvatar, nextSkeleton) {
  const previousAvatar = avatar;

  if (previousAvatar && previousAvatar !== nextAvatar) {
    scene.remove(previousAvatar);
  }

  avatar = nextAvatar;
  skeleton = nextSkeleton;
  clearBoneNameMap();

  if (avatar.parent !== scene) {
    scene.add(avatar);
  }

  if (!skeleton) {
    console.warn("No skeleton found!");
  } else {
    skeleton.bones.forEach((bone) => {
      const key = normalizeBoneName(bone.name);
      boneNameMap[key] = bone;
    });
    log("Skeleton bones normalized:", Object.keys(boneNameMap));
  }
}

function loadAvatarAsset(url) {
  const modelUrl = normalizeModelUrl(url);

  if (modelAssetCache.has(modelUrl)) {
    return Promise.resolve(modelAssetCache.get(modelUrl));
  }

  if (modelAssetPromises.has(modelUrl)) {
    return modelAssetPromises.get(modelUrl);
  }

  log("Loading avatar asset:", modelUrl);

  const loader = new THREE.GLTFLoader();
  const promise = new Promise((resolve, reject) => {
    loader.load(
      modelUrl,
      function (gltf) {
        const asset = prepareLoadedAvatar(gltf);
        modelAssetCache.set(modelUrl, asset);
        modelAssetPromises.delete(modelUrl);
        log("Avatar asset cached:", modelUrl);
        resolve(asset);
      },
      undefined,
      function (err) {
        modelAssetPromises.delete(modelUrl);
        console.error("GLB load failed:", err);
        reject(err);
      }
    );
  });

  modelAssetPromises.set(modelUrl, promise);
  return promise;
}

async function loadModel(url = DEFAULT_MODEL_URL, options = {}) {
  const modelUrl = normalizeModelUrl(url);
  const requestId = ++modelLoadRequestId;
  const shouldFetchDefaultAnimation = Boolean(options.fetchDefaultAnimation);

  log("Trying to load model from:", modelUrl);

  try {
    const { nextAvatar, nextSkeleton } = await loadAvatarAsset(modelUrl);

    if (requestId !== modelLoadRequestId) {
      return false;
    }

    installLoadedAvatar(nextAvatar, nextSkeleton);
    currentModelUrl = modelUrl;
    rememberCurrentAvatarModel();

    log("GLB installed successfully.");

    if (shouldFetchDefaultAnimation) {
      await fetchDefaultAnimationFromBackend();
    } else if (currentAnimationPayload) {
      applyAnimationPayload(currentAnimationPayload);
    } else {
      resetAnimationPlayback();
    }

    return true;
  } catch (err) {
    console.error("Avatar model switch failed:", err);
    throw err;
  }
}

async function loadAvatarManifest() {
  if (!avatarManifestPromise) {
    avatarManifestPromise = fetch("/models/avatars.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`avatars.json HTTP ${response.status}`);
        }
        return response.json();
      });
  }

  return avatarManifestPromise;
}

async function getManifestDefaultAvatarModelUrl() {
  try {
    const avatars = await loadAvatarManifest();
    const defaultEntry = Object.values(avatars).find((avatarData) => avatarData.default);
    return defaultEntry ? normalizeModelUrl(defaultEntry.model) : DEFAULT_MODEL_URL;
  } catch (err) {
    console.warn("Avatar manifest default failed:", err);
    return DEFAULT_MODEL_URL;
  }
}

async function getInitialAvatarModelUrl() {
  return getStoredAvatarModelUrl() || await getManifestDefaultAvatarModelUrl();
}

async function preloadAvatarModelsFromManifest() {
  try {
    const avatars = await loadAvatarManifest();
    const modelUrls = Object.values(avatars)
      .map((item) => normalizeModelUrl(item.model))
      .filter((url, index, all) => url && all.indexOf(url) === index);

    modelUrls.forEach((modelUrl) => {
      if (modelUrl === currentModelUrl) return;

      loadAvatarAsset(modelUrl)
        .then(() => {
          log("Avatar preloaded:", modelUrl);
        })
        .catch((err) => {
          console.warn("Avatar preload failed:", modelUrl, err);
        });
    });
  } catch (err) {
    console.warn("Avatar manifest preload failed:", err);
  }
}

function preparePoseFrames(data) {
  poseFrames = {};
  maxFrame = 0;

  if (!data || !data.bones) {
    console.error("Invalid animation data: missing bones");
    return;
  }

  for (const name in data.bones) {
    const arr = [];
    const keyframes = data.bones[name];

    keyframes.forEach((k) => {
      arr[k.f] = new THREE.Quaternion(
        k.rot[0],
        k.rot[1],
        k.rot[2],
        k.rot[3]
      );
      if (k.f > maxFrame) maxFrame = k.f;
    });

    const norm = normalizeBoneName(name);
    poseFrames[norm] = arr;
  }

  log("Prepared animation. fps =", fps, "maxFrame =", maxFrame);
}

function applyPoseFrame(frame) {
  if (!avatar || !skeleton) return;

  for (const name in poseFrames) {
    const bone = boneNameMap[name];
    if (!bone) continue;

    const q = poseFrames[name][frame];
    if (!q) continue;

    bone.quaternion.copy(q);
  }

  avatar.updateMatrixWorld(true);
}

function resetAnimationPlayback() {
  updateAnimation.time = 0;
  applyPoseFrame(0);
}

function applyAnimationPayload(payload) {
  if (!payload) {
    console.error("applyAnimationPayload: payload is empty");
    return;
  }

  log("Applying payload:", payload);
  currentAnimationPayload = payload;

  const framesData = payload.frames;
  if (!framesData) {
    console.error("Payload missing frames field");
    return;
  }

  fps = framesData.fps || 30;
  currentAnimationName = payload.animation || "unknown";

  preparePoseFrames(framesData);

  resetAnimationPlayback();

  log("Animation applied:", currentAnimationName);
}

async function fetchDefaultAnimationFromBackend() {
  try {
    log("Fetching default animation from /api/start");
    const res = await fetch("/api/start");
    if (!res.ok) {
      throw new Error(`GET /api/start failed: HTTP ${res.status}`);
    }

    const payload = await res.json();
    applyAnimationPayload(payload);
  } catch (err) {
    console.error("fetchDefaultAnimationFromBackend failed:", err);
  }
}

async function fetchEndAnimationFromBackend() {
  try {
    log("Fetching end animation from /api/end");
    const res = await fetch("/api/end", {
      method: "POST"
    });

    if (!res.ok) {
      throw new Error(`POST /api/end failed: HTTP ${res.status}`);
    }

    const payload = await res.json();
    applyAnimationPayload(payload);
  } catch (err) {
    console.error("fetchEndAnimationFromBackend failed:", err);
  }
}

function updateAnimation(dt) {
  if (!avatar || !skeleton || !maxFrame) return;

  if (updateAnimation.time === undefined) updateAnimation.time = 0;
  updateAnimation.time += dt;

  const totalFrames = maxFrame + 1;
  const currentFrame = Math.floor((updateAnimation.time * fps) % totalFrames);

  applyPoseFrame(currentFrame);
}

function animate() {
  requestAnimationFrame(animate);

  const dt = clock.getDelta();
  updateAnimation(dt);

  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

function onWindowResize() {
  if (!renderer || !camera || !stageEl) return;

  const width = stageEl.clientWidth || 800;
  const height = stageEl.clientHeight || 600;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

window.switchAvatarModel = async function (modelUrl) {
  return loadModel(modelUrl, { fetchDefaultAnimation: false });
};

window.resetAnimationPlayback = resetAnimationPlayback;

window.getCurrentAnimationName = function () {
  return currentAnimationName;
};

window.addEventListener("error", function (e) {
  console.error("Global JS error:", e.message, "@", e.filename + ":" + e.lineno);
});

window.addEventListener("unhandledrejection", function (e) {
  console.error("Unhandled promise rejection:", e.reason);
});

window.addEventListener("beforeunload", rememberCurrentAvatarModel);

log("Page loaded. href =", window.location.href);
log("THREE version:", THREE.REVISION);

if (!THREE.GLTFLoader) {
  console.error("GLTFLoader NOT detected!");
} else {
  log("GLTFLoader detected.");
}

async function startAvatarApp() {
  const initialModelUrl = await getInitialAvatarModelUrl();

  try {
    await loadModel(initialModelUrl, { fetchDefaultAnimation: true });
    preloadAvatarModelsFromManifest();
  } catch (err) {
    console.error("Initial avatar load failed:", err);

    if (initialModelUrl !== DEFAULT_MODEL_URL) {
      try {
        await loadModel(DEFAULT_MODEL_URL, { fetchDefaultAnimation: true });
        preloadAvatarModelsFromManifest();
      } catch (fallbackErr) {
        console.error("Fallback avatar load failed:", fallbackErr);
      }
    }
  }
}

init();
startAvatarApp();
animate();
