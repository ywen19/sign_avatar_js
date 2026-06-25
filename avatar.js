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
const DEMO_MATERIAL_FIX_MODEL_URLS = new Set(["/models/man.glb", "/models/woman.glb"]);
const AVATAR_MODEL_STORAGE_KEY = "sign-demo-avatar-model";
const ANIMATION_LOOP_PAUSE_SECONDS = 3.0;
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

function applyFixedAvatarCamera() {
  if (!camera) return;

  const target = new THREE.Vector3(0, 13.1, 0);
  const distance = 16.2;
  camera.position.set(target.x, target.y, target.z + distance);
  camera.lookAt(target);
}

function init() {
  log("stageEl =", stageEl);

  if (!stageEl) {
    console.error("avatar-stage container not found.");
    return;
  }

  scene = new THREE.Scene();
  scene.background = null;

  const width = stageEl.clientWidth || 800;
  const height = stageEl.clientHeight || 600;

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
  camera.position.set(0, 8.0, 23.0);
  camera.lookAt(0, 8.0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputEncoding = THREE.sRGBEncoding;
  if (THREE.ACESFilmicToneMapping !== undefined) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.78;
  }
  renderer.setClearAlpha(0);

  stageEl.appendChild(renderer.domElement);
  applyFixedAvatarCamera();

  const hemi = new THREE.HemisphereLight(0xf6fbff, 0xb9d8df, 0.78);
  scene.add(hemi);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.18);
  keyLight.position.set(7, 4, 8);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xdceffc, 0.12);
  fillLight.position.set(-6, 5, 6);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xb9d8df, 0.24);
  rimLight.position.set(5, 6, -5);
  scene.add(rimLight);

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

function applyProductionPBRMaterialDefaults(material) {
  if (!material) return;

  // Production GLBs from the outsourcing team should use the standard PBR path.
  // Keep base color, bump/normal, alpha, and roughness maps intact here.
  if (material.envMapIntensity !== undefined) {
    material.envMapIntensity = 1;
  }

  material.needsUpdate = true;
}

function applyTemporaryDemoGLBMaterialFixes(material) {
  if (!material) return;

  // TODO: Remove this function when the final outsourced avatar models arrive.
  // The current demo GLBs contain KHR_materials_specular with specularColorFactor [2, 2, 2]
  // and a texture named "Glossiness" wired into glTF's metallicRoughnessTexture slot.
  // In Three.js, roughness is multiplied by the green channel of roughnessMap, and these
  // files have very low green-channel values, so the clothes render with strong baked shine
  // even when material.roughness is set high. These overrides are intentionally limited to
  // /models/man.glb and /models/woman.glb so final PBR assets can keep their proper maps.
  if (material.roughness !== undefined) {
    material.roughness = 0.58;
  }

  if (material.metalness !== undefined) {
    material.metalness = 0;
  }

  if (material.roughnessMap !== undefined) {
    material.roughnessMap = null;
  }

  if (material.metalnessMap !== undefined) {
    material.metalnessMap = null;
  }

  if (material.specularMap !== undefined) {
    material.specularMap = null;
  }

  if (material.specularColorMap !== undefined) {
    material.specularColorMap = null;
  }

  if (material.envMapIntensity !== undefined) {
    material.envMapIntensity = 0.28;
  }

  if (material.clearcoat !== undefined) {
    material.clearcoat = 0;
  }

  if (material.clearcoatRoughness !== undefined) {
    material.clearcoatRoughness = 1;
  }

  if (material.reflectivity !== undefined) {
    material.reflectivity = 0.22;
  }

  if (material.specularIntensity !== undefined) {
    material.specularIntensity = 0.62;
  }

  if (material.specularColor && material.specularColor.set) {
    material.specularColor.set(0xdceffc);
  }

  if (material.shininess !== undefined) {
    material.shininess = 26;
  }

  material.needsUpdate = true;
}

function applyAvatarMaterialProfile(material, modelUrl) {
  applyProductionPBRMaterialDefaults(material);

  if (DEMO_MATERIAL_FIX_MODEL_URLS.has(modelUrl)) {
    applyTemporaryDemoGLBMaterialFixes(material);
  }
}

function prepareLoadedAvatar(gltf, modelUrl) {
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
    if (obj.isMesh || obj.isSkinnedMesh) {
      obj.frustumCulled = false;

      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      materials.forEach((material) => applyAvatarMaterialProfile(material, modelUrl));
    }

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

  applyFixedAvatarCamera();

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
        const asset = prepareLoadedAvatar(gltf, modelUrl);
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

  log("Applying animation payload:", {
    animation: payload.animation,
    fps: payload.frames ? payload.frames.fps : undefined,
    boneCount: payload.frames && payload.frames.bones ? Object.keys(payload.frames.bones).length : 0,
  });
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
  const animationDuration = totalFrames / fps;
  const loopDuration = animationDuration + ANIMATION_LOOP_PAUSE_SECONDS;
  const loopTime = updateAnimation.time % loopDuration;

  if (loopTime >= animationDuration) {
    applyPoseFrame(maxFrame);
    return;
  }

  const currentFrame = Math.min(
    Math.floor(loopTime * fps),
    maxFrame
  );

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
window.applyAvatarAnimationPayload = applyAnimationPayload;
window.loadAvatarAnimationFromJson = function (animationName, framesData) {
  applyAnimationPayload({
    animation: animationName,
    camera: "start",
    frames: framesData,
  });
};

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
