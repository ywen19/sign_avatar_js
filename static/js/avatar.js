/**
 * Orchestrates avatar model loading and the public frontend avatar API.
 *
 * This module owns GLB loading, material preparation, skeleton discovery,
 * avatar switching, manifest startup, and coordination with the scene and
 * animation modules. Other frontend scripts call the stable window APIs
 * registered near the end of this file.
 */

import {
  applyAnimationPayload,
  getAnimationLoopEnabled,
  getCurrentAnimationName,
  normalizeBoneName,
  pauseAnimationPlayback,
  playAnimationFromStart,
  reapplyCurrentAnimation,
  resetAnimationPlayback,
  setAnimationLoopEnabled,
  setAnimationRig,
  updateAnimation,
} from "./avatar_animation.js";
import {
  applyFixedAvatarCamera,
  getAvatarScene,
  initAvatarScene,
  renderAvatarScene,
} from "./avatar_scene.js?v=camera-angle-20260713";

console.log("AVATAR JS LOADED");

// Active model state shared with the animation module after installation.
let avatar = null;
let skeleton = null;
const boneNameMap = {};

// Track the selected model and reject stale asynchronous model switches.
let currentModelUrl = null;
let modelLoadRequestId = 0;

const DEFAULT_MODEL_URL = "/models/woman.glb";
const DEMO_MATERIAL_FIX_MODEL_URLS = new Set([
  "/models/man.glb",
  "/models/woman.glb",
]);
const AVATAR_MODEL_STORAGE_KEY = "sign-demo-avatar-model-v4";
const DEFAULT_AVATAR_SCENE_SCALE = 10;
const STANDARD_HIPS_WORLD_Y = 10.089534521102905;

// Deduplicate only simultaneous requests. Completed GLBs are deliberately not
// retained so a replaced model file is fetched again on the next switch.
const modelAssetPromises = new Map();
const modelDisplayScaleMultipliers = new Map();
let avatarManifestPromise = null;
let modelRequestSequence = 0;

const clock = new THREE.Clock();

/**
 * Write a consistently prefixed avatar diagnostic message.
 *
 * @param {...unknown} args - Values to write to the browser console.
 */
function log(...args) {
  console.log("[AVATAR]", ...args);
}

/**
 * Remove bone references belonging to the previously installed skeleton.
 */
function clearBoneNameMap() {
  Object.keys(boneNameMap).forEach((key) => {
    delete boneNameMap[key];
  });
}

/**
 * Resolve a model filename or path to an application model URL.
 *
 * @param {string} url - Absolute or model-directory-relative URL.
 * @returns {string} Normalized model URL.
 */
function normalizeModelUrl(url) {
  if (!url) return DEFAULT_MODEL_URL;
  return url.startsWith("/") ? url : `/models/${url}`;
}

/**
 * Read the previously selected model from browser storage.
 *
 * @returns {string|null} Stored model URL, or null when unavailable.
 */
function getStoredAvatarModelUrl() {
  try {
    const stored = window.localStorage.getItem(AVATAR_MODEL_STORAGE_KEY);
    return stored ? normalizeModelUrl(stored) : null;
  } catch (err) {
    console.warn("Could not read stored avatar model:", err);
    return null;
  }
}

/**
 * Persist the installed avatar so startup can restore the same selection.
 */
function rememberCurrentAvatarModel() {
  if (!currentModelUrl) return;

  try {
    window.localStorage.setItem(AVATAR_MODEL_STORAGE_KEY, currentModelUrl);
  } catch (err) {
    console.warn("Could not store avatar model:", err);
  }
}

// Material preparation -----------------------------------------------------

/**
 * Apply defaults intended for final production PBR avatar materials.
 *
 * @param {THREE.Material|null} material - Material to prepare.
 */
function applyProductionPBRMaterialDefaults(material) {
  if (!material) return;

  // Production GLBs from the outsourcing team should use the standard PBR path.
  // Keep base color, bump/normal, alpha, and roughness maps intact here.
  if (material.envMapIntensity !== undefined) {
    material.envMapIntensity = 1;
  }

  material.needsUpdate = true;
}

/**
 * Correct known roughness and specular issues in the temporary demo GLBs.
 *
 * @param {THREE.Material|null} material - Demo material to correct.
 */
function applyTemporaryDemoGLBMaterialFixes(material) {
  if (!material) return;

  // TODO: Remove this function when the final outsourced avatar models arrive.
  // The current demo GLBs contain KHR_materials_specular with a
  // specularColorFactor of [2, 2, 2]
  // A texture named "Glossiness" is also wired into the glTF
  // metallicRoughnessTexture slot.
  // Three.js multiplies roughness by the green channel of roughnessMap. These
  // files have low green-channel values, so clothes retain strong baked shine
  // even with high material roughness. Limit these overrides to the demo
  // models so final PBR assets can keep their proper maps.
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

/**
 * Apply the production profile and any model-specific temporary corrections.
 *
 * @param {THREE.Material|null} material - Material to prepare.
 * @param {string} modelUrl - Normalized source model URL.
 */
function applyAvatarMaterialProfile(material, modelUrl) {
  applyProductionPBRMaterialDefaults(material);

  if (DEMO_MATERIAL_FIX_MODEL_URLS.has(modelUrl)) {
    applyTemporaryDemoGLBMaterialFixes(material);
  }
}

// Model preparation and loading -------------------------------------------

/**
 * Scale, vertically align, prepare materials, and discover a loaded GLB rig.
 *
 * @param {object} gltf - Parsed GLTFLoader result.
 * @param {string} modelUrl - Normalized source model URL.
 * @returns {{nextAvatar: THREE.Object3D, nextSkeleton: THREE.Skeleton|null}}
 * Prepared model rig.
 */
function prepareLoadedAvatar(gltf, modelUrl) {
  const nextAvatar = gltf.scene;
  let nextSkeleton = null;
  const displayScaleMultiplier =
    modelDisplayScaleMultipliers.get(modelUrl) || 1;
  const sceneScale =
    DEFAULT_AVATAR_SCENE_SCALE * displayScaleMultiplier;

  nextAvatar.scale.set(sceneScale, sceneScale, sceneScale);

  const bbox = new THREE.Box3().setFromObject(nextAvatar);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bbox.getSize(size);
  bbox.getCenter(center);

  log("Model bbox size:", size.toArray());
  log("Model bbox center:", center.toArray());
  log("Model display scale:", sceneScale);

  // These rigs are authored against the shared scene origin. Box3 in this
  // Three.js version measures bind-space vertices for SkinnedMesh objects, so
  // using its center would move models with different bind transforms by
  // different depths even when their rendered meshes are aligned.
  nextAvatar.position.set(0, 0, 0);

  nextAvatar.traverse((obj) => {
    if (obj.isMesh || obj.isSkinnedMesh) {
      obj.frustumCulled = false;

      const materials = Array.isArray(obj.material)
        ? obj.material
        : [obj.material];
      materials.forEach((material) => {
        applyAvatarMaterialProfile(material, modelUrl);
      });
    }

    if (obj.isSkinnedMesh && !nextSkeleton) {
      nextSkeleton = obj.skeleton;
    }
  });

  if (nextSkeleton) {
    const hips = nextSkeleton.bones.find(
      (bone) => normalizeBoneName(bone.name) === "Hips"
    );

    if (hips) {
      nextAvatar.updateMatrixWorld(true);
      const hipsWorldPosition = new THREE.Vector3();
      hips.getWorldPosition(hipsWorldPosition);
      const verticalOffset = STANDARD_HIPS_WORLD_Y - hipsWorldPosition.y;
      nextAvatar.position.y += verticalOffset;
      nextAvatar.updateMatrixWorld(true);
      log("Hips source Y and vertical offset:", [
        hipsWorldPosition.y,
        verticalOffset,
      ]);
    }
  }

  return { nextAvatar, nextSkeleton };
}

/**
 * Replace the current scene model and publish its normalized rig to animation.
 *
 * @param {THREE.Object3D} nextAvatar - Prepared avatar object.
 * @param {THREE.Skeleton|null} nextSkeleton - Discovered model skeleton.
 */
function installLoadedAvatar(nextAvatar, nextSkeleton) {
  const scene = getAvatarScene();
  if (!scene) {
    throw new Error("Avatar scene is not initialized.");
  }

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

  // Signal the animation module only after the complete rig map is ready.
  setAnimationRig(avatar, skeleton, boneNameMap);
}

/**
 * Fetch and prepare a fresh avatar asset with GLTFLoader.
 *
 * @param {string} url - Model path to load.
 * @returns {Promise<object>} Prepared avatar and skeleton asset.
 */
function loadAvatarAsset(url) {
  const modelUrl = normalizeModelUrl(url);

  if (modelAssetPromises.has(modelUrl)) {
    // Share an in-flight load instead of requesting the same GLB again.
    return modelAssetPromises.get(modelUrl);
  }

  log("Loading avatar asset:", modelUrl);

  const loader = new THREE.GLTFLoader();
  const separator = modelUrl.includes("?") ? "&" : "?";
  const requestUrl = `${modelUrl}${separator}reload=${Date.now()}-${++modelRequestSequence}`;
  const promise = new Promise((resolve, reject) => {
    loader.load(
      requestUrl,
      function (gltf) {
        const asset = prepareLoadedAvatar(gltf, modelUrl);
        modelAssetPromises.delete(modelUrl);
        log("Fresh avatar asset loaded:", modelUrl);
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

/**
 * Load and install one avatar while ignoring stale asynchronous switch results.
 *
 * @param {string} url - Requested model path.
 * @param {object} options - Model installation options.
 * @param {boolean} options.fetchDefaultAnimation - Fetch startup animation.
 * @returns {Promise<boolean>} Whether this request installed its model.
 */
async function loadModel(url = DEFAULT_MODEL_URL, options = {}) {
  const modelUrl = normalizeModelUrl(url);
  const requestId = ++modelLoadRequestId;
  const shouldFetchDefaultAnimation = Boolean(options.fetchDefaultAnimation);

  log("Trying to load model from:", modelUrl);

  try {
    const { nextAvatar, nextSkeleton } = await loadAvatarAsset(modelUrl);

    // A newer picker request owns installation when request IDs differ.
    if (requestId !== modelLoadRequestId) {
      return false;
    }

    installLoadedAvatar(nextAvatar, nextSkeleton);
    currentModelUrl = modelUrl;
    rememberCurrentAvatarModel();

    log("GLB installed successfully.");

    if (shouldFetchDefaultAnimation) {
      await fetchDefaultAnimationFromBackend();
    } else if (!reapplyCurrentAnimation()) {
      resetAnimationPlayback();
    }

    return true;
  } catch (err) {
    console.error("Avatar model switch failed:", err);
    throw err;
  }
}

// Avatar manifest ---------------------------------------------------------

/**
 * Fetch the avatar manifest for startup and avatar selection.
 *
 * @returns {Promise<object>} Avatar manifest entries keyed by avatar ID.
 */
async function loadAvatarManifest() {
  if (!avatarManifestPromise) {
    avatarManifestPromise = fetch("/models/avatars.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`avatars.json HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((avatars) => {
        modelDisplayScaleMultipliers.clear();

        Object.values(avatars).forEach((avatarData) => {
          const modelUrl = normalizeModelUrl(avatarData.model);
          const multiplier = Number(avatarData.displayScale);

          if (Number.isFinite(multiplier) && multiplier > 0) {
            modelDisplayScaleMultipliers.set(modelUrl, multiplier);
          }
        });

        return avatars;
      });
  }

  return avatarManifestPromise;
}

/**
 * Resolve the manifest default model, falling back to the built-in URL.
 *
 * @returns {Promise<string>} Normalized default model URL.
 */
async function getManifestDefaultAvatarModelUrl() {
  try {
    const avatars = await loadAvatarManifest();
    const defaultEntry = Object.values(avatars).find(
      (avatarData) => avatarData.default
    );
    return defaultEntry
      ? normalizeModelUrl(defaultEntry.model)
      : DEFAULT_MODEL_URL;
  } catch (err) {
    console.warn("Avatar manifest default failed:", err);
    return DEFAULT_MODEL_URL;
  }
}

/**
 * Prefer the stored selection, then use the manifest default model.
 *
 * @returns {Promise<string>} Initial model URL for application startup.
 */
async function getInitialAvatarModelUrl() {
  const defaultModelUrl = await getManifestDefaultAvatarModelUrl();
  return getStoredAvatarModelUrl() || defaultModelUrl;
}

// Backend animation requests ----------------------------------------------

/**
 * Fetch and apply the backend's initial animation payload.
 *
 * @returns {Promise<void>}
 */
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

/**
 * Fetch and apply the legacy end-animation payload.
 *
 * @returns {Promise<void>}
 */
async function fetchEndAnimationFromBackend() {
  try {
    log("Fetching end animation from /api/end");
    const res = await fetch("/api/end", {
      method: "POST",
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

// Render-loop orchestration ------------------------------------------------

/**
 * Advance animation state and render one scene frame on each browser tick.
 */
function animate() {
  requestAnimationFrame(animate);

  const dt = clock.getDelta();
  updateAnimation(dt);

  renderAvatarScene();
}

// Public frontend API ------------------------------------------------------

// ui.js uses this signal to request an asynchronously loaded avatar model.
window.switchAvatarModel = async function (modelUrl) {
  return loadModel(modelUrl, { fetchDefaultAnimation: false });
};

// ui.js and text_input.js use these stable animation control functions.
window.resetAnimationPlayback = resetAnimationPlayback;
window.playAnimationFromStart = playAnimationFromStart;
window.pauseAnimationPlayback = pauseAnimationPlayback;
window.setAnimationLoopEnabled = setAnimationLoopEnabled;
window.getAnimationLoopEnabled = getAnimationLoopEnabled;
window.applyAvatarAnimationPayload = applyAnimationPayload;
window.loadAvatarAnimationFromJson = function (animationName, framesData) {
  applyAnimationPayload({
    animation: animationName,
    camera: "start",
    frames: framesData,
  });
};

window.getCurrentAnimationName = getCurrentAnimationName;

// Keep console diagnostics for errors outside local request error handling.
window.addEventListener("error", function (e) {
  const location = `${e.filename}:${e.lineno}`;
  console.error("Global JS error:", e.message, "@", location);
});

window.addEventListener("unhandledrejection", function (e) {
  console.error("Unhandled promise rejection:", e.reason);
});

window.addEventListener("beforeunload", rememberCurrentAvatarModel);

// Startup ------------------------------------------------------------------

log("Page loaded. href =", window.location.href);
log("THREE version:", THREE.REVISION);

if (!THREE.GLTFLoader) {
  console.error("GLTFLoader NOT detected!");
} else {
  log("GLTFLoader detected.");
}

/**
 * Restore the preferred avatar and fall back to the built-in model on failure.
 *
 * @returns {Promise<void>}
 */
async function startAvatarApp() {
  const initialModelUrl = await getInitialAvatarModelUrl();

  try {
    await loadModel(initialModelUrl, { fetchDefaultAnimation: true });
  } catch (err) {
    console.error("Initial avatar load failed:", err);

    if (initialModelUrl !== DEFAULT_MODEL_URL) {
      try {
        await loadModel(DEFAULT_MODEL_URL, { fetchDefaultAnimation: true });
      } catch (fallbackErr) {
        console.error("Fallback avatar load failed:", fallbackErr);
      }
    }
  }
}

// Initialize scene state before model installation and frame rendering.
initAvatarScene();
startAvatarApp();
animate();
