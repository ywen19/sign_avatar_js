/**
 * Owns the Three.js scene used by the BSL avatar.
 *
 * This module creates the camera, renderer, and lighting, keeps the renderer
 * sized to the avatar stage, and renders frames requested by avatar.js.
 */

let scene = null;
let camera = null;
let renderer = null;

const stageEl = document.getElementById("avatar-stage");

/**
 * Position the camera for the fixed full-body avatar view.
 */
export function applyFixedAvatarCamera() {
  if (!camera) return;

  // Frame the signing area while cropping unnecessary lower-body space.
  const target = new THREE.Vector3(0, 14.1, 0);
  const distance = 16.2;
  const cameraHeight = target.y - 0.6;
  camera.position.set(target.x, cameraHeight, target.z + distance);
  camera.lookAt(target);
}

/**
 * Resize the camera projection and renderer to the current stage dimensions.
 */
function onWindowResize() {
  if (!renderer || !camera || !stageEl) return;

  const width = stageEl.clientWidth || 800;
  const height = stageEl.clientHeight || 600;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

/**
 * Initialize the avatar scene, camera, renderer, lighting, and resize signal.
 *
 * @returns {boolean} Whether the scene initialized successfully.
 */
export function initAvatarScene() {
  console.log("[AVATAR] stageEl =", stageEl);

  if (!stageEl) {
    console.error("avatar-stage container not found.");
    return false;
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
  return true;
}

/**
 * Return the initialized scene used to install and remove avatar models.
 *
 * @returns {THREE.Scene|null} Current scene, or null before initialization.
 */
export function getAvatarScene() {
  return scene;
}

/**
 * Render the current scene when initialization has completed.
 */
export function renderAvatarScene() {
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}
