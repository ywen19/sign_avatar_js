console.log("AVATAR JS LOADED");

let scene, camera, renderer, avatar;
let skeleton = null;
const boneNameMap = {};

let poseFrames = {};
let maxFrame = 0;
let fps = 30;
let currentAnimationName = null;

const clock = new THREE.Clock();
const stageEl = document.getElementById("avatar-stage");

function normalizeBoneName(name) {
  if (!name) return name;
  return name.replace(/^mixamorig[:_]?/i, "");
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

function loadModel() {
  const url = "/models/model.glb";
  log("Trying to load model from:", url);

  const loader = new THREE.GLTFLoader();

  loader.load(
    url,
    async function (gltf) {
      avatar = gltf.scene;
      avatar.scale.set(100, 100, 100);
      scene.add(avatar);

      const bbox = new THREE.Box3().setFromObject(avatar);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      bbox.getSize(size);
      bbox.getCenter(center);

      log("Model bbox size:", size);
      log("Model bbox center:", center);

      avatar.position.sub(center);

      avatar.traverse((obj) => {
        if (obj.isSkinnedMesh && !skeleton) {
          skeleton = obj.skeleton;
        }
      });

      if (!skeleton) {
        console.warn("No skeleton found!");
      } else {
        skeleton.bones.forEach((bone) => {
          const key = normalizeBoneName(bone.name);
          boneNameMap[key] = bone;
        });
        log("Skeleton bones normalized:", Object.keys(boneNameMap));
      }

      log("GLB loaded successfully.");

      // 模型加载成功后，再从后端拿默认动画
      await fetchDefaultAnimationFromBackend();
    },
    undefined,
    function (err) {
      console.error("GLB load failed:", err);
    }
  );
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

function applyAnimationPayload(payload) {
  if (!payload) {
    console.error("applyAnimationPayload: payload is empty");
    return;
  }

  log("Applying payload:", payload);

  const framesData = payload.frames;
  if (!framesData) {
    console.error("Payload missing frames field");
    return;
  }

  fps = framesData.fps || 30;
  currentAnimationName = payload.animation || "unknown";

  preparePoseFrames(framesData);

  // 切换动画时从第一帧开始播
  updateAnimation.time = 0;

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

  for (const name in poseFrames) {
    const bone = boneNameMap[name];
    if (!bone) continue;

    const q = poseFrames[name][currentFrame];
    if (!q) continue;

    bone.quaternion.copy(q);
  }

  avatar.updateMatrixWorld(true);
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

window.fetchEndAnimationFromBackend = fetchEndAnimationFromBackend;

window.getCurrentAnimationName = function () {
  return currentAnimationName;
};

window.addEventListener("error", function (e) {
  console.error("Global JS error:", e.message, "@", e.filename + ":" + e.lineno);
});

window.addEventListener("unhandledrejection", function (e) {
  console.error("Unhandled promise rejection:", e.reason);
});

log("Page loaded. href =", window.location.href);
log("THREE version:", THREE.REVISION);

if (!THREE.GLTFLoader) {
  console.error("GLTFLoader NOT detected!");
} else {
  log("GLTFLoader detected.");
}

init();
loadModel();
animate();