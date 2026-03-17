let scene, camera, renderer, avatar;
let skeleton = null;
const boneNameMap = {};

let animData = null;
let poseFrames = {};
let maxFrame = 0;
let fps = 30;

const clock = new THREE.Clock();
const stageEl = document.getElementById("avatar-stage");

function normalizeBoneName(name) {
  if (!name) return name;
  return name.replace(/^mixamorig[:_]?/i, "");
}

function init() {
  if (!stageEl) {
    console.error("avatar-stage container not found.");
    return;
  }

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202020);

  const width = stageEl.clientWidth || 800;
  const height = stageEl.clientHeight || 600;

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 1.6, 3);

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
  const url = "model.glb";
  console.log("Trying to load model from:", url);

  const loader = new THREE.GLTFLoader();

  loader.load(
    url,
    function (gltf) {
      avatar = gltf.scene;

      avatar.scale.set(100, 100, 100);
      scene.add(avatar);

      const bbox = new THREE.Box3().setFromObject(avatar);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      bbox.getSize(size);
      bbox.getCenter(center);

      console.log("Model bbox size:", size);
      console.log("Model bbox center:", center);

      avatar.position.sub(center);

      const targetY = 8.0;
      const distance = 23.0;
      camera.position.set(0, targetY, distance);
      camera.lookAt(0, targetY, 0);

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
        console.log("Skeleton bones normalized:", Object.keys(boneNameMap));
      }

      console.log("GLB loaded successfully.");
    },
    undefined,
    function (err) {
      console.error("GLB load failed:", err);
    }
  );
}

function loadAnimation() {
  const jsonUrl = "Dancing_mixamo_com_frames.json";
  console.log("Loading JSON animation from:", jsonUrl);

  fetch(jsonUrl)
    .then((res) => {
      console.log("JSON fetch status =", res.status);
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then((data) => {
      console.log("JSON animation loaded OK.");
      animData = data;
      fps = data.fps || fps;
      preparePoseFrames(data);
    })
    .catch((err) => {
      console.error("Load JSON animation failed:", err);
    });
}

function preparePoseFrames(data) {
  poseFrames = {};
  maxFrame = 0;

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

  console.log("Prepared JSON animation. fps =", fps, "maxFrame =", maxFrame);
}

function updateAnimation(dt) {
  if (!avatar || !animData || !skeleton) return;

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

window.addEventListener("error", function (e) {
  console.error("Global JS error:", e.message, "@", e.filename + ":" + e.lineno);
});

window.addEventListener("unhandledrejection", function (e) {
  console.error("Unhandled promise rejection:", e.reason);
});

console.log("Page loaded. href =", window.location.href);
console.log("THREE version:", THREE.REVISION);

if (!THREE.GLTFLoader) {
  console.error("GLTFLoader NOT detected!");
} else {
  console.log("GLTFLoader detected.");
}

init();
loadModel();
loadAnimation();
animate();