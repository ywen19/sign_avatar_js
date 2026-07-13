/**
 * Owns avatar pose preparation and animation playback state.
 *
 * avatar.js provides the active rig after each model installation and calls
 * updateAnimation from the shared render loop. This module does not load
 * models, create scenes, or expose browser globals directly.
 */

let avatar = null;
let skeleton = null;
let boneNameMap = {};

let poseFrames = {};
let maxFrame = 0;
let fps = 30;
let currentAnimationName = null;
let currentAnimationPayload = null;
let isAnimationPlaying = true;
let isLoopEnabled = false;

const ANIMATION_LOOP_PAUSE_SECONDS = 3.0;

// Only prepare tracks that drive the visible signing pose. Bones omitted here
// are never changed by animation playback and remain in their model pose.
const ALLOWED_ANIMATION_BONE_NAMES = new Set([
  "Spine",
  "Spine1",
  "Spine2",
  "Neck",
  "Head",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "LeftHandThumb1",
  "LeftHandThumb2",
  "LeftHandThumb3",
  "LeftHandIndex1",
  "LeftHandIndex2",
  "LeftHandIndex3",
  "LeftHandMiddle1",
  "LeftHandMiddle2",
  "LeftHandMiddle3",
  "LeftHandRing1",
  "LeftHandRing2",
  "LeftHandRing3",
  "LeftHandPinky1",
  "LeftHandPinky2",
  "LeftHandPinky3",
  "RightHandThumb1",
  "RightHandThumb2",
  "RightHandThumb3",
  "RightHandIndex1",
  "RightHandIndex2",
  "RightHandIndex3",
  "RightHandMiddle1",
  "RightHandMiddle2",
  "RightHandMiddle3",
  "RightHandRing1",
  "RightHandRing2",
  "RightHandRing3",
  "RightHandPinky1",
  "RightHandPinky2",
  "RightHandPinky3",
]);

/**
 * Normalize Mixamo bone names for both skeleton and motion-data lookup.
 *
 * @param {string} name - Source bone name.
 * @returns {string} Normalized bone name.
 */
export function normalizeBoneName(name) {
  if (!name) return name;
  return String(name).trim().replace(/^mixamorig[:_.]?/i, "");
}

/**
 * Provide the active model rig used when applying animation frames.
 *
 * @param {THREE.Object3D|null} nextAvatar - Installed avatar object.
 * @param {THREE.Skeleton|null} nextSkeleton - Installed avatar skeleton.
 * @param {object} nextBoneNameMap - Normalized bone lookup table.
 */
export function setAnimationRig(
  nextAvatar,
  nextSkeleton,
  nextBoneNameMap
) {
  avatar = nextAvatar;
  skeleton = nextSkeleton;
  boneNameMap = nextBoneNameMap;
}

/**
 * Convert motion JSON keyframes into Three.js quaternion lookup arrays.
 *
 * @param {object} data - Motion data containing fps and keyed bone frames.
 */
function preparePoseFrames(data) {
  poseFrames = {};
  maxFrame = 0;

  if (!data || !data.bones) {
    console.error("Invalid animation data: missing bones");
    return;
  }

  for (const name in data.bones) {
    const normalizedName = normalizeBoneName(name);
    if (!ALLOWED_ANIMATION_BONE_NAMES.has(normalizedName)) continue;

    const frames = [];
    const keyframes = data.bones[name];

    keyframes.forEach((keyframe) => {
      frames[keyframe.f] = new THREE.Quaternion(
        keyframe.rot[0],
        keyframe.rot[1],
        keyframe.rot[2],
        keyframe.rot[3]
      );

      if (keyframe.f > maxFrame) maxFrame = keyframe.f;
    });

    poseFrames[normalizedName] = frames;
  }

  console.log(
    "[AVATAR] Prepared animation. fps =",
    fps,
    "maxFrame =",
    maxFrame
  );
}

/**
 * Apply one prepared motion frame to every matching skeleton bone.
 *
 * @param {number} frame - Frame index to apply.
 */
function applyPoseFrame(frame) {
  if (!avatar || !skeleton) return;

  for (const name in poseFrames) {
    const bone = boneNameMap[name];
    if (!bone) continue;

    const quaternion = poseFrames[name][frame];
    if (!quaternion) continue;

    bone.quaternion.copy(quaternion);
  }

  avatar.updateMatrixWorld(true);
}

/**
 * Restart the current motion from its first frame.
 */
export function playAnimationFromStart() {
  updateAnimation.time = 0;
  isAnimationPlaying = true;
  applyPoseFrame(0);
}

/**
 * Pause frame advancement while preserving the current avatar pose.
 */
export function pauseAnimationPlayback() {
  isAnimationPlaying = false;
}

/**
 * Reset playback using the same behavior as an explicit replay command.
 */
export function resetAnimationPlayback() {
  playAnimationFromStart();
}

/**
 * Enable or disable looping while preserving the existing timing behavior.
 *
 * @param {boolean} enabled - Requested loop state.
 * @returns {boolean} Applied loop state.
 */
export function setAnimationLoopEnabled(enabled) {
  const nextLoopEnabled = Boolean(enabled);
  const wasLoopEnabled = isLoopEnabled;
  isLoopEnabled = nextLoopEnabled;

  if (maxFrame) {
    const totalFrames = maxFrame + 1;
    const animationDuration = totalFrames / fps;
    const loopDuration =
      animationDuration + ANIMATION_LOOP_PAUSE_SECONDS;

    if (
      !isLoopEnabled &&
      wasLoopEnabled &&
      updateAnimation.time !== undefined
    ) {
      const loopTime = updateAnimation.time % loopDuration;
      updateAnimation.time =
        loopTime < animationDuration ? loopTime : 0;
      isAnimationPlaying = true;

      if (loopTime >= animationDuration) {
        applyPoseFrame(0);
      }
    }

    if (isLoopEnabled && !isAnimationPlaying) {
      isAnimationPlaying = true;

      if (
        updateAnimation.time === undefined ||
        updateAnimation.time < animationDuration
      ) {
        updateAnimation.time = animationDuration;
      }
    }
  }

  return isLoopEnabled;
}

/**
 * Return whether playback looping is enabled.
 *
 * @returns {boolean} Current loop state.
 */
export function getAnimationLoopEnabled() {
  return isLoopEnabled;
}

/**
 * Prepare and start an animation payload returned by the backend.
 *
 * @param {object} payload - Animation metadata and frame data.
 */
export function applyAnimationPayload(payload) {
  if (!payload) {
    console.error("applyAnimationPayload: payload is empty");
    return;
  }

  console.log("[AVATAR] Applying animation payload:", {
    animation: payload.animation,
    fps: payload.frames ? payload.frames.fps : undefined,
    boneCount:
      payload.frames && payload.frames.bones
        ? Object.keys(payload.frames.bones).length
        : 0,
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
  playAnimationFromStart();

  console.log("[AVATAR] Animation applied:", currentAnimationName);
}

/**
 * Reapply the current payload after avatar model installation.
 *
 * @returns {boolean} Whether a current payload was available.
 */
export function reapplyCurrentAnimation() {
  if (!currentAnimationPayload) return false;

  applyAnimationPayload(currentAnimationPayload);
  return true;
}

/**
 * Advance animation timing and apply the frame for the current time.
 *
 * @param {number} deltaTime - Seconds elapsed since the previous render frame.
 */
export function updateAnimation(deltaTime) {
  if (!avatar || !skeleton || !maxFrame || !isAnimationPlaying) return;

  if (updateAnimation.time === undefined) updateAnimation.time = 0;
  updateAnimation.time += deltaTime;

  const totalFrames = maxFrame + 1;
  const animationDuration = totalFrames / fps;

  if (!isLoopEnabled && updateAnimation.time >= animationDuration) {
    applyPoseFrame(maxFrame);
    isAnimationPlaying = false;
    return;
  }

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

/**
 * Return the name of the animation currently loaded for playback.
 *
 * @returns {string|null} Current animation name.
 */
export function getCurrentAnimationName() {
  return currentAnimationName;
}
