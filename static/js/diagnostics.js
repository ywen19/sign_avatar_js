logDebug("three.min.js tag executed");
logDebug("window.THREE exists =", !!window.THREE);
if (window.THREE) {
  logDebug("THREE.REVISION =", window.THREE.REVISION);
}

logDebug("GLTFLoader.js tag executed");
logDebug(
  "THREE.GLTFLoader exists =",
  !!(window.THREE && window.THREE.GLTFLoader)
);

logDebug("camera.js module tag added");
logDebug("text_input.js module tag added");
logDebug("avatar.js module tag added");
