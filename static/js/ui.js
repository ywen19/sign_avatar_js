/**
 * Coordinates the BSL Avatar page controls and frontend diagnostics.
 *
 * This classic script configures beta visibility, renders the avatar picker,
 * forwards replay and loop commands to avatar.js through its public window
 * APIs, and exposes logDebug for diagnostics.js.
 */

const IS_BETA = false;
const appEl = document.querySelector(".app");
if (appEl) {
  appEl.classList.toggle("beta-mode", IS_BETA);
  appEl.classList.toggle("release-mode", !IS_BETA);
}

const debugLogEl = document.getElementById("debug-log");

/**
 * Write diagnostic values to the console and the optional beta debug panel.
 *
 * This function remains global because diagnostics.js calls it after loading.
 *
 * @param {...unknown} args - Values to format and log.
 */
function logDebug(...args) {
  const msg = args
    .map((value) => {
      try {
        if (typeof value === "string") return value;
        return JSON.stringify(value);
      } catch {
        return String(value);
      }
    })
    .join(" ");

  console.log("[INDEX]", ...args);

  if (debugLogEl) {
    debugLogEl.textContent += "\n" + msg;
    debugLogEl.scrollTop = debugLogEl.scrollHeight;
  }
}

if (debugLogEl) {
  debugLogEl.textContent = "index.html loaded";
}
logDebug("index.html loaded");
logDebug("href =", window.location.href);

const avatarPickerBtn = document.getElementById("avatar-picker-btn");
const avatarReplayBtn = document.getElementById("avatar-replay-btn");
const avatarLoopBtn = document.getElementById("avatar-loop-btn");
const avatarPopover = document.getElementById("avatar-popover");
const AVATAR_MODEL_STORAGE_KEY = "sign-demo-avatar-model-v4";
let isSwitchingAvatar = false;

/**
 * Resolve a manifest model or snapshot value to an application asset path.
 *
 * @param {string} model - Absolute or model-directory-relative path.
 * @returns {string} Normalized asset path, or an empty string when omitted.
 */
function normalizeAvatarModelPath(model) {
  if (!model) return "";
  return model.startsWith("/") ? model : `/models/${model}`;
}

/**
 * Read the previously selected avatar path without failing in restricted
 * browser-storage environments.
 *
 * @returns {string} Stored model path, or an empty string when unavailable.
 */
function getStoredAvatarModelPath() {
  try {
    return window.localStorage.getItem(AVATAR_MODEL_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

/**
 * Synchronize popover visibility with its accessibility attributes.
 *
 * @param {boolean} isOpen - Whether the avatar picker should be visible.
 */
function setAvatarPopoverOpen(isOpen) {
  if (!avatarPopover || !avatarPickerBtn) return;
  avatarPopover.classList.toggle("open", isOpen);
  avatarPopover.setAttribute("aria-hidden", String(!isOpen));
  avatarPickerBtn.setAttribute("aria-expanded", String(isOpen));
}

/**
 * Mark one avatar choice as selected visually and for assistive technology.
 *
 * @param {HTMLButtonElement} activeChoice - Choice that became active.
 */
function setActiveAvatarChoice(activeChoice) {
  if (!avatarPopover) return;

  avatarPopover.querySelectorAll(".avatar-choice").forEach((choice) => {
    const isActive = choice === activeChoice;
    choice.classList.toggle("active", isActive);
    choice.setAttribute("aria-pressed", String(isActive));
  });
}

/**
 * Build the avatar picker buttons from the backend-served manifest.
 *
 * @param {object} avatars - Avatar manifest entries keyed by avatar ID.
 */
function renderAvatarChoices(avatars) {
  if (!avatarPopover) return;

  avatarPopover.replaceChildren();

  const storedModelPath = getStoredAvatarModelPath();

  Object.entries(avatars).forEach(([avatarId, avatar]) => {
    const choice = document.createElement("button");
    choice.className = "avatar-choice";
    choice.type = "button";
    const modelPath = normalizeAvatarModelPath(avatar.model);

    const snapshotPath = normalizeAvatarModelPath(avatar.snapshot);
    const label = avatar.label || avatarId;

    choice.dataset.avatar = avatarId;
    choice.dataset.model = avatar.model || "";
    choice.dataset.snapshot = avatar.snapshot || "";
    choice.setAttribute("aria-label", label);
    choice.setAttribute("aria-pressed", "false");

    if (snapshotPath) {
      const image = document.createElement("img");
      image.className = "avatar-choice-img";
      image.src = snapshotPath;
      image.alt = "";
      image.loading = "lazy";
      choice.appendChild(image);
    }

    if (storedModelPath ? modelPath === storedModelPath : avatar.default) {
      choice.classList.add("active");
      choice.setAttribute("aria-pressed", "true");
    }

    choice.addEventListener("click", async () => {
      setAvatarPopoverOpen(false);

      if (isSwitchingAvatar) {
        logDebug("avatar switch ignored = another switch is active");
        return;
      }

      if (!window.switchAvatarModel) {
        logDebug("avatar switch failed = switchAvatarModel is not ready");
        return;
      }

      isSwitchingAvatar = true;

      try {
        logDebug("avatar switch requested =", avatarId, modelPath);
        const switched = await window.switchAvatarModel(modelPath);

        if (switched === false) {
          return;
        }

        setActiveAvatarChoice(choice);

        try {
          window.localStorage.setItem(AVATAR_MODEL_STORAGE_KEY, modelPath);
        } catch {
          // Avatar switching still succeeds when browser storage is blocked.
        }
        logDebug("avatar switched =", avatarId, avatar.model);
      } catch (error) {
        logDebug("avatar switch failed =", String(error));
      } finally {
        isSwitchingAvatar = false;
      }
    });

    avatarPopover.appendChild(choice);
  });
}

/**
 * Fetch the avatar manifest and render its available picker choices.
 *
 * @returns {Promise<void>}
 */
async function loadAvatarChoices() {
  try {
    const response = await fetch("/models/avatars.json", {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`avatars.json HTTP ${response.status}`);
    }

    const avatars = await response.json();
    renderAvatarChoices(avatars);
    logDebug("avatar choices loaded =", Object.keys(avatars).join(", "));
  } catch (error) {
    logDebug("avatar choices failed =", String(error));
  }
}

if (avatarPickerBtn && avatarPopover) {
  avatarPickerBtn.setAttribute("aria-haspopup", "true");
  avatarPickerBtn.setAttribute("aria-expanded", "false");

  avatarPickerBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    setAvatarPopoverOpen(!avatarPopover.classList.contains("open"));
  });

  avatarPopover.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  document.addEventListener("click", () => {
    setAvatarPopoverOpen(false);
  });

  loadAvatarChoices();
}

if (avatarReplayBtn) {
  avatarReplayBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    setAvatarPopoverOpen(false);

    if (window.resetAnimationPlayback) {
      window.resetAnimationPlayback();
      logDebug("avatar animation replayed");
    } else {
      logDebug(
        "avatar replay failed = resetAnimationPlayback is not ready"
      );
    }
  });
}

/**
 * Synchronize the loop checkbox with the playback state owned by avatar.js.
 */
function syncLoopButtonState() {
  if (!avatarLoopBtn) return;

  const enabled = window.getAnimationLoopEnabled
    ? window.getAnimationLoopEnabled()
    : false;
  avatarLoopBtn.checked = enabled;
}

if (avatarLoopBtn) {
  avatarLoopBtn.addEventListener("change", (event) => {
    event.stopPropagation();
    setAvatarPopoverOpen(false);

    if (window.setAnimationLoopEnabled) {
      const enabled = window.setAnimationLoopEnabled(avatarLoopBtn.checked);
      syncLoopButtonState();
      logDebug("avatar loop enabled =", enabled);
    } else {
      logDebug("avatar loop toggle failed = loop API is not ready");
    }
  });

  syncLoopButtonState();
}

window.addEventListener("error", (e) => {
  const location = `${e.filename}:${e.lineno}`;
  logDebug("window error:", e.message, "@", location);
});

window.addEventListener("unhandledrejection", (e) => {
  logDebug("unhandled rejection:", String(e.reason));
});
