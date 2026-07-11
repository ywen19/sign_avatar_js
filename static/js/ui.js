    const IS_BETA = false;
    const appEl = document.querySelector(".app");
    if (appEl) {
      appEl.classList.toggle("beta-mode", IS_BETA);
      appEl.classList.toggle("release-mode", !IS_BETA);
    }

    const debugLogEl = document.getElementById("debug-log");

    function logDebug(...args) {
      const msg = args.map(v => {
        try {
          if (typeof v === "string") return v;
          return JSON.stringify(v);
        } catch {
          return String(v);
        }
      }).join(" ");

      console.log("[INDEX]", ...args);

      if (debugLogEl) {
        debugLogEl.textContent += "\\n" + msg;
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
    const AVATAR_MODEL_STORAGE_KEY = "sign-demo-avatar-model";

    function normalizeAvatarModelPath(model) {
      if (!model) return "";
      return model.startsWith("/") ? model : `/models/${model}`;
    }

    function getStoredAvatarModelPath() {
      try {
        return window.localStorage.getItem(AVATAR_MODEL_STORAGE_KEY) || "";
      } catch {
        return "";
      }
    }

    function setAvatarPopoverOpen(isOpen) {
      if (!avatarPopover || !avatarPickerBtn) return;
      avatarPopover.classList.toggle("open", isOpen);
      avatarPopover.setAttribute("aria-hidden", String(!isOpen));
      avatarPickerBtn.setAttribute("aria-expanded", String(isOpen));
    }

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
        }

        choice.addEventListener("click", async () => {
          setAvatarPopoverOpen(false);

          if (!window.switchAvatarModel) {
            logDebug("avatar switch failed = switchAvatarModel is not ready");
            return;
          }

          try {
            logDebug("avatar switch requested =", avatarId, modelPath);
            const switched = await window.switchAvatarModel(modelPath);

            if (switched === false) {
              return;
            }

            avatarPopover.querySelectorAll(".avatar-choice").forEach((item) => {
              item.classList.remove("active");
            });
            choice.classList.add("active");
            try {
              window.localStorage.setItem(AVATAR_MODEL_STORAGE_KEY, modelPath);
            } catch {}
            logDebug("avatar switched =", avatarId, avatar.model);
          } catch (error) {
            logDebug("avatar switch failed =", String(error));
          }
        });

        avatarPopover.appendChild(choice);
      });
    }

    async function loadAvatarChoices() {
      try {
        const response = await fetch("/models/avatars.json", { cache: "no-store" });
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
          logDebug("avatar replay failed = resetAnimationPlayback is not ready");
        }
      });
    }

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
      logDebug("window error:", e.message, "@", e.filename + ":" + e.lineno);
    });

    window.addEventListener("unhandledrejection", (e) => {
      logDebug("unhandled rejection:", String(e.reason));
    });
