const textInput = document.getElementById("text-input");
const sendTextBtn = document.getElementById("send-text-btn");
const cameraStatus = document.getElementById("camera-status");
const subtitleOutput = document.getElementById("subtitle-output");
let isSubmitting = false;

function setStatus(message) {
  if (cameraStatus) {
    cameraStatus.textContent = message;
  }
}

async function sendTextToBackend(text) {
  const response = await fetch("/api/text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  });

  if (!response.ok) {
    throw new Error("Failed to send text to backend");
  }

  return await response.json();
}

async function handleTextSubmit() {
  if (!textInput || isSubmitting) return;

  const text = textInput.value.trim();
  if (!text) return;

  isSubmitting = true;

  try {
    setStatus("Sending text...");
    if (window.pauseAnimationPlayback) {
      window.pauseAnimationPlayback();
    }

    const result = await sendTextToBackend(text);
    console.log("[TEXT_INPUT] backend response:", {
      ok: result.ok,
      animation: result.animation,
      tracedTokenCount: result.traced_tokens ? result.traced_tokens.length : 0,
      missingMotionTokenCount: result.missing_motion_tokens
        ? result.missing_motion_tokens.length
        : 0,
    });

    if (subtitleOutput && result.answer_text) {
      subtitleOutput.textContent = result.answer_text;
    }

    if (result.frames && window.loadAvatarAnimationFromJson) {
      window.loadAvatarAnimationFromJson(
        result.animation || "answer_motion",
        result.frames
      );
    }

    if (result.missing_motion_tokens && result.missing_motion_tokens.length) {
      console.warn("[TEXT_INPUT] missing motion tokens:", result.missing_motion_tokens);
    }

    setStatus("SmolLM answer received");
    textInput.value = "";
  } catch (error) {
    console.error("[TEXT_INPUT] error:", error);
    setStatus("Failed to send text");
  } finally {
    isSubmitting = false;
  }
}

if (sendTextBtn) {
  sendTextBtn.addEventListener("click", handleTextSubmit);
}

if (textInput) {
  textInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;

    event.preventDefault();

    if (!event.repeat) {
      handleTextSubmit();
    }
  });
}
