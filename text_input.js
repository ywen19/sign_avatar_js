const textInput = document.getElementById("text-input");
const sendTextBtn = document.getElementById("send-text-btn");
const statusOutput = document.getElementById("camera-status");
const subtitleOutput = document.getElementById("subtitle-output");
let isSubmitting = false;

function setStatus(message) {
  if (statusOutput) {
    statusOutput.textContent = message;
  }
}

async function sendTextToBackend(text) {
  const response = await fetch("/api/text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    let errorMessage = "Failed to send text to backend";

    try {
      const errorPayload = await response.json();
      if (errorPayload.error) {
        errorMessage = errorPayload.error;
      }
    } catch {
      // Keep the fallback message when the backend response is not JSON.
    }

    throw new Error(errorMessage);
  }

  return response.json();
}

function applyTextResponse(result) {
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
    console.warn(
      "[TEXT_INPUT] missing motion tokens:",
      result.missing_motion_tokens
    );
  }
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
    applyTextResponse(result);

    setStatus("SmolLM answer received");
    textInput.value = "";
  } catch (error) {
    console.error("[TEXT_INPUT] error:", error);
    setStatus("Failed to send text");

    if (window.playAnimationFromStart) {
      window.playAnimationFromStart();
    }
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
