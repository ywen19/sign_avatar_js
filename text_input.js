/**
 * Handles text submitted from the frontend to the backend text API.
 *
 * This script coordinates textarea and button events, reports request status,
 * forwards successful motion data to avatar.js, and restores animation after
 * failures. It uses asynchronous requests, not worker threads; isSubmitting
 * prevents overlapping requests from repeated user events.
 */

// Cache the page elements used for input, commands, and output updates.
const textInput = document.getElementById("text-input");
const sendTextBtn = document.getElementById("send-text-btn");
const statusOutput = document.getElementById("camera-status");
const subtitleOutput = document.getElementById("subtitle-output");

// Allow only one asynchronous backend request to be active at a time.
let isSubmitting = false;

/**
 * Publish a processing-status message in the shared frontend status area.
 *
 * @param {string} message - Status text to display.
 */
function setStatus(message) {
  if (statusOutput) {
    statusOutput.textContent = message;
  }
}

/**
 * Send user text to the backend and return its parsed response payload.
 *
 * @param {string} text - Normalized, non-empty text from the textarea.
 * @returns {Promise<object>} Parsed backend response data.
 * @throws {Error} Backend detail when available, otherwise a fallback error.
 */
async function sendTextToBackend(text) {
  // Await the network signal so the caller can coordinate request state.
  const response = await fetch("/api/text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    let errorMessage = "Failed to send text to backend";

    // Prefer the backend's diagnostic detail for failed JSON responses.
    try {
      const errorPayload = await response.json();
      if (errorPayload.error) {
        errorMessage = errorPayload.error;
      }
    } catch {
      // Keep the fallback message when the backend response is not JSON.
    }

    // Reject the request so the submission handler enters its recovery path.
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Apply a successful backend response to subtitles and avatar playback.
 *
 * @param {object} result - Parsed text, diagnostics, and animation payload.
 */
function applyTextResponse(result) {
  // Log compact request diagnostics without printing the complete frame data.
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

  // Signal avatar.js through its public window API when motion data exists.
  if (result.frames && window.loadAvatarAnimationFromJson) {
    window.loadAvatarAnimationFromJson(
      result.animation || "answer_motion",
      result.frames
    );
  }

  // Missing tokens are non-fatal, but remain visible for motion diagnostics.
  if (result.missing_motion_tokens && result.missing_motion_tokens.length) {
    console.warn(
      "[TEXT_INPUT] missing motion tokens:",
      result.missing_motion_tokens
    );
  }
}

/**
 * Validate and submit the current textarea value as one coordinated request.
 *
 * Duplicate UI signals are ignored until the active request settles.
 *
 * @returns {Promise<void>}
 */
async function handleTextSubmit() {
  // Stop when input is unavailable or another request is already in flight.
  if (!textInput || isSubmitting) return;

  const text = textInput.value.trim();
  if (!text) return;

  // Acquire the submission guard before the first asynchronous operation.
  isSubmitting = true;

  try {
    setStatus("Sending text...");

    // Signal avatar.js to hold the current pose while processing the request.
    if (window.pauseAnimationPlayback) {
      window.pauseAnimationPlayback();
    }

    const result = await sendTextToBackend(text);
    applyTextResponse(result);

    // Publish success only after the response has been applied to the UI.
    setStatus("SmolLM answer received");
    textInput.value = "";
  } catch (error) {
    console.error("[TEXT_INPUT] error:", error);
    setStatus("Failed to send text");

    // Restore the previous animation after a failed backend request.
    if (window.playAnimationFromStart) {
      window.playAnimationFromStart();
    }
  } finally {
    // Release the guard for success, failure, and unexpected exceptions.
    isSubmitting = false;
  }
}

// Treat a send-button click as a submission signal when the button exists.
if (sendTextBtn) {
  sendTextBtn.addEventListener("click", handleTextSubmit);
}

// Keep Shift+Enter for newlines and use a single Enter press to submit.
if (textInput) {
  textInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;

    // Stop Enter from inserting a newline when it represents submission.
    event.preventDefault();

    // Ignore auto-repeat events produced when the Enter key is held down.
    if (!event.repeat) {
      handleTextSubmit();
    }
  });
}
