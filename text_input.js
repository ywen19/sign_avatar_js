const textInput = document.getElementById("text-input");
const sendTextBtn = document.getElementById("send-text-btn");
const cameraStatus = document.getElementById("camera-status");

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
  if (!textInput) return;

  const text = textInput.value.trim();
  if (!text) return;

  try {
    setStatus("Sending text...");
    const result = await sendTextToBackend(text);
    console.log("[TEXT_INPUT] backend response:", result);
    setStatus("Text sent: " + text);
    textInput.value = "";
  } catch (error) {
    console.error("[TEXT_INPUT] error:", error);
    setStatus("Failed to send text");
  }
}

if (sendTextBtn) {
  sendTextBtn.addEventListener("click", handleTextSubmit);
}

if (textInput) {
  textInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleTextSubmit();
    }
  });
}