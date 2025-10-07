// Socket.IO client setup
const socket = io();

// DOM elements
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const conversationHistory = document.getElementById("conversationHistory");
const statusIndicator = document.getElementById("statusIndicator");

// Connection status management
function updateConnectionStatus(connected) {
  if (connected) {
    statusIndicator.className = "status-indicator connected";
  } else {
    statusIndicator.className = "status-indicator disconnected";
  }
}

// Socket event handlers
socket.on("connect", function () {
  console.log("Connected to server");
  updateConnectionStatus(true);
});

socket.on("disconnect", function () {
  console.log("Disconnected from server");
  updateConnectionStatus(false);
});

socket.on("message_response", function (data) {
  console.log("Received message response:", data);

  displayConversation(data.conversation_history);

  if (data.intent_info) {
    console.log("Intent recognized:", data.intent_info);
  }

  messageInput.value = "";
  messageInput.focus();
});

socket.on("intent_recognition", function (data) {
  console.log("Intent recognition result:", data);
  showIntentRecognition(data.intent_info);
});

function displayConversation(history) {
  conversationHistory.innerHTML = "";

  if (!history || history.length === 0) {
    conversationHistory.innerHTML =
      '<div class="no-messages">No messages yet. Start a conversation!</div>';
    return;
  }

  history.forEach((entry) => {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${entry.type}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = entry.message;

    // Intent information for assistant messages
    if (entry.type === "assistant" && entry.intent) {
      const intentInfo = document.createElement("div");
      intentInfo.className = "intent-info";

      let layerBadge = '';
      if (entry.layer_used) {
        let layerName = entry.layer_used.charAt(0).toUpperCase() + entry.layer_used.slice(1);
        const layerClass = `layer-badge`;
        layerBadge = ` <span class="${layerClass}">${layerName}</span>`;
      }

      intentInfo.innerHTML = `
        <small>
          Intent: <strong>${entry.intent}</strong> 
          (${entry.confidence}, ${(entry.similarity * 100).toFixed(1)}%)
          ${layerBadge}
        </small>
      `;
      messageContent.appendChild(intentInfo);
    }

    const messageTime = document.createElement("div");
    messageTime.className = "message-time";
    messageTime.textContent = formatTimestamp(entry.timestamp);

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);
    conversationHistory.appendChild(messageDiv);
  });

  conversationHistory.scrollTop = conversationHistory.scrollHeight;
}

function formatTimestamp(timestamp) {
  const now = new Date();
  return now.toLocaleTimeString();
}

function sendMessage() {
  const message = messageInput.value.trim();

  if (!message) {
    alert("Please enter a message");
    return;
  }

  if (!socket.connected) {
    alert("Not connected to server. Please wait for connection.");
    return;
  }

  sendButton.disabled = true;

  // Send message to server
  socket.emit("send_message", { message: message });

  setTimeout(() => {
    sendButton.disabled = false;
  }, 200);
}

function clearHistory() {
  if (confirm("Are you sure you want to clear the conversation history?")) {
    conversationHistory.innerHTML =
      '<div class="no-messages">No messages yet. Start a conversation!</div>';
    socket.emit("clear_history");
  }
}

// Show intent recognition result
function showIntentRecognition(intentInfo) {
  const result = `Intent: ${intentInfo.intent}\nConfidence: ${
    intentInfo.confidence
  }\nSimilarity: ${(intentInfo.similarity * 100).toFixed(1)}%\nLayer: ${intentInfo.layer_used}`;
  alert(`Intent Recognition Result:\n\n${result}`);
}

// Event listeners
messageInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});

// Initialize
document.addEventListener("DOMContentLoaded", function () {
  messageInput.focus();
  displayConversation([]);
  console.log("Voice Assistant client initialized");
});