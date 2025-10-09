const socket = io();

const DOM = {
  messageInput: null,
  sendButton: null,
  conversationHistory: null,
  statusIndicator: null,
  clearButton: null
};

function initializeDOM() {
  DOM.messageInput = document.getElementById('messageInput');
  DOM.sendButton = document.getElementById('sendButton');
  DOM.conversationHistory = document.getElementById('conversationHistory');
  DOM.statusIndicator = document.getElementById('statusIndicator');
  DOM.clearButton = document.getElementById('clearButton');
}

function initializeEventListeners() {
  // Button clicks
  DOM.sendButton.addEventListener('click', sendMessage);
  DOM.clearButton.addEventListener('click', clearHistory);

  // Enter key to send
  DOM.messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Socket event listeners
  socket.on('connect', handleConnect);
  socket.on('disconnect', handleDisconnect);
  socket.on('message_response', handleMessageResponse);
  socket.on('intent_recognition', handleIntentRecognition);
}

// SOCKET EVENT HANDLERS
function handleConnect() {
  console.log('Connected to server');
  updateConnectionStatus(true);
}

function handleDisconnect() {
  console.log('Disconnected from server');
  updateConnectionStatus(false);
}

function handleMessageResponse(data) {
  console.log('Received message response:', data);

  try {
    removeTypingIndicator();

    if (data && data.conversation_history) {
      displayConversation(data.conversation_history);
    } else {
      console.error('Invalid message response data:', data);
      showError('Failed to load conversation history');
    }

    if (data && data.intent_info) {
      console.log('Intent recognized:', data.intent_info);
    }

    DOM.messageInput.focus();
  } catch (error) {
    console.error('Error handling message response:', error);
    showError('Failed to display message');
  }
}

function handleIntentRecognition(data) {
  console.log('Intent recognition result:', data);

  if (data && data.intent_info) {
    showIntentRecognition(data.intent_info);
  }
}

// UI UPDATE FUNCTIONS
function updateConnectionStatus(connected) {
  if (!DOM.statusIndicator) return;

  DOM.statusIndicator.className = connected
    ? 'status-indicator connected'
    : 'status-indicator disconnected';
}

function displayConversation(history) {
  if (!DOM.conversationHistory) return;

  DOM.conversationHistory.innerHTML = '';

  if (!history || history.length === 0) {
    showNoMessages();
    return;
  }

  const fragment = document.createDocumentFragment();

  history.forEach((entry) => {
    const messageElement = createMessageElement(entry);
    fragment.appendChild(messageElement);
  });

  DOM.conversationHistory.appendChild(fragment);
  scrollToBottom();
}

function createMessageElement(entry) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${entry.type}`;

  const messageContent = createMessageContent(entry);
  const messageTime = createMessageTime(entry.timestamp);

  messageDiv.appendChild(messageContent);
  messageDiv.appendChild(messageTime);

  return messageDiv;
}

function createMessageContent(entry) {
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = entry.message;

  if (entry.type === 'assistant' && entry.intent) {
    const intentInfo = createIntentInfo(entry);
    contentDiv.appendChild(intentInfo);
  }

  return contentDiv;
}

function createIntentInfo(entry) {
  const intentDiv = document.createElement('div');
  intentDiv.className = 'intent-info';

  const layerBadge = entry.layer_used
    ? createLayerBadge(entry.layer_used)
    : '';

  intentDiv.innerHTML = `
    <small>
      Intent: <strong>${sanitizeHTML(entry.intent)}</strong> 
      (${sanitizeHTML(entry.confidence)}, ${(entry.similarity * 100).toFixed(1)}%)
      ${layerBadge}
    </small>
  `;

  return intentDiv;
}

function createLayerBadge(layerUsed) {
  const layerName = capitalizeFirst(layerUsed);
  return `<span class="layer-badge">${sanitizeHTML(layerName)}</span>`;
}

function createMessageTime(timestamp) {
  const timeDiv = document.createElement('div');
  timeDiv.className = 'message-time';
  timeDiv.textContent = formatTimestamp(timestamp);
  return timeDiv;
}

function showTypingIndicator() {
  removeTypingIndicator();

  const typingDiv = document.createElement('div');
  typingDiv.className = 'message assistant typing';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('span');
    dot.className = 'dot';
    contentDiv.appendChild(dot);
  }

  typingDiv.appendChild(contentDiv);
  DOM.conversationHistory.appendChild(typingDiv);
  scrollToBottom();
}

function removeTypingIndicator() {
  const typingIndicator = DOM.conversationHistory.querySelector('.message.assistant.typing');

  if (typingIndicator) {
    typingIndicator.remove();
  }
}

function showNoMessages() {
  DOM.conversationHistory.innerHTML =
    '<div class="no-messages">No messages yet. Start a conversation!</div>';
}

function showError(message) {
  console.error(message);
}

function sendMessage() {
  const message = DOM.messageInput.value.trim();

  if (!message) {
    return;
  }

  if (!socket.connected) {
    alert('Not connected to server. Please wait for connection.');
    return;
  }

  removeNoMessagesPlaceholder();
  addUserMessageToUI(message);
  showTypingIndicator();
  sendMessageToServer(message);
  clearInput();
}

function addUserMessageToUI(message) {
  const userEntry = {
    type: 'user',
    message: message,
    timestamp: Date.now()
  };

  const messageElement = createMessageElement(userEntry);
  DOM.conversationHistory.appendChild(messageElement);
  scrollToBottom();
}

function sendMessageToServer(message) {
  socket.emit('send_message', { message });

  DOM.sendButton.disabled = true;
  setTimeout(() => {
    DOM.sendButton.disabled = false;
  }, 200);
}

function clearInput() {
  DOM.messageInput.value = '';
  DOM.messageInput.focus();
}

function clearHistory() {
  if (!confirm('Are you sure you want to clear the conversation history?')) {
    return;
  }

  showNoMessages();
  socket.emit('clear_history');
}

function removeNoMessagesPlaceholder() {
  const noMsgDiv = DOM.conversationHistory.querySelector('.no-messages');
  if (noMsgDiv) {
    noMsgDiv.remove();
  }
}

// INTENT RECOGNITION DISPLAY
function showIntentRecognition(intentInfo) {
  if (!intentInfo) return;

  const details = [
    `Intent: ${intentInfo.intent}`,
    `Confidence: ${intentInfo.confidence}`,
    `Similarity: ${(intentInfo.similarity * 100).toFixed(1)}%`,
    `Layer: ${intentInfo.layer_used}`
  ];

  if (intentInfo.llm_explanation) {
    details.push(`Explanation: ${intentInfo.llm_explanation}`);
  }

  const result = details.join('\n');
  alert(`Intent Recognition Result:\n\n${result}`);
}

// UTILITY FUNCTIONS
function formatTimestamp(timestamp) {
  try {
    const date = timestamp ? new Date(timestamp) : new Date();
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (error) {
    console.error('Error formatting timestamp:', error);
    return new Date().toLocaleTimeString();
  }
}

function capitalizeFirst(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function sanitizeHTML(str) {
  if (typeof str !== 'string') return str;

  const temp = document.createElement('div');
  temp.textContent = str;
  return temp.innerHTML;
}

function scrollToBottom() {
  if (DOM.conversationHistory) {
    DOM.conversationHistory.scrollTop = DOM.conversationHistory.scrollHeight;
  }
}

// APPLICATION ENTRY POINT
document.addEventListener('DOMContentLoaded', function() {
  console.log('Voice Assistant client initializing...');

  try {
    initializeDOM();
    initializeEventListeners();
    showNoMessages();
    DOM.messageInput.focus();

    console.log('Voice Assistant client initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Voice Assistant client:', error);
  }
});