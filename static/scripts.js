const socket = io()

const DOM = {
  orb: null,
  orbContainer: null,
  voiceStatus: null,
  userTranscript: null,
  assistantTranscript: null,
  responseAudio: null,
  modeToggle: null,
  voiceContainer: null,
  chatContainer: null,
  chatMessages: null,
  chatInput: null,
  sendBtn: null,
  chatResponseAudio: null,
}

let mediaRecorder = null
let audioChunks = []
let isRecording = false
let isProcessing = false
let isConnected = false
let currentMode = "voice"
let conversationHistory = []

function initializeDOM() {
  DOM.orb = document.getElementById("orb")
  DOM.orbContainer = document.getElementById("orbContainer")
  DOM.voiceStatus = document.querySelector(".status-message")
  DOM.userTranscript = document.getElementById("userTranscript")
  DOM.assistantTranscript = document.getElementById("assistantTranscript")
  DOM.responseAudio = document.getElementById("responseAudio")
  DOM.modeToggle = document.getElementById("modeToggle")
  DOM.voiceContainer = document.getElementById("voiceContainer")
  DOM.chatContainer = document.getElementById("chatContainer")
  DOM.chatMessages = document.getElementById("chatMessages")
  DOM.chatInput = document.getElementById("chatInput")
  DOM.sendBtn = document.getElementById("sendBtn")
  DOM.chatResponseAudio = document.getElementById("chatResponseAudio")
}

function initializeEventListeners() {
  DOM.orbContainer.addEventListener("click", handleOrbClick)
  DOM.responseAudio.addEventListener("ended", handleAudioEnded)
  DOM.chatResponseAudio.addEventListener("ended", () => console.log("Chat audio playback ended"))
  DOM.modeToggle.addEventListener("click", toggleMode)
  DOM.sendBtn.addEventListener("click", sendTextMessage)
  DOM.chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendTextMessage()
    }
  })

  socket.on("connect", handleConnect)
  socket.on("disconnect", handleDisconnect)
  socket.on("transcription_status", handleTranscriptionStatus)
  socket.on("transcription_result", handleTranscriptionResult)
  socket.on("voice_response", handleVoiceResponse)
  socket.on("text_response", handleTextResponse)
  socket.on("error", handleError)
  socket.on("history_cleared", () => {
      DOM.chatMessages.innerHTML = ""
  })
}

function handleConnect() {
  console.log("Connected to server")
  isConnected = true
  updateConnectionStatus(true)
  updateStatusMessage("Tap to speak")
  socket.emit("clear_history")
  conversationHistory = []
}

function handleDisconnect() {
  console.log("Disconnected from server")
  isConnected = false
  updateConnectionStatus(false)
  updateStatusMessage("Disconnected")
}

function updateConnectionStatus(connected) {
  DOM.orb.classList.toggle("connected", connected)
}

function updateStatusMessage(message) {
  DOM.voiceStatus.textContent = message
}

function setOrbState(state) {
  DOM.orb.classList.remove("listening", "processing", "speaking")
  if (state) DOM.orb.classList.add(state)
}

function handleOrbClick() {
  if (!isConnected) {
    updateStatusMessage("Not connected")
    return
  }
  isRecording ? stopRecording() : startRecording()
}

async function startRecording() {
  if (isProcessing) {
    console.log("Still processing previous request")
    return
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      },
    })

    audioChunks = []
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" })

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data)
    }

    mediaRecorder.onstop = handleRecordingStop

    mediaRecorder.start()
    isRecording = true
    setOrbState("listening")
    updateStatusMessage("Listening")
    console.log("Recording started")
  } catch (error) {
    console.error("Error accessing microphone:", error)
    updateStatusMessage("Microphone access denied")
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop()
    isRecording = false
    mediaRecorder.stream.getTracks().forEach((track) => track.stop())
    setOrbState("processing")
    updateStatusMessage("Processing")
    console.log("Recording stopped")
  }
}

async function handleRecordingStop() {
  isProcessing = true
  const audioBlob = new Blob(audioChunks, { type: "audio/webm" })
  const reader = new FileReader()
  reader.onloadend = () => socket.emit("voice_input", { audio: reader.result })
  reader.readAsDataURL(audioBlob)
}

function handleTranscriptionStatus(data) {
  console.log("Transcription status:", data.status)
  if (data.status === "processing") {
    setOrbState("processing")
    updateStatusMessage("Transcribing")
  }
}

function handleTranscriptionResult(data) {
  console.log("Transcription result:", data)

  if (data.error) {
    updateStatusMessage(data.error)
    setOrbState("")
    isProcessing = false
    return
  }

  if (data.text) {
    showTranscript(data.text, "user")
    updateStatusMessage("Thinking")
  }
}

function handleVoiceResponse(data) {
  console.log("Voice response:", data)

  if (data.conversation_history) conversationHistory = data.conversation_history
  if (data.assistant_response) showTranscript(data.assistant_response, "assistant")

  if (data.audio_url) {
    playAudio(data.audio_url, DOM.responseAudio, "speaking", "Speaking")
  } else {
    setOrbState("")
    updateStatusMessage("Tap to speak")
    isProcessing = false
  }
}

function handleTextResponse(data) {
  console.log("Text response:", data)

  if (data.conversation_history) conversationHistory = data.conversation_history

  removeLoadingIndicator()

  if (data.assistant_response) {
    const metadata = {
      intent: data.intent_info.intent,
      layer_used: data.intent_info.layer_used,
      similarity: data.intent_info.similarity
    }
    addChatMessage(data.assistant_response, "assistant", metadata)
  }

  if (data.audio_url) playAudio(data.audio_url, DOM.chatResponseAudio)

  DOM.sendBtn.disabled = false
}

function handleError(data) {
  console.error("Server error:", data)

  if (currentMode === "voice") {
    updateStatusMessage("Error: " + data.message)
    setOrbState("")
    isProcessing = false
  } else {
    removeLoadingIndicator()
    addChatMessage("Error: " + data.message, "system", null)
    DOM.sendBtn.disabled = false
  }
}

function showTranscript(text, type) {
  const transcriptElement = type === "user" ? DOM.userTranscript : DOM.assistantTranscript
  transcriptElement.textContent = text
  transcriptElement.classList.add("show")

  clearTimeout(transcriptElement.hideTimeout)

  if (type === "user") {
    transcriptElement.hideTimeout = setTimeout(() => {
      if (!isProcessing && (!DOM.responseAudio.duration || DOM.responseAudio.ended)) {
        transcriptElement.classList.remove("show")
      }
    }, 8000)
  }
}

function playAudio(audioUrl, audioElement, orbState = null, statusMessage = null) {
  try {
    if (orbState) setOrbState(orbState)
    if (statusMessage) updateStatusMessage(statusMessage)

    audioElement.src = audioUrl
    audioElement.load()
    audioElement.play().catch((error) => {
      console.error("Error playing audio:", error)
      if (audioElement === DOM.responseAudio) handleAudioEnded()
    })
  } catch (error) {
    console.error("Error setting up audio:", error)
    if (audioElement === DOM.responseAudio) handleAudioEnded()
  }
}

function handleAudioEnded() {
  setOrbState("")
  updateStatusMessage("Tap to speak")
  isProcessing = false

  setTimeout(() => {
    DOM.userTranscript.classList.remove("show")
    DOM.assistantTranscript.classList.remove("show")
  }, 1000)
}

function toggleMode() {
  if (currentMode === "voice") {
    currentMode = "chat"
    DOM.voiceContainer.classList.add("hidden")
    DOM.chatContainer.classList.add("active")
    DOM.modeToggle.classList.add("chat-mode")
    DOM.modeToggle.title = "Switch to voice mode"
    renderChatHistory()
    DOM.chatInput.focus()
  } else {
    currentMode = "voice"
    DOM.chatContainer.classList.remove("active")
    DOM.voiceContainer.classList.remove("hidden")
    DOM.modeToggle.classList.remove("chat-mode")
    DOM.modeToggle.title = "Switch to text mode"
  }
}

function renderChatHistory() {
  DOM.chatMessages.innerHTML = ""

  conversationHistory.forEach((entry) => {
    if (entry.type === "user") {
      addChatMessage(entry.message, "user", null, false)
    } else if (entry.type === "assistant") {
      const metadata = {
        intent: entry.intent,
        layer_used: entry.layer_used,
        similarity: entry.similarity
      }
      addChatMessage(entry.message, "assistant", metadata, false)
    }
  })
}

function addChatMessage(text, type, metadata = null, shouldScroll = true) {
  const messageDiv = document.createElement("div")
  messageDiv.className = `chat-message ${type}`

  const textDiv = document.createElement("div")
  textDiv.className = "message-text"
  textDiv.textContent = text
  messageDiv.appendChild(textDiv)

  if (type === "assistant" && metadata) {
    const metaDiv = document.createElement("div")
    metaDiv.className = "message-metadata"
    metaDiv.textContent = `${metadata.intent} · ${metadata.layer_used} · ${metadata.similarity.toFixed(2)}`
    messageDiv.appendChild(metaDiv)
  }

  DOM.chatMessages.appendChild(messageDiv)
  if (shouldScroll) DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight
}

function addLoadingIndicator() {
  const loadingDiv = document.createElement("div")
  loadingDiv.className = "chat-message assistant loading-indicator"
  loadingDiv.id = "loadingIndicator"

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("span")
    dot.className = "loading-dot"
    loadingDiv.appendChild(dot)
  }

  DOM.chatMessages.appendChild(loadingDiv)
  DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight
}

function removeLoadingIndicator() {
  const loadingIndicator = document.getElementById("loadingIndicator")
  if (loadingIndicator) loadingIndicator.remove()
}

function sendTextMessage() {
  const message = DOM.chatInput.value.trim()

  if (!message) return

  if (!isConnected) {
    addChatMessage("Not connected to server", "system", null)
    return
  }

  addChatMessage(message, "user", null)
  DOM.chatInput.value = ""
  DOM.sendBtn.disabled = true

  addLoadingIndicator()

  socket.emit("text_input", { text: message })
  console.log("Sent text message:", message)
}

document.addEventListener("DOMContentLoaded", () => {
  console.log("Voice Assistant initializing...")

  try {
    initializeDOM()
    initializeEventListeners()
    console.log("Voice Assistant initialized successfully")
  } catch (error) {
    console.error("Failed to initialize Voice Assistant:", error)
  }
})