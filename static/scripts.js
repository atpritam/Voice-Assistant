const socket = io();

const DOM = {
  statusIndicator: null,
  statusText: null,
  orb: null,
  voiceStatus: null,
  userTranscript: null,
  assistantTranscript: null,
  micButton: null,
  closeButton: null,
  responseAudio: null
};

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isProcessing = false;

function initializeDOM() {
  DOM.statusIndicator = document.getElementById('statusIndicator');
  DOM.statusText = document.querySelector('.status-text');
  DOM.orb = document.getElementById('orb');
  DOM.voiceStatus = document.querySelector('.status-message');
  DOM.userTranscript = document.getElementById('userTranscript');
  DOM.assistantTranscript = document.getElementById('assistantTranscript');
  DOM.micButton = document.getElementById('micButton');
  DOM.closeButton = document.getElementById('closeButton');
  DOM.responseAudio = document.getElementById('responseAudio');
}

function initializeEventListeners() {
  DOM.micButton.addEventListener('click', toggleRecording);
  DOM.closeButton.addEventListener('click', clearConversation);

  socket.on('connect', handleConnect);
  socket.on('disconnect', handleDisconnect);
  socket.on('transcription_status', handleTranscriptionStatus);
  socket.on('transcription_result', handleTranscriptionResult);
  socket.on('voice_response', handleVoiceResponse);
  socket.on('error', handleError);

  DOM.responseAudio.addEventListener('ended', handleAudioEnded);
}

function handleConnect() {
  console.log('Connected to server');
  updateConnectionStatus(true);
  updateStatusMessage('Tap to speak');
}

function handleDisconnect() {
  console.log('Disconnected from server');
  updateConnectionStatus(false);
  updateStatusMessage('Disconnected');
}

function updateConnectionStatus(connected) {
  if (connected) {
    DOM.statusIndicator.classList.add('connected');
    DOM.statusText.textContent = 'Connected';
  } else {
    DOM.statusIndicator.classList.remove('connected');
    DOM.statusText.textContent = 'Disconnected';
  }
}

function updateStatusMessage(message) {
  DOM.voiceStatus.textContent = message;
}

function setOrbState(state) {
  DOM.orb.className = `orb ${state}`;
}

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (isProcessing) {
    console.log('Still processing previous request');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioChunks = [];

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = handleRecordingStop;

    mediaRecorder.start();
    isRecording = true;

    DOM.micButton.classList.add('recording');
    setOrbState('listening');
    updateStatusMessage('Listening...');

    console.log('Recording started');

  } catch (error) {
    console.error('Error accessing microphone:', error);
    updateStatusMessage('Microphone access denied');
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;

    mediaRecorder.stream.getTracks().forEach(track => track.stop());

    DOM.micButton.classList.remove('recording');
    setOrbState('processing');
    updateStatusMessage('Processing...');

    console.log('Recording stopped');
  }
}

async function handleRecordingStop() {
  isProcessing = true;

  const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

  const reader = new FileReader();
  reader.onloadend = () => {
    const base64Audio = reader.result;
    socket.emit('voice_input', { audio: base64Audio });
  };
  reader.readAsDataURL(audioBlob);
}

function handleTranscriptionStatus(data) {
  console.log('Transcription status:', data.status);
  if (data.status === 'processing') {
    setOrbState('processing');
    updateStatusMessage('Transcribing...');
  }
}

function handleTranscriptionResult(data) {
  console.log('Transcription result:', data);

  if (data.error) {
    updateStatusMessage(data.error);
    setOrbState('');
    isProcessing = false;
    return;
  }

  if (data.text) {
    showTranscript(data.text, 'user');
    updateStatusMessage('Thinking...');
  }
}

function handleVoiceResponse(data) {
  console.log('Voice response:', data);

  if (data.assistant_response) {
    showTranscript(data.assistant_response, 'assistant');
  }

  if (data.audio_url) {
    playResponseAudio(data.audio_url);
  } else {
    setOrbState('');
    updateStatusMessage('Tap to speak');
    isProcessing = false;
  }
}

function handleError(data) {
  console.error('Server error:', data);
  updateStatusMessage('Error: ' + data.message);
  setOrbState('');
  isProcessing = false;
}

function showTranscript(text, type) {
  const transcriptElement = type === 'user' ? DOM.userTranscript : DOM.assistantTranscript;

  transcriptElement.textContent = text;
  transcriptElement.classList.add('show');

  setTimeout(() => {
    transcriptElement.classList.remove('show');
  }, 8000);
}

function playResponseAudio(audioUrl) {
  try {
    setOrbState('speaking');
    updateStatusMessage('Speaking...');

    DOM.responseAudio.src = audioUrl;
    DOM.responseAudio.load();

    DOM.responseAudio.play().catch(error => {
      console.error('Error playing audio:', error);
      handleAudioEnded();
    });

  } catch (error) {
    console.error('Error setting up audio:', error);
    handleAudioEnded();
  }
}

function handleAudioEnded() {
  setOrbState('');
  updateStatusMessage('Tap to speak');
  isProcessing = false;
}

function clearConversation() {
  if (!confirm('Clear conversation?')) {
    return;
  }

  DOM.userTranscript.textContent = '';
  DOM.assistantTranscript.textContent = '';
  DOM.userTranscript.classList.remove('show');
  DOM.assistantTranscript.classList.remove('show');

  socket.emit('clear_history');
  updateStatusMessage('Conversation cleared');

  setTimeout(() => {
    updateStatusMessage('Tap to speak');
  }, 2000);
}

document.addEventListener('DOMContentLoaded', function() {
  console.log('Voice Assistant initializing...');

  try {
    initializeDOM();
    initializeEventListeners();
    console.log('Voice Assistant initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Voice Assistant:', error);
  }
});