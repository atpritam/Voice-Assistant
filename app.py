from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os

# Utils directory path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.intent_recognizer import IntentRecognizer

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'hjbasfbue76t34g76wgv3bywyu47'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Conversation history
conversation_history = []

# Initialize intent recognizer
intent_recognizer = IntentRecognizer()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('send_message')
def handle_message(data):
    """Handle incoming text messages with intent recognition"""
    message = data.get('message', '')
    if message:
        # Add user message to conversation history
        conversation_history.append({
            'type': 'user',
            'message': message,
            'timestamp': len(conversation_history)
        })

        # Recognize intent and generate appropriate response
        intent_info = intent_recognizer.recognize_intent(message)
        response = intent_recognizer.generate_response(intent_info, message)

        # Add assistant response to conversation history
        conversation_history.append({
            'type': 'assistant',
            'message': response,
            'timestamp': len(conversation_history),
            'intent': intent_info.intent,
            'confidence': intent_info.confidence_level,
            'similarity': intent_info.confidence
        })

        # Send response back to client
        emit('message_response', {
            'user_message': message,
            'assistant_response': response,
            'conversation_history': conversation_history,
            'intent_info': {
                'intent': intent_info.intent,
                'confidence': intent_info.confidence_level,
                'similarity': intent_info.confidence
            }
        })

@socketio.on('recognize_intent')
def handle_recognize_intent(data):
    """Recognize intent for a given message without generating response"""
    message = data.get('message', '')
    if message:
        intent_info = intent_recognizer.recognize_intent(message)
        emit('intent_recognition', {
            'message': message,
            'intent_info': intent_info
        })

@socketio.on('clear_history')
def handle_clear_history():
    conversation_history.clear()

@app.route('/')
def index():
    """Static HTML page"""
    return app.send_static_file('index.html')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
