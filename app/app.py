from flask import Flask, request, jsonify, render_template
import sys
import os

# Add the 'scripts' folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from predict import predict_emotion
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from emotion_response import get_bot_response

app = Flask(__name__)

RESPONSES = {
    "happy": "Yay! You sound cheerful! Let's play a game.",
    "sad": "I'm here for you. Want to talk about it?",
    "angry": "Take a deep breath. Let's calm down together.",
    "neutral": "Let's chat! What would you like to do?",
    "fearful": "Don't worry, you're safe here.",
    "disgust": "Yikes! Let's find something fun to do.",
    "surprised": "Wow! That was unexpected!"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files["audio"]
    emotion = predict_emotion(audio_file)
    response = RESPONSES.get(emotion, "Let's keep talking!")
    return jsonify({"emotion": emotion, "response": response})

if __name__ == "__main__":
    app.run(debug=True)