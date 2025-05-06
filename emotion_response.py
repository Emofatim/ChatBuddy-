def get_bot_response(emotion):
    RESPONSES = {
        "happy": "Yay! You sound cheerful! Let's play a game.",
        "sad": "I'm here for you. Want to talk about it?",
        "angry": "Take a deep breath. Let's calm down together.",
        "neutral": "Let's chat! What would you like to do?",
        "fearful": "Don't worry, you're safe here.",
        "disgust": "Yikes! Let's find something fun to do.",
        "surprised": "Wow! That was unexpected!"
    }
    return RESPONSES.get(emotion, "Let's keep talking!")
