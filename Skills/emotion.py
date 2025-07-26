import requests
import re

def detect_emotions(text):
    """
    Advanced emotion detection using keywords and simple heuristics.
    Returns a list of detected emotions.
    """
    emotion_keywords = {
        "happy": [r"\b(happy|joy|glad|excited|awesome|great|fantastic|yay|delighted|pleased|cheerful)\b"],
        "sad": [r"\b(sad|down|unhappy|depressed|blue|cry|upset|miserable|tearful)\b"],
        "angry": [r"\b(angry|mad|furious|annoyed|irritated|grr|rage|frustrated)\b"],
        "surprised": [r"\b(surprised|shocked|amazed|wow|whoa|astonished|startled)\b"],
        "love": [r"\b(love|like|adore|fond|affection|caring|cherish)\b"],
        "confused": [r"\b(confused|lost|unsure|puzzled|huh|uncertain|perplexed)\b"],
        "stressed": [r"\b(stressed|anxious|overwhelmed|nervous|worried|tense)\b"],
        "bored": [r"\b(bored|uninterested|dull|tedious|monotonous)\b"],
        "motivated": [r"\b(motivated|inspired|driven|determined|ambitious)\b"],
        "grateful": [r"\b(grateful|thankful|appreciative|blessed)\b"],
    }
    detected = []
    for emotion, patterns in emotion_keywords.items():
        for pat in patterns:
            if re.search(pat, text, re.I):
                detected.append(emotion)
                break
    return detected

def advanced_emotion_skill(user_input):
    """
    Detects multiple emotions and responds empathetically, suggesting actions or skills.
    """
    emotions = detect_emotions(user_input)
    if not emotions:
        return "I'm not sure how you're feeling, but I'm here for you! If you want to talk or need a suggestion, just ask."
    responses = []
    for emotion in emotions:
        if emotion == "happy":
            responses.append("I'm glad to hear you're happy! 😊 Keep spreading those good vibes.")
        elif emotion == "sad":
            responses.append("I'm sorry you're feeling sad. If you want to talk or need cheering up, I can tell you a joke or suggest a relaxing activity.")
        elif emotion == "angry":
            responses.append("It sounds like you're angry. Taking a few deep breaths or a short walk might help. Want to vent or hear something calming?")
        elif emotion == "surprised":
            responses.append("Whoa, something surprised you! Want to share more?")
        elif emotion == "love":
            responses.append("Love is wonderful! 💖 Who or what has your affection today?")
        elif emotion == "confused":
            responses.append("Feeling confused is normal. If you have questions, I'm here to help clarify!")
        elif emotion == "stressed":
            responses.append("Stress can be tough. Would you like a quick breathing exercise or a motivational quote?")
        elif emotion == "bored":
            responses.append("Boredom strikes! Want a fun fact, a joke, or a suggestion for something new?")
        elif emotion == "motivated":
            responses.append("Awesome! Stay motivated and keep pushing forward. Let me know if you need resources or inspiration.")
        elif emotion == "grateful":
            responses.append("Gratitude is powerful. It's great to appreciate the good things in life!")
    # Suggest relevant skills
    if "sad" in emotions or "stressed" in emotions or "bored" in emotions:
        responses.append("Try 'jolly' for a mood boost, or ask for a joke, a quote, or a relaxation tip!")
    return " ".join(responses)

def register(jarvis):
    jarvis.register_skill('emotion', advanced_emotion_skill)
