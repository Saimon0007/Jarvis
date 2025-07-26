import re
from difflib import get_close_matches

def register(jarvis):
    # Simple emotion detection keywords
    emotions = {
        'happy': [r'\b(happy|joy|glad|excited|awesome|great|fantastic|yay)\b'],
        'sad': [r'\b(sad|down|unhappy|depressed|blue|cry|upset)\b'],
        'angry': [r'\b(angry|mad|furious|annoyed|irritated|grr)\b'],
        'surprised': [r'\b(surprised|shocked|amazed|wow|whoa)\b'],
        'love': [r'\b(love|like|adore|fond)\b'],
        'confused': [r'\b(confused|lost|unsure|puzzled|huh)\b'],
    }
    emotion_responses = {
        'happy': "Yay! Your happiness is contagious! 😊 How can I make your day even better?",
        'sad': "Oh no, I'm here for you! If you want to talk or need a joke, just ask! 💙",
        'angry': "Uh-oh, sounds like something's ruffled your feathers. Want to vent or need a distraction? 😅",
        'surprised': "Whoa! That sounds exciting! Tell me more! 😲",
        'love': "Aww, spreading love is the best! 💖 Who or what do you love today?",
        'confused': "No worries, we all get confused! Ask me anything, I'll do my best to help! 🤔",
    }
    patterns = [
        (re.compile(r"\b(hi|hello|hey|greetings)\b", re.I), "Hey there! I'm Jarvis, your jolly AI buddy! How are you feeling today?"),
        (re.compile(r"how are you", re.I), "I'm always in a good mood, ready to help and spread some cheer!"),
        (re.compile(r"what.*can.*you.*do|help|abilities|skills", re.I), "I can chat, fetch info, translate, and even try to cheer you up! Try: hello, fetch <url>, gtranslate <lang_code> <text>.")
    ]
    known_skills = list(jarvis.skills.keys())

    def jolly_nlp(user_input):
        # Emotion detection
        for emotion, regexes in emotions.items():
            for regex in regexes:
                if re.search(regex, user_input, re.I):
                    return emotion_responses[emotion]
        # Pattern matching for common intents
        for pattern, response in patterns:
            if pattern.search(user_input):
                return response
        # Suggest a skill if user input is close to a skill name
        words = user_input.lower().split()
        matches = get_close_matches(words[0], known_skills, n=1, cutoff=0.7)
        if matches:
            return f"Did you mean: {matches[0]}? Try '{matches[0]}' as a command."
        # Default jolly fallback
        return "I'm here to brighten your day! Try a command or just tell me how you're feeling! 😄"

    jarvis.register_skill('jolly', jolly_nlp)
