"""
Wikipedia skill for Jarvis: Fetches and summarizes Wikipedia articles for a given topic.
"""

def wikipedia_skill(user_input, conversation_history=None, search_skill=None):
    """
    Fetches and summarizes a Wikipedia article for the given topic.
    Usage: wikipedia <topic>
    """
    import requests
    import re
    topic = user_input[len("wikipedia"):].strip()
    if not topic:
        return "Please provide a topic to look up on Wikipedia."
    # Use Wikipedia API
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'extract' in data:
                return data['extract']
            elif 'description' in data:
                return data['description']
            else:
                return f"No summary found for '{topic}'."
        else:
            return f"Wikipedia API error: {resp.status_code}"
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"

def register(jarvis):
    jarvis.register_skill("wikipedia", wikipedia_skill)
