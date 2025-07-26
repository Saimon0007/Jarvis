"""
Gemini LLM skill for Jarvis: Uses Gemini API for LLM responses.
"""
import os
import requests

def gemini_skill(user_input, conversation_history=None, search_skill=None):
    """
    Uses Gemini API to generate a response for the given user input.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key not found. Please set GEMINI_API_KEY in your .env file."
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_input}]}]
    }
    params = {"key": api_key}
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        # Gemini API returns candidates[0].content.parts[0].text
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        return "Gemini API did not return a valid response."
    except Exception as e:
        return f"Gemini API error: {e}"

def register(jarvis):
    jarvis.register_skill("gemini", gemini_skill)
