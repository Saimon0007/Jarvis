"""
Universal code skill for Jarvis: Answers questions, generates code, and explains programming concepts for all major languages using LLMs (Gemini, OpenAI, etc.).
"""
import os

def code_skill(user_input, conversation_history=None, search_skill=None):
    """
    Handles code generation, explanation, and programming Q&A for any language.
    Tries Gemini first, then other LLMs, then search if available.
    """
    jarvis = None
    # Try to get the jarvis instance from the conversation history if possible
    if conversation_history and hasattr(conversation_history, 'jarvis'):
        jarvis = conversation_history.jarvis
    # Compose a prompt for code generation/explanation
    prompt = f"You are a world-class coding assistant. {user_input}\n\nIf code is requested, provide clear, well-commented code in the requested language. If an explanation is needed, be concise and accurate."
    # Prefer Gemini
    response = None
    try:
        import importlib
        gemini = None
        if 'skills.gemini' in globals():
            gemini = globals()['skills.gemini']
        else:
            gemini = importlib.import_module('skills.gemini')
        if hasattr(gemini, 'gemini_skill'):
            response = gemini.gemini_skill(prompt, conversation_history=conversation_history, search_skill=search_skill)
    except Exception:
        pass
    # Try OpenAI or other LLMs if available
    if (not response or 'error' in str(response).lower()) and jarvis and 'ask' in jarvis.skills:
        response = jarvis.skills['ask'](prompt, conversation_history=conversation_history, search_skill=search_skill)
    # Try search as a last resort
    if (not response or 'error' in str(response).lower()) and search_skill:
        response = search_skill(f"{user_input} code example")
    if not response:
        return "Sorry, I couldn't generate or explain code for that right now."
    return response

def register(jarvis):
    jarvis.register_skill("code", code_skill)
