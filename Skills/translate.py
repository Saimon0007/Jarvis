"""
Translation skill for Jarvis: Translates text between languages using LLM or search.
"""

def translate_skill(user_input, conversation_history=None, search_skill=None):
    """
    Translates the provided text to the target language.
    Usage: translate <text> to <language>
    """
    import re
    match = re.match(r"translate (.+) to ([a-zA-Z\s]+)", user_input.strip(), re.IGNORECASE)
    if not match:
        return "Please use the format: translate <text> to <language>."
    text, language = match.groups()
    text, language = text.strip(), language.strip()
    # Use LLM if available
    try:
        from importlib import import_module
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        if jarvis_instance and "ask" in jarvis_instance.skills:
            prompt = f"Translate the following to {language}:\n{text}"
            return jarvis_instance.skills["ask"](f"ask {prompt}", conversation_history=conversation_history, search_skill=search_skill)
    except Exception:
        pass
    # Fallback: echo
    return f"[Translation to {language} not available. Here is the original text:] {text}"

def register(jarvis):
    jarvis.register_skill("translate", translate_skill)
