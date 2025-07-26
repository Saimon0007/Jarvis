"""
Error explanation skill for Jarvis: Explains error messages and suggests possible fixes using LLM.
"""

def error_explain_skill(user_input, conversation_history=None, search_skill=None):
    """
    Explains error messages and suggests possible fixes.
    Usage: explain error <error message>
    """
    error_msg = user_input[len("explain error"):].strip()
    if not error_msg:
        return "Please provide an error message to explain."
    try:
        from importlib import import_module
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        if jarvis_instance and "ask" in jarvis_instance.skills:
            prompt = f"Explain this error and suggest possible fixes:\n{error_msg}"
            return jarvis_instance.skills["ask"](f"ask {prompt}", conversation_history=conversation_history, search_skill=search_skill)
    except Exception:
        pass
    return "Sorry, I couldn't explain the error right now."

def register(jarvis):
    jarvis.register_skill("explain error", error_explain_skill)
