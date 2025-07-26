"""
Summarization skill for Jarvis: Summarizes long text or web content using LLM or search.
"""

def summarize_skill(user_input, conversation_history=None, search_skill=None):
    """
    Summarizes the provided text or the result of a search.
    Usage: summarize <text or topic>
    """
    import re
    # Extract the text to summarize
    text = user_input[len("summarize"):].strip()
    if not text and search_skill:
        # If no text, try to search for the topic
        topic = user_input.strip()
        search_result = search_skill(f"search {topic}") if search_skill else None
        if search_result:
            text = search_result
    if not text:
        return "Please provide text or a topic to summarize."
    # Use LLM if available
    try:
        from importlib import import_module
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        if jarvis_instance and "ask" in jarvis_instance.skills:
            prompt = f"Summarize the following in a concise paragraph:\n{text}"
            return jarvis_instance.skills["ask"](f"ask {prompt}", conversation_history=conversation_history, search_skill=search_skill)
    except Exception:
        pass
    # Fallback: simple truncation
    return text[:300] + ("..." if len(text) > 300 else "")

def register(jarvis):
    jarvis.register_skill("summarize", summarize_skill)
