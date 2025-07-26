"""
LLM Plugin skill for Jarvis: Use an external LLM for advanced Q&A or code generation.
Extend this module to connect to OpenAI, local models, or other LLM APIs.
"""

def llm_plugin_skill(user_input, conversation_history=None):
    """
    Use an external LLM for advanced tasks. Usage: 'llm <your prompt>', 'ask llm <question>'
    """
    try:
        # Placeholder: In production, call an LLM API here
        prompt = user_input.replace('llm', '', 1).strip()
        # Example: response = call_openai_api(prompt)
        response = f"[LLM Plugin] Pretend response for: '{prompt}'"
        return response
    except Exception as e:
        return f"Sorry, the LLM plugin failed. ({e})"

def register(jarvis):
    jarvis.register_skill('llm_plugin', llm_plugin_skill)
