"""
WolframAlpha skill for Jarvis: Answers advanced math, science, and factual queries using the WolframAlpha API.
"""

def wolframalpha_skill(user_input, conversation_history=None, search_skill=None):
    """
    Answers questions using WolframAlpha. Usage: wolfram <question>
    Requires WOLFRAMALPHA_APPID in environment or config.
    """
    import os
    import requests
    import urllib.parse
    import xml.etree.ElementTree as ET
    appid = os.getenv("WOLFRAMALPHA_APPID")
    if not appid:
        try:
            from config import WOLFRAMALPHA_APPID as appid
        except Exception:
            return "WolframAlpha AppID not set. Please set WOLFRAMALPHA_APPID in your environment or config."
    question = user_input[len("wolfram"):].strip()
    if not question:
        # Try to use last user question from conversation history
        if conversation_history:
            for role, msg in reversed(conversation_history):
                if role.lower() != "jarvis" and msg.strip():
                    question = msg.strip()
                    break
        if not question:
            return "Please provide a question for WolframAlpha."

    # Use the full API for richer results
    url = f"https://api.wolframalpha.com/v2/query?appid={appid}&input={urllib.parse.quote(question)}&output=XML&podstate=Step-by-step%20solution"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return f"WolframAlpha API error: {resp.status_code}. Try rephrasing your question or check your AppID."
        root = ET.fromstring(resp.text)
        if root.attrib.get('error') == 'true':
            return "WolframAlpha could not process your question. Please clarify or try a different query."
        pods = root.findall('.//pod')
        if not pods:
            # Suggest other skills if no answer
            suggestion = "I couldn't find an answer. Would you like me to search the web or use another skill?"
            if search_skill:
                web_result = search_skill(f"search {question}")
                if web_result:
                    suggestion += f"\nWeb search result: {web_result}"
            return suggestion
        # Build a conversational answer
        answer = []
        title_main = None
        for pod in pods:
            title = pod.attrib.get('title', '')
            plaintext = pod.findtext('subpod/plaintext')
            img = pod.find('subpod/img')
            if plaintext and plaintext.strip():
                if not title_main:
                    title_main = title
                answer.append(f"**{title}:** {plaintext.strip()}")
            elif img is not None and img.attrib.get('src'):
                answer.append(f"[Image: {title}]({img.attrib['src']})")
        if not answer:
            # If no plaintext, try to show at least an image
            for pod in pods:
                img = pod.find('subpod/img')
                if img is not None and img.attrib.get('src'):
                    answer.append(f"[Image: {pod.attrib.get('title','')}]({img.attrib['src']})")
        if not answer:
            return "Sorry, I couldn't find a clear answer. Could you clarify your question?"

        # If user asked for more details, try to expand
        if conversation_history:
            last_user = None
            for role, msg in reversed(conversation_history):
                if role.lower() != "jarvis" and msg.strip():
                    last_user = msg.strip().lower()
                    break
            detail_phrases = [
                "explain in more details", "more details", "explain in detail", "explain in greater detail", "explain further", "describe in detail", "elaborate", "go deeper", "tell me more", "expand", "give more details", "describe in more detail"
            ]
            if last_user and any(phrase in last_user for phrase in detail_phrases):
                # Try to find step-by-step or deeper pods
                for pod in pods:
                    if "step-by-step" in pod.attrib.get('title','').lower():
                        step_text = pod.findtext('subpod/plaintext')
                        if step_text:
                            answer.append(f"\n**Step-by-step solution:**\n{step_text.strip()}")
        # Conversational formatting
        response = f"Here's what I found for your question:\n\n" + "\n\n".join(answer)
        # Suggest follow-up
        response += "\n\nIf you need more details or want to ask a follow-up, just let me know!"
        return response
    except Exception as e:
        return f"WolframAlpha lookup failed: {e}\nIf this keeps happening, try rephrasing your question or use another skill."

def register(jarvis):
    jarvis.register_skill("wolfram", wolframalpha_skill)
