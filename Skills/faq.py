"""
FAQ/Knowledge Base skill for Jarvis: Answer frequently asked questions and learn new ones.
Extend this module for persistent, self-learning FAQ.
"""
import json
import os

FAQ_FILE = os.path.join(os.path.dirname(__file__), 'faq_data.json')

def load_faq():
    if os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_faq(faq):
    with open(FAQ_FILE, 'w', encoding='utf-8') as f:
        json.dump(faq, f, indent=2)

def faq_skill(user_input, conversation_history=None):
    """
    Answer FAQs or add new Q&A pairs. Usage: 'faq <question>', 'add faq <question> | <answer>'
    """
    faq = load_faq()
    if user_input.lower().startswith('add faq '):
        try:
            _, rest = user_input.split('add faq ', 1)
            question, answer = rest.split('|', 1)
            question = question.strip().lower()
            answer = answer.strip()
            faq[question] = answer
            save_faq(faq)
            return f"FAQ added: '{question}'"
        except Exception:
            return "To add an FAQ, use: add faq <question> | <answer>"
    else:
        # Always try to answer any question, not just those prefixed with 'faq'
        question = user_input.strip().lower()
        if question in faq:
            return faq[question]
        # Try fuzzy match
        for q in faq:
            if question in q or q in question:
                return faq[q]
        return "No answer found in FAQ. To add one, use: add faq <question> | <answer>"

def register(jarvis):
    jarvis.register_skill('faq', faq_skill)
