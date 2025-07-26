"""
Offline search skill for Jarvis: Looks up answers from a local knowledge base (text file or dictionary).
Usage: offline_search <query>
"""
import os

# Example: simple local knowledge base (can be replaced with a file/database)
KNOWLEDGE_BASE = {
    "noakhali science and technology university": "Noakhali Science and Technology University (NSTU) is a public university in Noakhali, Bangladesh, known for its science and technology programs.",
    "python": "Python is a popular high-level programming language known for its readability and versatility.",
    # Add more Q&A pairs as needed
}

def offline_search_skill(user_input):
    # Accept any user input, not just those prefixed with 'offline_search'
    query = user_input.strip().lower()
    if not query:
        return "Please provide a search query."
    # Simple keyword match
    for key, value in KNOWLEDGE_BASE.items():
        if key in query or query in key:
            return value
    return "Sorry, I couldn't find an offline answer for that."

def register(jarvis):
    jarvis.register_skill("offline_search", offline_search_skill)
