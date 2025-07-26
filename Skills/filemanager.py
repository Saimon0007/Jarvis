"""
File Manager skill for Jarvis: Search and list files in a directory.
Extend this module for more file operations (open, move, summarize, etc).
"""
import os

def filemanager_skill(user_input, conversation_history=None):
    """
    List files in a directory. Usage: 'list files', 'show my files', 'list files in <folder>'
    """
    try:
        # Parse folder from user input (very basic)
        folder = '.'
        tokens = user_input.lower().split()
        if 'in' in tokens:
            idx = tokens.index('in')
            if idx + 1 < len(tokens):
                folder = tokens[idx + 1]
        files = os.listdir(folder)
        if not files:
            return f"No files found in '{folder}'."
        file_lines = [f"- {f}" for f in files]
        return f"Files in '{folder}':\n" + "\n".join(file_lines)
    except Exception as e:
        return f"Sorry, I couldn't list files. ({e})"

def register(jarvis):
    jarvis.register_skill('filemanager', filemanager_skill)
