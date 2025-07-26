"""
Automation skill for Jarvis: Run custom Python scripts or shell commands.
Extend this module for more advanced automation workflows.
"""
import subprocess

def automation_skill(user_input, conversation_history=None):
    """
    Run a shell command. Usage: 'run <command>', 'execute <script.py>'
    """
    try:
        tokens = user_input.lower().split()
        if tokens[0] in ('run', 'execute') and len(tokens) > 1:
            cmd = user_input[len(tokens[0]):].strip()
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            return f"Command output:\n{output}"
        return "Please specify a command to run."
    except Exception as e:
        return f"Sorry, I couldn't run that command. ({e})"

def register(jarvis):
    jarvis.register_skill('automation', automation_skill)
