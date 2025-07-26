"""
Jarvis Web UI Server - Full-featured web interface for Jarvis AI Assistant
Serves the stunning UI and provides real-time chat functionality
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import logging
import os
import sys
import random
import inspect
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Jarvis

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)

# Initialize Jarvis instance
jarvis_instance = None

def get_jarvis():
    """Get or initialize Jarvis instance."""
    global jarvis_instance
    if jarvis_instance is None:
        jarvis_instance = Jarvis()
        logging.info("Jarvis initialized for web interface")
    return jarvis_instance

def call_skill_with_appropriate_params(skill_func, user_input, conversation_history=None, search_skill=None, skills=None):
    """Call a skill function with only the parameters it accepts."""
    try:
        # Get function signature
        sig = inspect.signature(skill_func)
        params = list(sig.parameters.keys())
        
        # Always pass user_input as the first parameter
        args = [user_input]
        kwargs = {}
        
        # Check which optional parameters the function accepts
        if 'conversation_history' in params and conversation_history is not None:
            kwargs['conversation_history'] = conversation_history
        
        if 'search_skill' in params and search_skill is not None:
            kwargs['search_skill'] = search_skill
            
        if 'skills' in params and skills is not None:
            kwargs['skills'] = skills
        
        # Call the function with appropriate parameters
        return skill_func(*args, **kwargs)
        
    except Exception as e:
        logging.error(f"Error calling skill function: {e}")
        # Fallback to basic call with just user_input
        try:
            return skill_func(user_input)
        except Exception as fallback_error:
            logging.error(f"Fallback skill call also failed: {fallback_error}")
            return "Sorry, I encountered an error processing that request."

def process_jarvis_message(message, user_id="web_user"):
    """Process message through Jarvis with simplified logic from main.py"""
    j = get_jarvis()
    
    # Get recent history for context
    recent_history = j.get_recent_history(user_id, limit=10)
    j.conversation_history = []
    for msg, resp in recent_history:
        j.conversation_history.append((user_id, msg))
        j.conversation_history.append(("Jarvis", resp))
    j.conversation_history.append((user_id, message))
    
    response = None
    
    # Check for special commands first
    lowered = message.lower().strip()
    
    if lowered in ['what can you do?', 'what can you do', 'help', 'skills', 'abilities']:
        response = j.list_skills()
    elif lowered == 'reload skills':
        j.reload_skills()
        response = "Skills reloaded successfully!"
    elif message.endswith('?') or any(w in lowered for w in ['what', 'who', 'when', 'where', 'how', 'explain', 'define', 'describe']):
        # Route questions to natural language processing
        response = j.route_natural_language(message)
    else:
        # Try direct skill matching
        handled = False
        for name, func in j.skills.items():
            if message.lower() == name or message.lower().startswith(name + " "):
                response = call_skill_with_appropriate_params(
                    func, 
                    message, 
                    conversation_history=j.conversation_history,
                    search_skill=j.skills.get("search"),
                    skills=j.skills
                )
                
                if isinstance(response, str):
                    if not response.lower().startswith("jarvis:"):
                        response = f"Jarvis: {response}"
                handled = True
                break
        
        # Try intent classification if not handled
        if not handled:
            intent_skill = j.classify_intent(message)
            if intent_skill and intent_skill in j.skills:
                try:
                    func = j.skills[intent_skill]
                    response = call_skill_with_appropriate_params(
                        func, 
                        message, 
                        conversation_history=j.conversation_history,
                        search_skill=j.skills.get("search"),
                        skills=j.skills
                    )
                    if isinstance(response, str) and not response.lower().startswith("jarvis:"):
                        response = f"Jarvis: {response}"
                except Exception as e:
                    logging.error(f"Error executing skill {intent_skill}: {e}")
                    response = f"Sorry, I encountered an error processing that request."
        
        # Fallback chain
        if not response:
            fallback_response = j.fallback_chain(message)
            if fallback_response:
                response = f"Jarvis: {fallback_response}"
            else:
                fallback_lines = [
                    "I'm here to help! Try asking me a question or use one of my skills.",
                    "Not sure what you're looking for? Try 'help' to see what I can do.",
                    "I didn't understand that, but I'm always learning! Try rephrasing your request.",
                    "Let me know how I can assist you today!"
                ]
                response = f"Jarvis: {random.choice(fallback_lines)}"
    
    # Clean up response
    if not response:
        response = "Jarvis: I'm not sure how to help with that. Try asking me something else!"
    
    # Save interaction
    j.save_interaction(user_id, message, response)
    
    return response

# Routes
@app.route('/')
def index():
    """Serve the main UI."""
    try:
        with open('static/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """<!DOCTYPE html>
<html><head><title>Jarvis UI Not Found</title></head>
<body>
<h1>Jarvis UI Files Not Found</h1>
<p>Please make sure the static files (index.html, styles.css, script.js) are in the 'static' folder.</p>
</body></html>""", 404

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    j = get_jarvis()
    return jsonify({
        "status": "healthy", 
        "service": "jarvis-web-ui",
        "skills_loaded": len(j.skills),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for interacting with Jarvis."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_message = data['message'].strip()
        user_id = data.get('user_id', 'web_user')
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        # Process the message through Jarvis
        response = process_jarvis_message(user_message, user_id)
        
        return jsonify({
            "response": response,
            "user_id": user_id,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/skills', methods=['GET'])
def list_skills():
    """List available skills."""
    try:
        j = get_jarvis()
        skills_list = list(j.skills.keys())
        return jsonify({"skills": skills_list, "count": len(skills_list)})
    except Exception as e:
        logging.error(f"Error listing skills: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get chat history for a user."""
    try:
        j = get_jarvis()
        limit = request.args.get('limit', 20, type=int)
        history = j.get_recent_history(user_id, limit)
        return jsonify({"history": history})
    except Exception as e:
        logging.error(f"Error getting history: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("    🤖 JARVIS AI Assistant - Web Interface")
    print("="*60)
    print("    Starting stunning web UI server...")
    print("    ") 
    print("    🌐 Open your browser and go to:")
    print("       http://localhost:5000")
    print("    ")
    print("    ✨ Features:")
    print("       • Stunning cyber-themed UI")
    print("       • Real-time chat with Jarvis")
    print("       • Voice input support")
    print("       • All Jarvis skills available")
    print("       • Mobile-responsive design")
    print("    ")
    print("    Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Jarvis Web UI server stopped. Goodbye!")
