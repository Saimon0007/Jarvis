import sys
import importlib
from pathlib import Path
import sqlite3
import threading
import logging

import sqlite3

from skills import auto_ingest
from context_aware_manager import ContextAwareManager

# Set global logging level to WARNING to suppress most info/debug logs
logging.basicConfig(level=logging.WARNING)

# Suppress specific noisy libraries (optional, for more control)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class Jarvis:
    def fallback_chain(self, user_input):
        """
        Try all LLMs, search, and auto-ingest in parallel for every query.
        """
        if "multi_llm_search_autoingest" in self.skills:
            return self.skills["multi_llm_search_autoingest"](
                user_input,
                conversation_history=self.conversation_history,
                skills=self.skills
            )

        return None
    def reload_skills(self):
        """
        Reload all skills from the skills directory.
        """
        import importlib
        self.skills.clear()
        skills_path = Path(__file__).parent / 'skills'
        for skill_file in skills_path.glob('*.py'):
            if skill_file.name.startswith('_'):
                continue
            module_name = f'skills.{skill_file.stem}'
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)
            if hasattr(module, 'register'):
                module.register(self)

    def unload_skill(self, skill_name):
        """
        Unload a specific skill by name.
        """
        if skill_name in self.skills:
            del self.skills[skill_name]
            return f"Skill '{skill_name}' unloaded."
        return f"Skill '{skill_name}' not found."
    def suggest_similar_skill(self, user_input):
        """
        Suggest the closest matching skill name to the user input.
        """
        from difflib import get_close_matches
        words = user_input.lower().split()
        if not words:
            return None
        matches = get_close_matches(words[0], self.skills.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        return None
    def skill_help(self, skill_name):
        """
        Return the docstring or usage instructions for a given skill.
        """
        func = self.skills.get(skill_name)
        if not func:
            return f"No such skill: {skill_name}"
        doc = None
        if hasattr(func, "__doc__") and func.__doc__:
            doc = func.__doc__
        else:
            mod = getattr(func, "__module__", None)
            if mod:
                try:
                    imported = __import__(mod)
                    doc = getattr(imported, "__doc__", None)
                except Exception:
                    pass
        if doc:
            return f"Help for '{skill_name}':\n{doc.strip()}"
        return f"No help available for '{skill_name}'."
    def classify_intent(self, user_input):
        """
        Enhanced intent classifier: maps user input to a skill name if possible.
        Detects questions and routes them to the LLM skill.
        Returns the skill name or None.
        """
        text = user_input.lower().strip()
        
        # Direct skill commands take precedence
        if text.split()[0] in self.skills:
            return text.split()[0]
            
        # Detect questions and route to LLM
        question_starters = {
            "what", "who", "where", "when", "why", "how", "can", "could", 
            "would", "should", "is", "are", "was", "were", "will", "do", "does"
        }
        
        # Check if input starts with a question word
        if text.split()[0] in question_starters:
            return "ask"
            
        # Check for question marks
        if "?" in text:
            return "ask"
        
        # Map keywords to skills (extend as needed)
        from intent_config import INTENT_KEYWORDS, MULTITASK_MAP
        intent_keywords = INTENT_KEYWORDS
        user_input_lower = user_input.lower()
        for skill, keywords in intent_keywords.items():
            for kw in keywords:
                if kw in user_input_lower and skill in self.skills:
                    return skill
        # Multi-task LLM routing: if user input matches a multi-task keyword, route to ask skill with instruction
        multitask_map = {
            "summarize": "Summarize the following:",
            "code": "Write code for the following:",
            "qa": "Answer the following question:",
        }
        for task, instr in multitask_map.items():
            for kw in intent_keywords.get(task, []):
                if kw in user_input_lower and "ask" in self.skills:
                    # Prepend instruction for LLM prompt
                    def routed_ask(user_input, conversation_history=None, search_skill=None, instr=instr):
                        prompt = f"{instr} {user_input}"
                        return self.skills["ask"](f"ask {prompt}", conversation_history=conversation_history, search_skill=search_skill)
                    return routed_ask
        return None
    def list_skills(self):
        """
        Return a summary of all available skills and their descriptions.
        """
        summaries = []
        for name, func in self.skills.items():
            # Try to get docstring from the function or its module
            doc = None
            if hasattr(func, "__doc__") and func.__doc__:
                doc = func.__doc__
            else:
                mod = getattr(func, "__module__", None)
                if mod:
                    try:
                        imported = __import__(mod)
                        doc = getattr(imported, "__doc__", None)
                    except Exception:
                        pass
            if doc:
                first_line = doc.strip().split("\n")[0]
                summaries.append(f"- {name}: {first_line}")
            else:
                summaries.append(f"- {name}")
        if not summaries:
            return "I currently have no skills loaded."
        return "Here are my available skills:\n" + "\n".join(summaries)

    def discover_and_register_skill_modules(self):
        """
        Dynamically discover and register all skill modules in the skills directory.
        """
        import importlib.util
        skills_path = Path(__file__).parent / 'skills'
        discovered_skills = []
        for skill_file in skills_path.glob('*.py'):
            if skill_file.name.startswith('_'):
                continue

            module_name = f'skills.{skill_file.stem}'
            try:
                spec = importlib.util.spec_from_file_location(module_name, skill_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'register'):
                    module.register(self)
                    discovered_skills.append(module_name)
            except Exception as e:
                logging.warning(f'Failed to load skill module {module_name}: {e}')

        logging.info(f'Discovered and registered skills: {discovered_skills}')

    def initialize(self):
        """
        Initialize system components and auto-discover skills.
        """
        self.discover_and_register_skill_modules()
    def route_natural_language(self, user_input):
        """
        Route general knowledge or AI-related questions to LLM (ask) and search skill (Google/web) in parallel.
        If both return results, combine them into a single best answer. If only one returns, use that. If neither, show fallback.
        """
        llm_response = None
        search_response = None
        # Try self-learned knowledge first
        if "learn" in self.skills:
            learned = self.skills["learn"](user_input, conversation_history=self.conversation_history)
            if learned:
                return learned

        # Query available skills in parallel for best response
        gemini_response = None
        if "gemini" in self.skills:
            gemini_response = self.skills["gemini"](user_input, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
        if "ask" in self.skills:
            # No need to add "ask" prefix since the LLM skill will handle it
            llm_response = self.skills["ask"](user_input, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
        if "search" in self.skills and not (gemini_response or llm_response):
            # Only search if other skills didn't provide an answer
            search_response = self.skills["search"](f"search {user_input}")

        # Clean up responses
        def is_valid_llm(resp):
            return resp and not resp.lower().startswith("llm request failed") and not resp.lower().startswith("sorry")
        def is_valid_search(resp):
            return resp and not resp.lower().startswith("search failed") and not resp.lower().startswith("sorry")
        def is_valid_gemini(resp):
            return resp and not resp.lower().startswith("gemini api error") and not resp.lower().startswith("sorry")

        valid_gemini = is_valid_gemini(gemini_response)
        valid_llm = is_valid_llm(llm_response)
        valid_search = is_valid_search(search_response)

        # Combine if both are valid, with professional formatting
        if valid_gemini:
            if valid_search:
                # If Gemini response already includes search info, avoid duplication
                if search_response.strip() in gemini_response:
                    return self._format_response(gemini_response)
                combined = self._format_response(
                    f"{gemini_response.strip()}\n\n-----------------------------\nWeb Insight:\n{search_response.strip()}"
                )
                return combined
            return self._format_response(gemini_response)
        elif valid_llm:
            return self._format_response(llm_response)
        elif valid_search:
            return self._format_response(search_response)
    def _format_response(self, text):
        """
        Format Jarvis responses in a professional, well-structured way.
        - Adds a header, trims whitespace, and ensures clear separation of sections.
        """
        text = text.strip()
        # Add a professional header if not already present
        if not text.lower().startswith("jarvis:"):
            text = f"Jarvis:\n{text}"
        return text
    def voice_authenticate_user(self):
        import speech_recognition as sr
        import os
        creator_username = "Dell PC"  # Set your Windows username here
        passphrase = "open_sesame"    # Set your secret passphrase here
        current_user = os.getlogin()
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print(f"Jarvis: Hello {current_user}, please say the passphrase to access advanced features.")
        for attempt in range(3):
            with mic as source:
                print("Jarvis: Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
            try:
                spoken = recognizer.recognize_google(audio)
                print(f"You said: {spoken}")
                if spoken.strip().lower() == passphrase.lower():
                    if current_user == creator_username:
                        print("Jarvis: Welcome back, boss! Ready to assist you like always.")
                        return "boss"
                    else:
                        print(f"Jarvis: Welcome, {current_user}! You have standard access.")
                        return current_user
                else:
                    print("Jarvis: Sorry, that's not the right passphrase.")
            except sr.UnknownValueError:
                print("Jarvis: Sorry, I could not understand your speech.")
            except sr.RequestError as e:
                print(f"Jarvis: Could not request results; {e}")
        print("Jarvis: Access denied. Goodbye!")
        return None
    def __init__(self):
        self.skills = {}
        self.conversation_history = []  # Stores (role, message) tuples
        self.user_name = None  # User's preferred name
        self.db_path = "chat_history.db"
        self.db_lock = threading.Lock()  # Lock for thread-safe DB access
        self._init_db()
        # Initialize context-aware manager
        try:
            self.context_manager = ContextAwareManager()
        except Exception as e:
            logging.warning(f"Failed to initialize ContextAwareManager: {e}")
            self.context_manager = None
        self.load_skills()

    def _init_db(self):
        """Initialize the SQLite database for chat history."""
        # Create initial connection to set up the database schema
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS chats (user_id TEXT, message TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
        conn.close()

    def save_interaction(self, user_id, message, response):
        """Save a user interaction to the database."""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chats (user_id, message, response) VALUES (?, ?, ?)",
                (user_id, message, response),
            )
            conn.commit()
            conn.close()

    def get_recent_history(self, user_id, limit=20):
        """
        Retrieve the most recent conversation history for the user.
        Returns a list of (message, response) tuples, most recent last.
        """
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message, response FROM chats WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()
            # Return in chronological order
            return rows[::-1]

    def load_skills(self):
        skills_path = Path(__file__).parent / 'skills'
        if not skills_path.exists():
            skills_path.mkdir()
        for skill_file in skills_path.glob('*.py'):
            if skill_file.name.startswith('_'):
                continue
            module_name = f'skills.{skill_file.stem}'
            module = importlib.import_module(module_name)
            if hasattr(module, 'register'):
                module.register(self)
        # Prefer code skill for code-related queries
        if "code" in self.skills:
            self.skills["generate code"] = self.skills["code"]
            self.skills["explain code"] = self.skills["code"]

    def register_skill(self, name, func):
        self.skills[name] = func


    def authenticate_user(self):
        import getpass
        import os
        creator_username = "Dell PC"  # Set your Windows username here
        passphrase = "open_sesame"    # Set your secret passphrase here
        current_user = os.getlogin()
        print(f"Jarvis: Hello {current_user}, please enter the passphrase to access advanced features.")
        for _ in range(3):
            entered = getpass.getpass("Passphrase: ")
            if entered == passphrase:
                if current_user == creator_username:
                    print("Jarvis: Welcome back, boss! Ready to assist you like always.")
                    return "boss"
                else:
                    print(f"Jarvis: Welcome, {current_user}! You have standard access.")
                    return current_user
            else:
                print("Jarvis: Sorry, that's not the right passphrase.")
        print("Jarvis: Access denied. Goodbye!")
        return None

    def run(self):
        catchphrases = [
            "What's up? Ready to roll!",
            "Hey there, let's make some magic happen!",
            "Yo! How can I help you today?",
            "At your service! What's on your mind?",
            "Let's get things done!"
        ]
        import random
        self.user_name = None
        print(random.choice([
            "Jarvis here! Just say the word if you need anything.",
            "Hey! Jarvis online. How can I brighten your day?",
            "Yo! It's Jarvis. What's the plan?"
        ]))
        last_skill = None
        user_id = "default_user"
        while True:
            try:
                prompt_name = self.user_name if self.user_name else "you"
                user_id = self.user_name if self.user_name else "default_user"
                user_input = input(f'{prompt_name}: ').strip()
                if user_input.lower() in ('exit', 'quit'):
                    print(f"Jarvis: Catch you later, {prompt_name}! ✌️")
                    break
                lowered = user_input.lower()
                if (lowered.startswith("call me ") or lowered.startswith("my name is ")):
                    if lowered.startswith("call me "):
                        new_name = user_input[8:].strip()
                    else:
                        new_name = user_input[11:].strip()
                    if new_name:
                        self.user_name = new_name
                        print(f"Jarvis: Sweet! I'll call you {self.user_name} from now on.")
                        continue
                if lowered in ("authenticate", "who am i", "who are you talking to", "verify me"):
                    print("Jarvis: Let's do a quick check!")
                    print("Jarvis: Choose authentication method:")
                    print("1. Text passphrase (type 1)")
                    print("2. Voice passphrase (type 2)")
                    method = input("Enter 1 or 2: ").strip()
                    if method == "2":
                        user_role = self.voice_authenticate_user()
                    else:
                        user_role = self.authenticate_user()
                    if user_role:
                        self.user_name = user_role
                        print(f"Jarvis: Welcome, {self.user_name}! You're all set.")
                    else:
                        print("Jarvis: Sorry, couldn't verify you. Let's keep it casual!")
                    continue
                # --- MEMORY: Retrieve recent conversation history for context window ---
                recent_history = self.get_recent_history(user_id, limit=20)
                self.conversation_history = []
                for msg, resp in recent_history:
                    self.conversation_history.append((prompt_name, msg))
                    self.conversation_history.append(("Jarvis", resp))
                self.conversation_history.append((prompt_name, user_input))
                handled = False
                # Detect if user is asking for more details
                detail_phrases = [
                    "explain in more details", "more details", "explain in detail", "explain in greater detail", "explain further", "describe in detail", "elaborate", "go deeper", "tell me more", "expand", "give more details", "describe in more detail"
                ]
                is_detail_request = any(phrase in lowered for phrase in detail_phrases)
                if is_detail_request:
                    # Try to get the last valid non-user message as the topic
                    last_topic = None
                    for role, msg in reversed(self.conversation_history[:-1]):
                        if role == "Jarvis" and msg and not msg.lower().startswith("jarvis: sorry"):
                            last_topic = msg
                            break
                    if not last_topic:
                        # Fallback: try to get the last user input
                        for role, msg in reversed(self.conversation_history[:-1]):
                            if role == prompt_name and msg:
                                last_topic = msg
                                break
                    if last_topic:
                        detail_query = f"Explain in more detail: {last_topic}"
                        response = None
                        if "learn" in self.skills:
                            response = self.skills["learn"](detail_query, conversation_history=self.conversation_history)
                        if (not response or response.lower().startswith("i couldn't")) and "gemini" in self.skills:
                            response = self.skills["gemini"](detail_query, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
                        if (not response or response.lower().startswith("gemini api error")) and "ask" in self.skills:
                            response = self.skills["ask"](detail_query, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
                        if (not response or response.lower().startswith("llm request failed")) and "search" in self.skills:
                            response = self.skills["search"](detail_query)
                        if not response:
                            response = "Sorry, I couldn't find more details right now."
                        print(f"Jarvis: {response}")
                        self.conversation_history.append(("Jarvis", response))
                        self.save_interaction(user_id, user_input, response)
                        handled = True
                    else:
                        response = "Sorry, I couldn't find a previous answer or topic to elaborate on."
                        print(f"Jarvis: {response}")
                        self.conversation_history.append(("Jarvis", response))
                        self.save_interaction(user_id, user_input, response)
                        handled = True
                # Always combine LLM and search for general questions
                if not handled and (user_input.endswith("?") or any(w in lowered for w in ["what", "who", "when", "where", "how", "explain", "define", "describe"])):
                    response = self.route_natural_language(user_input)
                    if not response:
                        response = "Sorry, I couldn't find an answer to that. Try rephrasing, or check if my skills are configured."
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    # Update context-aware manager
                    if self.context_manager:
                        try:
                            self.context_manager.update_context(user_id, user_input, response)
                        except Exception as e:
                            logging.warning(f"Failed to update context: {e}")
                    handled = True
                # Multi-turn follow-up: if input is a follow-up, route to last_skill
                followup_phrases = ["and ", "what about", "how about", "also", "more", "continue", "next"]
                is_followup = any(user_input.lower().startswith(p) for p in followup_phrases)
                if not handled and is_followup and last_skill and last_skill in self.skills:
                    func = self.skills[last_skill]
                    try:
                        if last_skill == "ask":
                            response = func(user_input, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
                        else:
                            response = func(user_input, conversation_history=self.conversation_history)
                    except TypeError:
                        response = func(user_input)
                    if isinstance(response, str):
                        if not response.lower().startswith("jarvis:"):
                            response = f"Jarvis: {response}"
                        response = response.replace("boss", prompt_name)
                        response = random.choice(catchphrases) + " " + response if random.random() < 0.2 else response
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    # Update context-aware manager
                    if self.context_manager:
                        try:
                            self.context_manager.update_context(user_id, user_input, response)
                        except Exception as e:
                            logging.warning(f"Failed to update context: {e}")
                    handled = True
                # Direct command match (stricter: must be exact or valid command)
                if not handled:
                    for name, func in self.skills.items():
                        if user_input.lower() == name or user_input.lower().startswith(name + " "):
                            try:
                                if name == "ask":
                                    response = func(user_input, conversation_history=self.conversation_history, search_skill=self.skills.get("search"))
                                else:
                                    response = func(user_input, conversation_history=self.conversation_history)
                            except TypeError:
                                response = func(user_input)
                            if isinstance(response, str):
                                if not response.lower().startswith("jarvis:"):
                                    response = f"Jarvis: {response}"
                                response = response.replace("boss", prompt_name)
                                response = random.choice(catchphrases) + " " + response if random.random() < 0.2 else response
                            print(response)
                            self.conversation_history.append(("Jarvis", response))
                            self.save_interaction(user_id, user_input, response)
                            # Update context-aware manager
                            if self.context_manager:
                                try:
                                    self.context_manager.update_context(user_id, user_input, response)
                                except Exception as e:
                                    logging.warning(f"Failed to update context: {e}")
                            last_skill = name
                            handled = True
                            break
                # Intent classification if not handled
                if not handled:
                    intent_skill = self.classify_intent(user_input)
                    if intent_skill:
                        func = self.skills[intent_skill]
                        try:
                            response = func(user_input, conversation_history=self.conversation_history)
                        except TypeError:
                            response = func(user_input)
                        if isinstance(response, str):
                            if not response.lower().startswith("jarvis:"):
                                response = f"Jarvis: {response}"
                        response = response.replace("boss", prompt_name)
                        response = random.choice(catchphrases) + " " + response if random.random() < 0.2 else response
                        print(response)
                        self.conversation_history.append(("Jarvis", response))
                        self.save_interaction(user_id, user_input, response)
                        # Update context-aware manager
                        if self.context_manager:
                            try:
                                self.context_manager.update_context(user_id, user_input, response)
                            except Exception as e:
                                logging.warning(f"Failed to update context: {e}")
                        last_skill = intent_skill
                        handled = True
                # Hot-reload all skills
                if not handled and user_input.strip().lower() == "reload skills":
                    self.reload_skills()
                    response = "Jarvis: Skills reloaded."
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    handled = True
                # Unload a specific skill
                if not handled and user_input.strip().lower().startswith("unload "):
                    skill_name = user_input.strip().split(maxsplit=1)[1].lower()
                    response = f"Jarvis: {self.unload_skill(skill_name)}"
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    handled = True
                # Help for a specific skill: help <skill>
                if not handled and user_input.strip().lower().startswith("help "):
                    skill_name = user_input.strip().split(maxsplit=1)[1].lower()
                    response = f"Jarvis: {self.skill_help(skill_name)}"
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    handled = True
                # List skills if user asks what Jarvis can do
                if not handled and user_input.strip().lower() in [
                    "what can you do?", "what can you do", "what else can you do?", "what else can you do", "help", "skills", "abilities", "list skills"]:
                    response = f"Jarvis: {self.list_skills()}"
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
                    handled = True
                # Try modular fallback chain if not handled
                if not handled:
                    fallback_response = self.fallback_chain(user_input)
                    if fallback_response:
                        response = f"Jarvis: {fallback_response}"
                        response = random.choice(catchphrases) + " " + response if random.random() < 0.2 else response
                    else:
                        fallback_lines = [
                            f"Just hanging out! Let me know if you want to do something cool, {prompt_name}.",
                            f"I'm here if you need anything, {prompt_name}.",
                            f"Sounds good! If you want to try a skill, just say so.",
                            f"Alright! If you want to solve some maths, just type the problem.",
                            f"No worries, {prompt_name}. I'm always ready!"
                        ]
                        response = f"Jarvis: {random.choice(fallback_lines)}"
                    print(response)
                    self.conversation_history.append(("Jarvis", response))
                    self.save_interaction(user_id, user_input, response)
            except (KeyboardInterrupt, EOFError):
                print(f'\nJarvis: See you soon!')
                break

if __name__ == '__main__':
    jarvis = Jarvis()
    # Auto-ingest knowledge on startup
    try:
        # Example topics; you can expand or load from a file
        topics = [
            "artificial intelligence", "machine learning", "latest technology news", "world news", "quantum computing", "blockchain", "climate change", "global economy"
        ]
        if hasattr(jarvis, 'skills') and 'auto_ingest' in jarvis.skills:
            jarvis.skills['auto_ingest'](topics)
    except Exception as e:
        print(f"[Auto-ingest error] {e}")
    jarvis.run()
