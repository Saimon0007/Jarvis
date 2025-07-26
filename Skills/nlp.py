import re
import logging
from difflib import get_close_matches
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from collections import defaultdict
import json
from pathlib import Path

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class NLPConfig:
    """Configuration class for NLP settings."""
    def __init__(
        self,
        skill_match_cutoff: float = 0.7,
        intent_config_path: Optional[str] = None,
        stop_words: Optional[Set[str]] = None
    ):
        self.skill_match_cutoff = skill_match_cutoff
        self.intent_config_path = intent_config_path
        self.stop_words = stop_words or {"a", "an", "the", "is", "are", "and", "or"}

class AdvancedNLP:
    """An enhanced NLP processor for intent detection and skill matching."""
    
    def __init__(self, skills: Dict[str, Any], config: NLPConfig = None):
        """
        Initialize the NLP processor.
        
        Args:
            skills: Dictionary of available skills.
            config: Optional NLP configuration object.
        """
        self.config = config or NLPConfig()
        self.known_skills = list(skills.keys())
        self.intent_patterns: List[Tuple[re.Pattern, Callable[[], str]]] = []
        self.intent_cache: Dict[str, str] = {}
        self._initialize_intents()
        logger.info(f"Initialized NLP with {len(self.known_skills)} skills")

    def _load_intent_config(self) -> List[Dict[str, str]]:
        """Load intent patterns from a configuration file if provided."""
        if not self.config.intent_config_path:
            return []
        try:
            config_path = Path(self.config.intent_config_path)
            if config_path.exists():
                with config_path.open("r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "intents" in data:
                        return data["intents"]
                    elif isinstance(data, list):
                        return data
                    else:
                        logger.error("Intent config file must be a list or a dict with an 'intents' key.")
                        return []
            logger.warning(f"Intent config file not found: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in intent config: {e}")
        return []

    def _build_intent_patterns(self) -> List[Tuple[re.Pattern, Callable[[], str]]]:
        """Build default and custom intent patterns."""
        patterns = [
            (
                re.compile(r"\b(hi|hello|hey|greetings)\b", re.IGNORECASE),
                lambda: "Hello! How can I assist you today?"
            ),
            (
                re.compile(r"how are you\b", re.IGNORECASE),
                lambda: "I'm an AI, running smoothly and ready to help!"
            ),
            (
                re.compile(r"\b(what.*can.*you.*do|help|abilities|skills)\b", re.IGNORECASE),
                lambda: f"I can help with: {', '.join(self.known_skills)}."
            )
        ]

        # Load custom intents from config
        for intent in self._load_intent_config():
            try:
                pattern = re.compile(intent["pattern"], re.IGNORECASE)
                response = lambda text=intent["response"]: text
                patterns.append((pattern, response))
            except re.error as e:
                logger.error(f"Invalid regex pattern in intent config: {e}")

        return patterns

    def _initialize_intents(self) -> None:
        """Initialize intent patterns and cache."""
        try:
            self.intent_patterns = self._build_intent_patterns()
        except Exception as e:
            logger.error(f"Failed to initialize intents: {e}")
            self.intent_patterns = []

    def _tokenize_input(self, user_input: str) -> List[str]:
        """
        Tokenize and preprocess user input.
        
        Args:
            user_input: Raw user input string.
        
        Returns:
            List of processed tokens.
        """
        # Basic tokenization (can be enhanced with nltk/spacy)
        tokens = re.findall(r"\w+", user_input.lower())
        return [t for t in tokens if t not in self.config.stop_words]

    def find_intent(self, user_input: str) -> Optional[str]:
        """
        Detect intent in user input.
        
        Args:
            user_input: User input string.
        
        Returns:
            Intent response if found, else None.
        """
        input_hash = hash(user_input.lower())
        if input_hash in self.intent_cache:
            return self.intent_cache[input_hash]

        try:
            for pattern, response in self.intent_patterns:
                if pattern.search(user_input):
                    result = response()
                    self.intent_cache[input_hash] = result
                    return result
        except Exception as e:
            logger.error(f"Error finding intent: {e}")
        return None

    def find_skill_in_input(self, user_input: str) -> Optional[str]:
        """
        Identify the most likely skill in user input.
        
        Args:
            user_input: User input string.
        
        Returns:
            Matched skill name if found, else None.
        """
        tokens = self._tokenize_input(user_input)
        skill_scores = defaultdict(float)

        for token in tokens:
            matches = get_close_matches(
                token,
                self.known_skills,
                n=3,
                cutoff=self.config.skill_match_cutoff
            )
            for match in matches:
                # Weight matches by token length to reduce false positives
                skill_scores[match] += len(token) / len(match)

        if skill_scores:
            best_match = max(skill_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Matched skill '{best_match}' with score {skill_scores[best_match]}")
            return best_match
        return None

    def process(self, user_input: str) -> str:
        """
        Process user input and return a response.
        
        Args:
            user_input: User input string.
        
        Returns:
            Response string.
        """
        if not isinstance(user_input, str) or not user_input.strip():
            logger.warning("Invalid or empty input received")
            return "Please provide a valid input."

        logger.info(f"Processing input: '{user_input}'")
        intent_response = self.find_intent(user_input)
        if intent_response:
            return intent_response

        suggested_skill = self.find_skill_in_input(user_input)
        if suggested_skill:
            return f"Did you mean to use the '{suggested_skill}' skill? Try using its exact name."

        return "I didn't understand that. Try 'help' to see what I can do."

def register(jarvis: Any) -> None:
    """
    Register the NLP skill with the Jarvis system.
    """
    if not hasattr(jarvis, 'skills') or not hasattr(jarvis, 'register_skill'):
        raise AttributeError("Jarvis must have a 'skills' dictionary and a 'register_skill' method.")
    # Always resolve the path to intents.json relative to the Jarvis root
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intent_path = os.path.join(base_dir, "intents.json")
    intent_path = os.path.normpath(intent_path)  # Ensure correct path separators
    config = NLPConfig(intent_config_path=intent_path)
    nlp = AdvancedNLP(jarvis.skills, config)
    jarvis.register_skill('nlp', nlp.process)
    logger.info("NLP skill registered successfully")