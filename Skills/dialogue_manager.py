"""
Advanced Dialogue State Management for Jarvis: Provides sophisticated context-aware
conversation handling with memory, entity tracking, and intelligent follow-up processing.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts and tracks entities from conversation."""
    
    def __init__(self):
        self.entity_patterns = {
            'person': re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'),
            'location': re.compile(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'),
            'organization': re.compile(r'\b([A-Z][A-Z]+|[A-Z][a-z]+\s+Inc\.|[A-Z][a-z]+\s+Corp\.)\b'),
            'date': re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'),
            'time': re.compile(r'\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b', re.I),
            'number': re.compile(r'\b(\d+(?:\.\d+)?)\b'),
            'topic': re.compile(r'\babout\s+([a-z][a-zA-Z\s]+)\b', re.I)
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        entities = defaultdict(list)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type].extend(matches)
        
        return dict(entities)

class ConversationContext:
    """Manages conversation context and state."""
    
    def __init__(self, max_context_length: int = 20):
        self.max_context_length = max_context_length
        self.conversation_history = deque(maxlen=max_context_length)
        self.entities = defaultdict(set)
        self.topics = deque(maxlen=10)
        self.last_question_type = None
        self.last_skill_used = None
        self.pending_followups = []
        self.context_embeddings = {}
        
    def add_exchange(self, user_input: str, assistant_response: str, skill_used: str = None):
        """Add a conversation exchange to context."""
        timestamp = datetime.now()
        
        exchange = {
            'timestamp': timestamp,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'skill_used': skill_used,
            'entities': EntityExtractor().extract_entities(user_input)
        }
        
        self.conversation_history.append(exchange)
        
        # Update entity tracking
        for entity_type, entity_list in exchange['entities'].items():
            self.entities[entity_type].update(entity_list)
        
        # Update topic tracking
        if 'topic' in exchange['entities']:
            self.topics.extend(exchange['entities']['topic'])
        
        self.last_skill_used = skill_used
        
    def get_recent_context(self, num_exchanges: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        return list(self.conversation_history)[-num_exchanges:]
    
    def find_similar_context(self, current_input: str) -> List[Dict]:
        """Find similar previous exchanges based on content similarity."""
        current_words = set(current_input.lower().split())
        similar_exchanges = []
        
        for exchange in self.conversation_history:
            prev_words = set(exchange['user_input'].lower().split())
            similarity = len(current_words.intersection(prev_words)) / max(len(current_words), 1)
            
            if similarity > 0.3:  # Threshold for similarity
                similar_exchanges.append((exchange, similarity))
        
        # Sort by similarity
        similar_exchanges.sort(key=lambda x: x[1], reverse=True)
        return [exchange for exchange, _ in similar_exchanges[:3]]
    
    def get_context_for_followup(self, user_input: str) -> Dict[str, Any]:
        """Get relevant context for processing a follow-up question."""
        context = {
            'recent_exchanges': self.get_recent_context(3),
            'relevant_entities': dict(self.entities),
            'current_topics': list(self.topics)[-3:],
            'last_skill': self.last_skill_used,
            'similar_context': self.find_similar_context(user_input)
        }
        
        return context

class FollowUpDetector:
    """Detects and classifies follow-up questions."""
    
    def __init__(self):
        self.followup_patterns = {
            'clarification': [
                r'\bwhat do you mean\b',
                r'\bcan you explain\b',
                r'\bwhat does that mean\b',
                r'\bclarify\b',
                r'\bI don\'t understand\b'
            ],
            'elaboration': [
                r'\btell me more\b',
                r'\bmore details\b',
                r'\belaborate\b',
                r'\bexpand on\b',
                r'\bgo deeper\b',
                r'\bfurther\b'
            ],
            'example': [
                r'\bfor example\b',
                r'\bgive me an example\b',
                r'\bshow me\b',
                r'\blike what\b',
                r'\bsuch as\b'
            ],
            'comparison': [
                r'\bcompare\b',
                r'\bwhat about\b',
                r'\bhow about\b',
                r'\bversus\b',
                r'\bdifference\b'
            ],
            'continuation': [
                r'\band then\b',
                r'\bwhat next\b',
                r'\bafter that\b',
                r'\bcontinue\b',
                r'\bnext\b'
            ],
            'alternative': [
                r'\bor\b',
                r'\botherwise\b',
                r'\balternatively\b',
                r'\binstead\b',
                r'\bwhat if\b'
            ]
        }
        
        self.pronoun_references = re.compile(r'\b(it|that|this|they|them|those|these)\b', re.I)
    
    def detect_followup_type(self, user_input: str) -> Optional[str]:
        """Detect the type of follow-up question."""
        user_input_lower = user_input.lower()
        
        for followup_type, patterns in self.followup_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return followup_type
        
        # Check for pronoun references
        if self.pronoun_references.search(user_input):
            return 'reference'
        
        return None
    
    def is_followup(self, user_input: str, context: ConversationContext) -> bool:
        """Determine if input is a follow-up question."""
        followup_type = self.detect_followup_type(user_input)
        
        if followup_type:
            return True
        
        # Check if input is very short and contextual
        if len(user_input.split()) <= 3 and context.conversation_history:
            return True
        
        # Check if input references recent entities
        recent_entities = set()
        for exchange in context.get_recent_context(2):
            for entity_list in exchange['entities'].values():
                recent_entities.update(entity_list)
        
        input_words = set(user_input.lower().split())
        entity_overlap = len(input_words.intersection(recent_entities)) > 0
        
        return entity_overlap

class DialogueStateManager:
    """Main dialogue state management system."""
    
    def __init__(self):
        self.context = ConversationContext()
        self.followup_detector = FollowUpDetector()
        self.response_cache = {}
        
    def process_input(self, user_input: str, conversation_history: List = None) -> Dict[str, Any]:
        """Process user input with full context awareness."""
        
        # Update context from conversation history if provided
        if conversation_history:
            self._sync_with_history(conversation_history)
        
        is_followup = self.followup_detector.is_followup(user_input, self.context)
        followup_type = self.followup_detector.detect_followup_type(user_input) if is_followup else None
        
        context_info = self.context.get_context_for_followup(user_input) if is_followup else {}
        
        result = {
            'is_followup': is_followup,
            'followup_type': followup_type,
            'context': context_info,
            'processed_input': self._enhance_input_with_context(user_input, context_info) if is_followup else user_input,
            'recommended_skills': self._recommend_skills(user_input, is_followup, followup_type),
            'confidence': self._calculate_confidence(user_input, is_followup, context_info)
        }
        
        return result
    
    def _sync_with_history(self, conversation_history: List):
        """Sync dialogue manager with provided conversation history."""
        # Clear current context and rebuild from history
        self.context = ConversationContext()
        
        for i in range(0, len(conversation_history), 2):
            if i + 1 < len(conversation_history):
                role1, msg1 = conversation_history[i]
                role2, msg2 = conversation_history[i + 1]
                
                if role1 != "Jarvis" and role2 == "Jarvis":
                    self.context.add_exchange(msg1, msg2)
    
    def _enhance_input_with_context(self, user_input: str, context_info: Dict) -> str:
        """Enhance user input with contextual information."""
        if not context_info or not context_info.get('recent_exchanges'):
            return user_input
        
        # Get the most recent exchange
        recent_exchange = context_info['recent_exchanges'][-1]
        
        # Handle pronoun resolution
        enhanced_input = self._resolve_pronouns(user_input, recent_exchange)
        
        # Add contextual prefix if needed
        if len(enhanced_input.split()) <= 5:  # Short input likely needs context
            last_topic = context_info.get('current_topics', [])
            if last_topic:
                enhanced_input = f"Regarding {last_topic[-1]}: {enhanced_input}"
        
        return enhanced_input
    
    def _resolve_pronouns(self, user_input: str, recent_exchange: Dict) -> str:
        """Resolve pronouns based on recent context."""
        # Simple pronoun resolution
        replacements = {
            r'\bit\b': self._find_main_subject(recent_exchange['assistant_response']),
            r'\bthat\b': self._find_main_subject(recent_exchange['assistant_response']),
            r'\bthis\b': self._find_main_subject(recent_exchange['user_input'])
        }
        
        enhanced = user_input
        for pattern, replacement in replacements.items():
            if replacement:
                enhanced = re.sub(pattern, replacement, enhanced, flags=re.I)
        
        return enhanced
    
    def _find_main_subject(self, text: str) -> Optional[str]:
        """Find the main subject of a sentence."""
        # Simple heuristic: find the first noun phrase
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:  # Likely a proper noun
                return word
        return None
    
    def _recommend_skills(self, user_input: str, is_followup: bool, followup_type: str) -> List[str]:
        """Recommend skills based on input analysis."""
        recommended = []
        
        if is_followup:
            if followup_type == 'clarification':
                recommended.extend(['reason', 'ask', 'gemini'])
            elif followup_type == 'elaboration':
                recommended.extend(['reason', 'process', 'ask'])
            elif followup_type == 'example':
                recommended.extend(['ask', 'search', 'gemini'])
            elif followup_type == 'comparison':
                recommended.extend(['reason', 'ask', 'search'])
            elif followup_type == 'continuation':
                recommended.extend([self.context.last_skill_used, 'ask'])
            else:
                recommended.extend(['ask', 'gemini', 'reason'])
        else:
            # Standard skill recommendation based on input content
            input_lower = user_input.lower()
            if any(word in input_lower for word in ['calculate', 'math', 'equation']):
                recommended.append('solve')
            elif any(word in input_lower for word in ['search', 'find', 'look up']):
                recommended.append('search')
            elif any(word in input_lower for word in ['translate', 'language']):
                recommended.append('gtranslate')
            else:
                recommended.extend(['ask', 'gemini', 'reason'])
        
        return recommended[:3]  # Return top 3 recommendations
    
    def _calculate_confidence(self, user_input: str, is_followup: bool, context_info: Dict) -> float:
        """Calculate confidence in the analysis."""
        confidence = 0.5  # Base confidence
        
        if is_followup:
            # Higher confidence if we have good context
            if context_info.get('recent_exchanges'):
                confidence += 0.3
            if context_info.get('relevant_entities'):
                confidence += 0.1
            if context_info.get('current_topics'):
                confidence += 0.1
        else:
            # Higher confidence for clear, specific inputs
            if len(user_input.split()) > 3:
                confidence += 0.2
            if any(char in user_input for char in '?!'):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def add_response(self, user_input: str, response: str, skill_used: str = None):
        """Add a response to the dialogue context."""
        self.context.add_exchange(user_input, response, skill_used)

def enhanced_dialogue_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    Enhanced dialogue processing that provides context-aware responses.
    This skill is called internally by the main system.
    """
    
    # Initialize or get existing dialogue manager
    if not hasattr(enhanced_dialogue_skill, 'dialogue_manager'):
        enhanced_dialogue_skill.dialogue_manager = DialogueStateManager()
    
    dialogue_manager = enhanced_dialogue_skill.dialogue_manager
    
    try:
        # Process the input
        analysis = dialogue_manager.process_input(user_input, conversation_history)
        
        # If it's a follow-up, enhance the processing
        if analysis['is_followup']:
            logger.info(f"Detected follow-up: {analysis['followup_type']} (confidence: {analysis['confidence']:.2f})")
            
            # Use the enhanced input for better processing
            enhanced_input = analysis['processed_input']
            
            # Try recommended skills in order
            for skill_name in analysis['recommended_skills']:
                try:
                    from importlib import import_module
                    main_mod = import_module("main")
                    jarvis_instance = getattr(main_mod, "jarvis", None)
                    
                    if jarvis_instance and skill_name in jarvis_instance.skills:
                        if skill_name == "ask":
                            response = jarvis_instance.skills[skill_name](
                                f"ask {enhanced_input}",
                                conversation_history=conversation_history
                            )
                        else:
                            response = jarvis_instance.skills[skill_name](
                                enhanced_input,
                                conversation_history=conversation_history
                            )
                        
                        if response and not response.lower().startswith(("sorry", "error", "failed")):
                            # Add context information to response
                            context_note = f"\\n\\n*[Follow-up: {analysis['followup_type']}, Context: {len(analysis['context'].get('recent_exchanges', []))} recent exchanges]*"
                            
                            dialogue_manager.add_response(user_input, response, skill_name)
                            return response + context_note
                            
                except Exception as e:
                    logger.warning(f"Skill {skill_name} failed for follow-up: {e}")
                    continue
        
        # If not a follow-up or follow-up processing failed, return analysis info
        return f"Context Analysis: {'Follow-up' if analysis['is_followup'] else 'New topic'} (confidence: {analysis['confidence']:.2f})"
        
    except Exception as e:
        logger.error(f"Error in dialogue processing: {e}")
        return f"I had trouble processing the context of your message: {str(e)}"

def register(jarvis):
    """Register the dialogue management skill."""
    # This skill is primarily used internally, but can be called directly for analysis
    jarvis.register_skill("dialogue", enhanced_dialogue_skill)
    jarvis.register_skill("context", enhanced_dialogue_skill)
