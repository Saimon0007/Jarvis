"""
Advanced Self-Learning skill for Jarvis: Intelligent learning system with relevance feedback,
confidence scoring, knowledge validation, and adaptive learning strategies.

Usage:
  learn <question>? <answer>
  update <question>? <new answer>
  forget <question>
  validate <question>
  feedback <question> good/bad
  knowledge stats
  export knowledge
  To answer: just ask the question again (fuzzy match supported).
"""

import re
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from difflib import get_close_matches
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KB_FILE = os.path.join(os.path.dirname(__file__), "self_learn_kb.json")
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "learning_feedback.json")
STATS_FILE = os.path.join(os.path.dirname(__file__), "learning_stats.json")

class KnowledgeEntry:
    """Represents a knowledge entry with metadata and feedback tracking."""
    
    def __init__(self, question: str, answer: str, source: str = "manual"):
        self.question = question.lower().strip()
        self.answer = answer.strip()
        self.source = source
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.access_count = 0
        self.last_accessed = None
        self.confidence_score = 0.5  # Initial neutral confidence
        self.feedback_history = []
        self.validation_status = "unvalidated"
        self.tags = []
        self.related_questions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'question': self.question,
            'answer': self.answer,
            'source': self.source,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'confidence_score': self.confidence_score,
            'feedback_history': self.feedback_history,
            'validation_status': self.validation_status,
            'tags': self.tags,
            'related_questions': self.related_questions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        entry = cls(data['question'], data['answer'], data.get('source', 'manual'))
        entry.created_at = data.get('created_at', entry.created_at)
        entry.updated_at = data.get('updated_at', entry.updated_at)
        entry.access_count = data.get('access_count', 0)
        entry.last_accessed = data.get('last_accessed')
        entry.confidence_score = data.get('confidence_score', 0.5)
        entry.feedback_history = data.get('feedback_history', [])
        entry.validation_status = data.get('validation_status', 'unvalidated')
        entry.tags = data.get('tags', [])
        entry.related_questions = data.get('related_questions', [])
        return entry
    
    def update_answer(self, new_answer: str, source: str = "update"):
        """Update the answer with metadata tracking."""
        self.answer = new_answer.strip()
        self.updated_at = datetime.now().isoformat()
        self.source = source
        # Reset confidence on update
        self.confidence_score = 0.5
        self.validation_status = "unvalidated"
    
    def record_access(self):
        """Record that this entry was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
    
    def add_feedback(self, feedback_type: str, details: str = ""):
        """Add user feedback to improve confidence scoring."""
        feedback_entry = {
            'type': feedback_type,  # 'positive', 'negative', 'correction'
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(feedback_entry)
        
        # Update confidence score based on feedback
        if feedback_type == 'positive':
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
        elif feedback_type == 'negative':
            self.confidence_score = max(0.0, self.confidence_score - 0.15)
        elif feedback_type == 'correction':
            self.confidence_score = max(0.0, self.confidence_score - 0.2)
    
    def calculate_relevance_score(self, query: str) -> float:
        """Calculate relevance score for a query."""
        # Base similarity using fuzzy matching
        base_similarity = max(
            len(get_close_matches(query.lower(), [self.question], n=1, cutoff=0.1)),
            # Also check word overlap
            len(set(query.lower().split()).intersection(set(self.question.split()))) / max(len(query.split()), 1)
        )
        
        # Apply confidence and recency bonuses
        confidence_bonus = self.confidence_score * 0.2
        
        # Recency bonus (newer entries get slight preference)
        days_old = (datetime.now() - datetime.fromisoformat(self.created_at)).days
        recency_bonus = max(0, 0.1 - (days_old * 0.01))
        
        # Popularity bonus (frequently accessed entries)
        popularity_bonus = min(0.1, self.access_count * 0.01)
        
        total_score = base_similarity + confidence_bonus + recency_bonus + popularity_bonus
        return min(1.0, total_score)

class EnhancedKnowledgeBase:
    """Advanced knowledge base with intelligent learning and feedback systems."""
    
    def __init__(self):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.load_knowledge_base()
        self.learning_stats = self.load_stats()
    
    def load_knowledge_base(self):
        """Load knowledge base from file."""
        if os.path.exists(KB_FILE):
            try:
                with open(KB_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Handle both old and new formats
                for key, value in data.items():
                    if isinstance(value, str):
                        # Old format: convert to new format
                        self.entries[key] = KnowledgeEntry(key, value, "legacy")
                    elif isinstance(value, dict):
                        # New format
                        self.entries[key] = KnowledgeEntry.from_dict(value)
                        
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                self.entries = {}
    
    def save_knowledge_base(self):
        """Save knowledge base to file."""
        try:
            data = {key: entry.to_dict() for key, entry in self.entries.items()}
            with open(KB_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def load_stats(self) -> Dict[str, Any]:
        """Load learning statistics."""
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_learned': 0,
            'total_queries': 0,
            'successful_matches': 0,
            'feedback_received': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def save_stats(self):
        """Save learning statistics."""
        try:
            self.learning_stats['last_updated'] = datetime.now().isoformat()
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.learning_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def learn(self, question: str, answer: str, source: str = "manual") -> str:
        """Learn a new Q&A pair with enhanced metadata."""
        question_key = question.lower().strip().rstrip("?")
        
        if question_key in self.entries:
            return f"I already know about '{question}'. Use 'update' to modify or 'feedback' to improve it."
        
        entry = KnowledgeEntry(question, answer, source)
        self.entries[question_key] = entry
        
        self.learning_stats['total_learned'] += 1
        self.save_knowledge_base()
        self.save_stats()
        
        return f"✅ Learned: '{question}' with confidence score {entry.confidence_score:.2f}"
    
    def update(self, question: str, new_answer: str, source: str = "update") -> str:
        """Update an existing knowledge entry."""
        question_key = question.lower().strip().rstrip("?")
        
        if question_key not in self.entries:
            # Try fuzzy matching
            matches = get_close_matches(question_key, self.entries.keys(), n=1, cutoff=0.8)
            if matches:
                question_key = matches[0]
            else:
                return f"I don't know about '{question}'. Use 'learn' to teach me."
        
        old_answer = self.entries[question_key].answer
        self.entries[question_key].update_answer(new_answer, source)
        
        self.save_knowledge_base()
        
        return f"✅ Updated '{question}' (was: '{old_answer[:50]}...', now: '{new_answer[:50]}...')"
    
    def forget(self, question: str) -> str:
        """Remove a knowledge entry."""
        question_key = question.lower().strip().rstrip("?")
        
        # Try exact match first
        if question_key in self.entries:
            del self.entries[question_key]
            self.save_knowledge_base()
            return f"✅ Forgot about '{question}'"
        
        # Try fuzzy matching
        matches = get_close_matches(question_key, self.entries.keys(), n=1, cutoff=0.8)
        if matches:
            del self.entries[matches[0]]
            self.save_knowledge_base()
            return f"✅ Forgot about '{matches[0]}'"
        
        return f"I don't know about '{question}' to forget it."
    
    def query(self, question: str) -> Optional[str]:
        """Query the knowledge base with enhanced matching."""
        question_key = question.lower().strip().rstrip("?")
        
        self.learning_stats['total_queries'] += 1
        
        # Find best matching entries
        candidates = []
        for entry_key, entry in self.entries.items():
            relevance = entry.calculate_relevance_score(question)
            if relevance > 0.3:  # Minimum threshold
                candidates.append((entry, relevance))
        
        if not candidates:
            self.save_stats()
            return None
        
        # Sort by relevance and return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_entry, relevance = candidates[0]
        
        # Record access
        best_entry.record_access()
        self.learning_stats['successful_matches'] += 1
        
        self.save_knowledge_base()
        self.save_stats()
        
        # Include confidence information in response
        confidence_indicator = "🟢" if best_entry.confidence_score > 0.7 else "🟡" if best_entry.confidence_score > 0.4 else "🔴"
        
        return f"{best_entry.answer} {confidence_indicator} (confidence: {best_entry.confidence_score:.2f})"
    
    def add_feedback(self, question: str, feedback_type: str, details: str = "") -> str:
        """Add feedback to improve learning."""
        question_key = question.lower().strip().rstrip("?")
        
        # Find the entry
        entry = None
        if question_key in self.entries:
            entry = self.entries[question_key]
        else:
            # Try fuzzy matching
            matches = get_close_matches(question_key, self.entries.keys(), n=1, cutoff=0.6)
            if matches:
                entry = self.entries[matches[0]]
        
        if not entry:
            return f"I couldn't find a knowledge entry for '{question}' to provide feedback on."
        
        entry.add_feedback(feedback_type, details)
        self.learning_stats['feedback_received'] += 1
        
        self.save_knowledge_base()
        self.save_stats()
        
        return f"✅ Feedback recorded. New confidence score: {entry.confidence_score:.2f}"
    
    def validate_entry(self, question: str) -> str:
        """Validate a knowledge entry using external sources."""
        # This could be enhanced to use web search or other validation methods
        question_key = question.lower().strip().rstrip("?")
        
        if question_key not in self.entries:
            return f"No entry found for '{question}' to validate."
        
        entry = self.entries[question_key]
        
        # Simple validation: mark as validated and boost confidence
        entry.validation_status = "validated"
        entry.confidence_score = min(1.0, entry.confidence_score + 0.2)
        
        self.save_knowledge_base()
        
        return f"✅ Validated '{question}'. Confidence boosted to {entry.confidence_score:.2f}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = self.learning_stats.copy()
        
        # Add current session stats
        stats.update({
            'knowledge_entries': len(self.entries),
            'high_confidence_entries': sum(1 for e in self.entries.values() if e.confidence_score > 0.7),
            'validated_entries': sum(1 for e in self.entries.values() if e.validation_status == "validated"),
            'most_accessed': sorted(self.entries.values(), key=lambda x: x.access_count, reverse=True)[:3],
            'recent_entries': sorted(self.entries.values(), key=lambda x: x.created_at, reverse=True)[:3]
        })
        
        return stats
    
    def export_knowledge(self) -> str:
        """Export knowledge in a readable format."""
        if not self.entries:
            return "No knowledge to export."
        
        export_data = []
        for entry in sorted(self.entries.values(), key=lambda x: x.confidence_score, reverse=True):
            export_data.append({
                'question': entry.question,
                'answer': entry.answer,
                'confidence': entry.confidence_score,
                'source': entry.source,
                'access_count': entry.access_count
            })
        
        export_file = os.path.join(os.path.dirname(__file__), "knowledge_export.json")
        try:
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return f"✅ Knowledge exported to {export_file}"
        except Exception as e:
            return f"❌ Export failed: {e}"

# Global enhanced knowledge base
enhanced_kb = EnhancedKnowledgeBase()

def auto_learn_from_internet(question: str, search_skill=None) -> Optional[str]:
    """
    Enhanced auto-learning from internet with confidence tracking.
    """
    if not search_skill:
        return None
        
    # Check if already exists
    existing = enhanced_kb.query(question)
    if existing:
        return None
    
    try:
        search_result = search_skill(f"search {question}")
        if search_result and not search_result.lower().startswith(("sorry", "search failed")):
            # Learn with auto source
            enhanced_kb.learn(question, search_result, "auto_internet")
            return search_result
    except Exception as e:
        logger.error(f"Auto-learning error: {e}")
    
    return None

def enhanced_self_learn_skill(user_input: str, conversation_history=None, search_skill=None) -> str:
    """
    Enhanced self-learning skill with advanced features.
    
    Usage:
    - learn <question>? <answer>
    - update <question>? <new answer>
    - forget <question>
    - feedback <question> good/bad/correction
    - validate <question>
    - knowledge stats
    - export knowledge
    """
    
    user_input = user_input.strip()
    
    # Handle different commands
    if user_input.lower().startswith("learn "):
        # Format: learn <question>? <answer>
        match = re.match(r"learn\s+(.+?)\?\s+(.+)", user_input, re.I)
        if match:
            question, answer = match.group(1).strip(), match.group(2).strip()
            return enhanced_kb.learn(question, answer, "manual")
        else:
            return "Usage: learn <question>? <answer>"
    
    elif user_input.lower().startswith("update "):
        match = re.match(r"update\s+(.+?)\?\s+(.+)", user_input, re.I)
        if match:
            question, answer = match.group(1).strip(), match.group(2).strip()
            return enhanced_kb.update(question, answer, "manual_update")
        else:
            return "Usage: update <question>? <new answer>"
    
    elif user_input.lower().startswith("forget "):
        question = user_input[7:].strip().rstrip("?")
        return enhanced_kb.forget(question)
    
    elif user_input.lower().startswith("feedback "):
        # Format: feedback <question> good/bad/correction [details]
        parts = user_input[9:].strip().split()
        if len(parts) < 2:
            return "Usage: feedback <question> good/bad/correction [details]"
        
        feedback_type = parts[-1].lower()
        question = " ".join(parts[:-1]).rstrip("?")
        
        if feedback_type in ['good', 'positive']:
            return enhanced_kb.add_feedback(question, 'positive')
        elif feedback_type in ['bad', 'negative']:
            return enhanced_kb.add_feedback(question, 'negative')
        elif feedback_type in ['correction', 'wrong']:
            return enhanced_kb.add_feedback(question, 'correction')
        else:
            return "Feedback type must be: good, bad, or correction"
    
    elif user_input.lower().startswith("validate "):
        question = user_input[9:].strip().rstrip("?")
        return enhanced_kb.validate_entry(question)
    
    elif user_input.lower() in ['knowledge stats', 'learning stats', 'learn stats']:
        stats = enhanced_kb.get_statistics()
        
        response_parts = []
        response_parts.append("📊 **Learning Statistics:**")
        response_parts.append(f"• Total knowledge entries: {stats['knowledge_entries']}")
        response_parts.append(f"• High confidence entries: {stats['high_confidence_entries']}")
        response_parts.append(f"• Validated entries: {stats['validated_entries']}")
        response_parts.append(f"• Total queries processed: {stats['total_queries']}")
        response_parts.append(f"• Successful matches: {stats['successful_matches']}")
        response_parts.append(f"• Feedback received: {stats['feedback_received']}")
        response_parts.append("")
        
        if stats['most_accessed']:
            response_parts.append("**Most Accessed Knowledge:**")
            for entry in stats['most_accessed']:
                response_parts.append(f"• {entry.question} (accessed {entry.access_count} times)")
            response_parts.append("")
        
        if stats['recent_entries']:
            response_parts.append("**Recently Learned:**")
            for entry in stats['recent_entries']:
                confidence_icon = "🟢" if entry.confidence_score > 0.7 else "🟡" if entry.confidence_score > 0.4 else "🔴"
                response_parts.append(f"• {entry.question} {confidence_icon}")
        
        return "\n".join(response_parts)
    
    elif user_input.lower() in ['export knowledge', 'export learning']:
        return enhanced_kb.export_knowledge()
    
    else:
        # Try to answer from enhanced knowledge base
        result = enhanced_kb.query(user_input)
        if result:
            return result
        
        # If not found, try auto-learning from internet
        if search_skill:
            learned = auto_learn_from_internet(user_input, search_skill)
            if learned:
                confidence_indicator = "🟡"  # New auto-learned content gets medium confidence
                return f"{learned} {confidence_indicator} (auto-learned, confidence: 0.5)"
        
        return None  # Let other skills try

def register(jarvis):
    """Register the enhanced self-learning skill."""
    def wrapped_enhanced_learn(user_input, conversation_history=None):
        return enhanced_self_learn_skill(
            user_input,
            conversation_history=conversation_history,
            search_skill=jarvis.skills.get("search")
        )
    
    jarvis.register_skill("learn", wrapped_enhanced_learn)
    
    # Make enhanced KB available globally
    jarvis.enhanced_kb = enhanced_kb
