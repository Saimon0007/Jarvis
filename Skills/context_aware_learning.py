"""
Context-Aware Learning and Retrieval skill for Jarvis: Personalized knowledge management
with conversation context integration, user preference learning, and semantic contextual matching.

Features:
- Context-aware knowledge storage and retrieval
- User preference learning and adaptation
- Conversation context integration
- Personalized relevance scoring
- Context embeddings for semantic matching
- User interaction patterns analysis
- Dynamic context weighting
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import hashlib
import pickle
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONTEXT_DATA_FILE = os.path.join(os.path.dirname(__file__), "context_data.json")
EMBEDDINGS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "context_embeddings.pkl")
USER_PROFILES_FILE = os.path.join(os.path.dirname(__file__), "user_profiles.json")

@dataclass
class ContextualKnowledge:
    """Represents knowledge with contextual information."""
    content: str
    topic: str
    source: str
    timestamp: str
    context_embedding: Optional[np.ndarray] = None
    conversation_context: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[str] = None
    user_feedback: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        if self.context_embedding is not None:
            data['context_embedding'] = self.context_embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextualKnowledge':
        """Create instance from dictionary."""
        # Convert list back to numpy array
        if data.get('context_embedding'):
            data['context_embedding'] = np.array(data['context_embedding'])
        return cls(**data)

@dataclass
class UserProfile:
    """Represents user preferences and interaction patterns."""
    user_id: str
    preferred_topics: Dict[str, float] = None  # topic -> preference score
    interaction_patterns: Dict[str, Any] = None
    context_preferences: Dict[str, float] = None  # context type -> preference
    learning_style: str = "balanced"  # balanced, detailed, concise, technical
    feedback_history: List[Dict[str, Any]] = None
    created_at: str = None
    last_updated: str = None
    
    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = defaultdict(float)
        if self.interaction_patterns is None:
            self.interaction_patterns = defaultdict(int)
        if self.context_preferences is None:
            self.context_preferences = defaultdict(float)
        if self.feedback_history is None:
            self.feedback_history = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class ContextAwareLearningManager:
    """Manages context-aware learning and retrieval with user personalization."""
    
    def __init__(self):
        self.knowledge_base: List[ContextualKnowledge] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.conversation_context: deque = deque(maxlen=10)  # Last 10 exchanges
        self.current_user_id: str = "default_user"
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        self.load_data()
    
    def load_data(self):
        """Load all contextual data and user profiles."""
        # Load knowledge base
        if os.path.exists(CONTEXT_DATA_FILE):
            try:
                with open(CONTEXT_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = [ContextualKnowledge.from_dict(item) for item in data]
            except Exception as e:
                logger.error(f"Error loading context data: {e}")
        
        # Load embeddings cache
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            try:
                with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading embeddings cache: {e}")
        
        # Load user profiles
        if os.path.exists(USER_PROFILES_FILE):
            try:
                with open(USER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    for user_id, profile_data in profiles_data.items():
                        self.user_profiles[user_id] = UserProfile(**profile_data)
            except Exception as e:
                logger.error(f"Error loading user profiles: {e}")
    
    def save_data(self):
        """Save all contextual data and user profiles."""
        try:
            # Save knowledge base
            with open(CONTEXT_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump([kb.to_dict() for kb in self.knowledge_base], f, ensure_ascii=False, indent=2)
            
            # Save embeddings cache
            with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            
            # Save user profiles
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = asdict(profile)
            
            with open(USER_PROFILES_FILE, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving context data: {e}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get or compute text embedding with caching."""
        if not self.embedding_model:
            return None
        
        # Create hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            embedding = self.embedding_model.encode(text)
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def update_conversation_context(self, user_input: str, assistant_response: str):
        """Update conversation context history."""
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'assistant_response': assistant_response,
            'user_id': self.current_user_id
        }
        self.conversation_context.append(context_entry)
        
    def update_context(self, query: str, response: str, user_id: str = "default", context_metadata: Dict = None):
        """Update context with new interaction."""
        self.update_conversation_context(query, response)  # Use existing method
        
        if context_metadata:
            # Store additional metadata if provided
            entry = self.conversation_context[-1]  # Get last entry
            entry.update({'metadata': context_metadata})
    
    def get_current_context(self) -> str:
        """Get current conversation context as text."""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for entry in list(self.conversation_context)[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {entry['user_input']}")
            context_parts.append(f"Assistant: {entry['assistant_response']}")
        
        return "\n".join(context_parts)
    
    def get_user_profile(self, user_id: Optional[str] = None) -> UserProfile:
        """Get or create user profile."""
        if user_id is None:
            user_id = self.current_user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.user_profiles[user_id]
    
    def learn_contextual_knowledge(self, 
                                  content: str, 
                                  topic: str, 
                                  source: str, 
                                  user_context: Optional[str] = None) -> str:
        """Learn new knowledge with contextual information."""
        # Get current conversation context
        conversation_context = user_context or self.get_current_context()
        
        # Generate context embedding
        context_text = f"{topic} {content} {conversation_context}"
        context_embedding = self.get_embedding(context_text)
        
        # Get user preferences
        user_profile = self.get_user_profile()
        
        # Create contextual knowledge entry
        knowledge = ContextualKnowledge(
            content=content,
            topic=topic,
            source=source,
            timestamp=datetime.now().isoformat(),
            context_embedding=context_embedding,
            conversation_context=conversation_context,
            user_preferences=asdict(user_profile),
            relevance_score=1.0  # Initial score
        )
        
        self.knowledge_base.append(knowledge)
        
        # Update user profile with new topic preference
        user_profile.preferred_topics[topic] += 0.1
        user_profile.interaction_patterns['knowledge_learned'] += 1
        user_profile.last_updated = datetime.now().isoformat()
        
        self.save_data()
        
        return f"✅ Learned contextual knowledge about '{topic}' from {source} with conversation context"
    
    def calculate_contextual_relevance(self, 
                                     knowledge: ContextualKnowledge, 
                                     query: str, 
                                     current_context: str) -> float:
        """Calculate relevance score based on content, context, and user preferences."""
        total_score = 0.0
        
        # Base content relevance (40%)
        content_similarity = 0.0
        if knowledge.context_embedding is not None:
            query_embedding = self.get_embedding(f"{query} {current_context}")
            if query_embedding is not None:
                content_similarity = np.dot(knowledge.context_embedding, query_embedding) / (
                    np.linalg.norm(knowledge.context_embedding) * np.linalg.norm(query_embedding)
                )
        
        total_score += content_similarity * 0.4
        
        # User preference relevance (25%)
        user_profile = self.get_user_profile()
        topic_preference = user_profile.preferred_topics.get(knowledge.topic, 0.0)
        total_score += min(topic_preference, 1.0) * 0.25
        
        # Recency factor (15%)
        try:
            knowledge_time = datetime.fromisoformat(knowledge.timestamp)
            time_diff = datetime.now() - knowledge_time
            recency_score = max(0, 1 - (time_diff.days / 365))  # Decay over 1 year
            total_score += recency_score * 0.15
        except Exception:
            pass
        
        # Access frequency (10%)
        access_score = min(knowledge.access_count / 10.0, 1.0)
        total_score += access_score * 0.10
        
        # Context similarity (10%)
        context_similarity = 0.0
        if knowledge.conversation_context and current_context:
            context_embedding1 = self.get_embedding(knowledge.conversation_context)
            context_embedding2 = self.get_embedding(current_context)
            if context_embedding1 is not None and context_embedding2 is not None:
                context_similarity = np.dot(context_embedding1, context_embedding2) / (
                    np.linalg.norm(context_embedding1) * np.linalg.norm(context_embedding2)
                )
        
        total_score += context_similarity * 0.10
        
        return min(total_score, 1.0)
    
    def retrieve_contextual_knowledge(self, 
                                    query: str, 
                                    max_results: int = 5, 
                                    min_relevance: float = 0.3) -> List[Tuple[ContextualKnowledge, float]]:
        """Retrieve relevant knowledge based on query and context."""
        current_context = self.get_current_context()
        
        # Calculate relevance scores for all knowledge entries
        scored_knowledge = []
        for knowledge in self.knowledge_base:
            relevance = self.calculate_contextual_relevance(knowledge, query, current_context)
            if relevance >= min_relevance:
                scored_knowledge.append((knowledge, relevance))
                
                # Update access statistics
                knowledge.access_count += 1
                knowledge.last_accessed = datetime.now().isoformat()
        
        # Sort by relevance and return top results
        scored_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        return scored_knowledge[:max_results]
    
    def provide_feedback(self, 
                        knowledge_id: str, 
                        feedback_type: str, 
                        feedback_value: float, 
                        comment: Optional[str] = None):
        """Provide feedback on knowledge relevance and quality."""
        # Find knowledge by content hash (simple ID system)
        knowledge_hash = hashlib.md5(knowledge_id.encode()).hexdigest()
        
        for knowledge in self.knowledge_base:
            content_hash = hashlib.md5(knowledge.content.encode()).hexdigest()
            if content_hash == knowledge_hash:
                if knowledge.user_feedback is None:
                    knowledge.user_feedback = []
                
                feedback_entry = {
                    'type': feedback_type,
                    'value': feedback_value,
                    'comment': comment,
                    'timestamp': datetime.now().isoformat(),
                    'user_id': self.current_user_id
                }
                
                knowledge.user_feedback.append(feedback_entry)
                
                # Update user profile based on feedback
                user_profile = self.get_user_profile()
                if feedback_type == 'relevance' and feedback_value > 0.5:
                    user_profile.preferred_topics[knowledge.topic] += 0.2
                elif feedback_type == 'relevance' and feedback_value < 0.5:
                    user_profile.preferred_topics[knowledge.topic] -= 0.1
                
                user_profile.feedback_history.append(feedback_entry)
                user_profile.last_updated = datetime.now().isoformat()
                
                self.save_data()
                return f"✅ Feedback recorded for knowledge about '{knowledge.topic}'"
        
        return "❌ Knowledge not found for feedback"
    
    def get_personalized_summary(self, query: str) -> str:
        """Get personalized summary based on user preferences and context."""
        relevant_knowledge = self.retrieve_contextual_knowledge(query, max_results=3)
        
        if not relevant_knowledge:
            return f"No contextual knowledge found for '{query}'. Consider learning more about this topic."
        
        user_profile = self.get_user_profile()
        learning_style = user_profile.learning_style
        
        response_parts = []
        response_parts.append(f"🎯 **Personalized Response for '{query}'** (Style: {learning_style})")
        response_parts.append("")
        
        for i, (knowledge, relevance) in enumerate(relevant_knowledge, 1):
            response_parts.append(f"**{i}. {knowledge.topic}** (Relevance: {relevance:.2f})")
            
            # Adapt response based on learning style
            if learning_style == "concise":
                # Provide brief summary
                content = knowledge.content[:200] + "..." if len(knowledge.content) > 200 else knowledge.content
            elif learning_style == "detailed":
                # Provide full content
                content = knowledge.content
            elif learning_style == "technical":
                # Focus on technical aspects
                content = knowledge.content
                response_parts.append(f"Source: {knowledge.source}")
                response_parts.append(f"Last accessed: {knowledge.last_accessed or 'Never'}")
            else:  # balanced
                content = knowledge.content[:400] + "..." if len(knowledge.content) > 400 else knowledge.content
            
            response_parts.append(content)
            response_parts.append("")
        
        # Add context awareness note
        if self.conversation_context:
            response_parts.append("💡 *This response considers your recent conversation context and preferences.*")
        
        return "\n".join(response_parts)
    
    def analyze_user_patterns(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user interaction patterns and preferences."""
        user_profile = self.get_user_profile(user_id)
        
        analysis = {
            'user_id': user_profile.user_id,
            'profile_created': user_profile.created_at,
            'last_updated': user_profile.last_updated,
            'learning_style': user_profile.learning_style,
            'total_interactions': sum(user_profile.interaction_patterns.values()),
            'top_topics': dict(sorted(user_profile.preferred_topics.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]),
            'interaction_summary': dict(user_profile.interaction_patterns),
            'feedback_count': len(user_profile.feedback_history),
            'average_feedback': np.mean([f.get('value', 0) for f in user_profile.feedback_history]) 
                               if user_profile.feedback_history else 0
        }
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about contextual learning."""
        stats = {
            'total_knowledge_entries': len(self.knowledge_base),
            'total_users': len(self.user_profiles),
            'total_embeddings_cached': len(self.embeddings_cache),
            'conversation_context_length': len(self.conversation_context),
            'knowledge_by_source': defaultdict(int),
            'knowledge_by_topic': defaultdict(int),
            'recent_activity': []
        }
        
        # Analyze knowledge base
        for knowledge in self.knowledge_base:
            stats['knowledge_by_source'][knowledge.source] += 1
            stats['knowledge_by_topic'][knowledge.topic] += 1
            
            # Recent activity (last 7 days)
            try:
                knowledge_time = datetime.fromisoformat(knowledge.timestamp)
                if datetime.now() - knowledge_time < timedelta(days=7):
                    stats['recent_activity'].append({
                        'topic': knowledge.topic,
                        'source': knowledge.source,
                        'timestamp': knowledge.timestamp
                    })
            except Exception:
                pass
        
        # Convert defaultdicts to regular dicts
        stats['knowledge_by_source'] = dict(stats['knowledge_by_source'])
        stats['knowledge_by_topic'] = dict(stats['knowledge_by_topic'])
        
        return stats

# Global context manager
context_manager = ContextAwareLearningManager()

def context_aware_learning_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    Context-aware learning and retrieval skill.
    
    Commands:
    - learn_context <topic> <content> - Learn with context
    - retrieve_context <query> - Retrieve with context
    - user_profile [user_id] - Show user profile
    - provide_feedback <knowledge_content> <type> <value> [comment] - Provide feedback
    - set_learning_style <style> - Set learning style (balanced/detailed/concise/technical)
    - context_stats - Show statistics
    """
    
    user_input = user_input.strip()
    
    # Update conversation context if history provided
    if conversation_history:
        context_manager.update_conversation_context(
            user_input, 
            ""  # Response will be updated after generation
        )
    
    if user_input.lower().startswith('learn_context '):
        parts = user_input[14:].split(' ', 1)
        if len(parts) < 2:
            return "Usage: learn_context <topic> <content>"
        
        topic = parts[0]
        content = parts[1]
        return context_manager.learn_contextual_knowledge(content, topic, "manual_input")
    
    elif user_input.lower().startswith('retrieve_context '):
        query = user_input[17:].strip()
        if not query:
            return "Usage: retrieve_context <query>"
        
        return context_manager.get_personalized_summary(query)
    
    elif user_input.lower().startswith('user_profile'):
        parts = user_input.split()
        user_id = parts[1] if len(parts) > 1 else None
        
        analysis = context_manager.analyze_user_patterns(user_id)
        
        response_parts = []
        response_parts.append(f"👤 **User Profile Analysis**")
        response_parts.append(f"• User ID: {analysis['user_id']}")
        response_parts.append(f"• Learning Style: {analysis['learning_style']}")
        response_parts.append(f"• Total Interactions: {analysis['total_interactions']}")
        response_parts.append(f"• Feedback Count: {analysis['feedback_count']}")
        response_parts.append(f"• Average Feedback: {analysis['average_feedback']:.2f}")
        response_parts.append("")
        
        if analysis['top_topics']:
            response_parts.append("**Top Topics:**")
            for topic, score in list(analysis['top_topics'].items())[:5]:
                response_parts.append(f"• {topic}: {score:.2f}")
            response_parts.append("")
        
        if analysis['interaction_summary']:
            response_parts.append("**Interaction Summary:**")
            for action, count in analysis['interaction_summary'].items():
                response_parts.append(f"• {action}: {count}")
        
        return "\n".join(response_parts)
    
    elif user_input.lower().startswith('provide_feedback '):
        parts = user_input[17:].split()
        if len(parts) < 3:
            return "Usage: provide_feedback <knowledge_content> <type> <value> [comment]"
        
        knowledge_id = parts[0]
        feedback_type = parts[1]
        try:
            feedback_value = float(parts[2])
        except ValueError:
            return "Feedback value must be a number (0-1)"
        
        comment = " ".join(parts[3:]) if len(parts) > 3 else None
        
        return context_manager.provide_feedback(knowledge_id, feedback_type, feedback_value, comment)
    
    elif user_input.lower().startswith('set_learning_style '):
        style = user_input[19:].strip().lower()
        valid_styles = ['balanced', 'detailed', 'concise', 'technical']
        
        if style not in valid_styles:
            return f"Invalid learning style. Choose from: {', '.join(valid_styles)}"
        
        user_profile = context_manager.get_user_profile()
        user_profile.learning_style = style
        user_profile.last_updated = datetime.now().isoformat()
        context_manager.save_data()
        
        return f"✅ Learning style set to '{style}'"
    
    elif user_input.lower() == 'context_stats':
        stats = context_manager.get_statistics()
        
        response_parts = []
        response_parts.append("📊 **Context-Aware Learning Statistics**")
        response_parts.append(f"• Total Knowledge Entries: {stats['total_knowledge_entries']}")
        response_parts.append(f"• Total Users: {stats['total_users']}")
        response_parts.append(f"• Embeddings Cached: {stats['total_embeddings_cached']}")
        response_parts.append(f"• Conversation Context: {stats['conversation_context_length']} exchanges")
        response_parts.append("")
        
        if stats['knowledge_by_source']:
            response_parts.append("**Knowledge by Source:**")
            for source, count in sorted(stats['knowledge_by_source'].items(), key=lambda x: x[1], reverse=True)[:5]:
                response_parts.append(f"• {source}: {count}")
            response_parts.append("")
        
        if stats['knowledge_by_topic']:
            response_parts.append("**Top Topics:**")
            for topic, count in sorted(stats['knowledge_by_topic'].items(), key=lambda x: x[1], reverse=True)[:5]:
                response_parts.append(f"• {topic}: {count}")
            response_parts.append("")
        
        if stats['recent_activity']:
            response_parts.append(f"**Recent Activity (Last 7 Days):** {len(stats['recent_activity'])} entries")
        
        return "\n".join(response_parts)
    
    else:
        # Default: try to retrieve contextual knowledge
        return context_manager.get_personalized_summary(user_input)

def register(jarvis):
    """Register the context-aware learning skill."""
    jarvis.register_skill("context_learning", context_aware_learning_skill)
    jarvis.register_skill("contextual_learning", context_aware_learning_skill)
    
    # Make context manager available globally
    jarvis.context_manager = context_manager
    
    logger.info("Context-aware learning skill registered successfully")
