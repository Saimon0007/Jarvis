"""
Context-Aware Learning and Retrieval Manager
Manages personalized knowledge with context embeddings and user interaction history
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import sqlite3
import hashlib
from collections import defaultdict
import logging

class ContextAwareManager:
    def __init__(self, db_path="jarvis_context.db"):
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        self.init_database()
        
        # Context tracking
        self.current_context = {}
        self.conversation_history = []
        self.user_preferences = {}
        self.interaction_patterns = defaultdict(int)
        
    def init_database(self):
        """Initialize the context database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Context embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                context_type TEXT NOT NULL,
                user_id TEXT DEFAULT 'default',
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, preference_key)
            )
        ''')
        
        # Interaction patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interaction_patterns (
                user_id TEXT NOT NULL,
                query_pattern TEXT NOT NULL,
                response_pattern TEXT NOT NULL,
                success_rate REAL DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, query_pattern)
            )
        ''')
        
        # Context relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES context_embeddings (id),
                FOREIGN KEY (target_id) REFERENCES context_embeddings (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def update_context(self, query: str, response: str, user_id: str = "default", 
                      context_metadata: Dict = None):
        """Update conversation context with new interaction"""
        interaction = {
            'query': query,
            'response': response,
            'timestamp': datetime.now(),
            'user_id': user_id,
            'metadata': context_metadata or {}
        }
        
        self.conversation_history.append(interaction)
        
        # Keep only recent history (last 100 interactions)
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
        
        # Update current context
        self.current_context = {
            'recent_queries': [item['query'] for item in self.conversation_history[-5:]],
            'recent_topics': self._extract_topics(self.conversation_history[-10:]),
            'user_focus': self._determine_user_focus(),
            'session_context': self._build_session_context()
        }
        
        # Store context embedding
        self._store_context_embedding(query, response, user_id, context_metadata)
        
        # Update interaction patterns
        self._update_interaction_patterns(query, response, user_id)
    
    def get_contextual_knowledge(self, query: str, user_id: str = "default", 
                                limit: int = 10) -> List[Dict]:
        """Retrieve knowledge with context awareness"""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Get user preferences
        user_prefs = self._get_user_preferences(user_id)
        
        # Get relevant context embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, content, embedding, context_type, relevance_score, 
                   access_count, last_accessed
            FROM context_embeddings 
            WHERE user_id = ? OR user_id = 'global'
            ORDER BY last_accessed DESC
        ''', (user_id,))
        
        results = []
        for row in cursor.fetchall():
            stored_embedding = np.frombuffer(row[2], dtype=np.float32)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            # Apply context-aware scoring
            context_score = self._calculate_context_score(
                row[0], query, user_prefs, similarity
            )
            
            results.append({
                'id': row[0],
                'content': row[1],
                'context_type': row[3],
                'relevance_score': row[4],
                'similarity': similarity,
                'context_score': context_score,
                'access_count': row[5],
                'last_accessed': row[6]
            })
        
        conn.close()
        
        # Sort by context score and return top results
        results.sort(key=lambda x: x['context_score'], reverse=True)
        return results[:limit]
    
    def learn_user_preference(self, user_id: str, preference_key: str, 
                            preference_value: str, weight: float = 1.0):
        """Learn and store user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_id, preference_key, preference_value, weight, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, preference_key, preference_value, weight))
        
        conn.commit()
        conn.close()
        
        # Update in-memory preferences
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][preference_key] = {
            'value': preference_value,
            'weight': weight
        }
    
    def get_personalized_response(self, query: str, base_response: str, 
                                user_id: str = "default") -> str:
        """Personalize response based on user context and preferences"""
        user_prefs = self._get_user_preferences(user_id)
        
        # Get recent context
        recent_context = self._get_recent_context(user_id)
        
        # Apply personalization
        personalized_response = self._apply_personalization(
            base_response, user_prefs, recent_context, query
        )
        
        return personalized_response
    
    def _store_context_embedding(self, query: str, response: str, user_id: str, 
                               metadata: Dict):
        """Store context embedding in database"""
        # Combine query and response for context
        context_text = f"Query: {query}\nResponse: {response}"
        embedding = self.model.encode([context_text])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO context_embeddings 
            (content, embedding, context_type, user_id, relevance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (context_text, embedding.tobytes(), 'conversation', user_id, 1.0))
        
        conn.commit()
        conn.close()
    
    def _update_interaction_patterns(self, query: str, response: str, user_id: str):
        """Update interaction patterns for learning"""
        query_pattern = self._extract_query_pattern(query)
        response_pattern = self._extract_response_pattern(response)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO interaction_patterns
            (user_id, query_pattern, response_pattern, frequency, last_used)
            VALUES (
                ?, ?, ?, 
                COALESCE((SELECT frequency FROM interaction_patterns 
                         WHERE user_id = ? AND query_pattern = ?), 0) + 1,
                CURRENT_TIMESTAMP
            )
        ''', (user_id, query_pattern, response_pattern, user_id, query_pattern))
        
        conn.commit()
        conn.close()
    
    def _extract_topics(self, interactions: List[Dict]) -> List[str]:
        """Extract topics from recent interactions"""
        # Simple topic extraction (can be enhanced with NLP)
        topics = []
        for interaction in interactions:
            words = interaction['query'].lower().split()
            # Extract potential topics (nouns, technical terms)
            for word in words:
                if len(word) > 4 and word.isalpha():
                    topics.append(word)
        return list(set(topics))
    
    def _determine_user_focus(self) -> str:
        """Determine current user focus from conversation history"""
        if not self.conversation_history:
            return "general"
        
        recent_queries = [item['query'] for item in self.conversation_history[-5:]]
        
        # Simple focus determination (can be enhanced)
        focus_keywords = {
            'technical': ['code', 'programming', 'debug', 'error', 'function'],
            'research': ['research', 'learn', 'understand', 'explain', 'why'],
            'creative': ['create', 'write', 'design', 'generate', 'make'],
            'problem_solving': ['solve', 'fix', 'help', 'issue', 'problem']
        }
        
        focus_scores = defaultdict(int)
        for query in recent_queries:
            query_lower = query.lower()
            for focus, keywords in focus_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        focus_scores[focus] += 1
        
        return max(focus_scores.items(), key=lambda x: x[1])[0] if focus_scores else "general"
    
    def _build_session_context(self) -> Dict:
        """Build current session context"""
        if not self.conversation_history:
            return {}
        
        return {
            'session_length': len(self.conversation_history),
            'session_start': self.conversation_history[0]['timestamp'],
            'last_interaction': self.conversation_history[-1]['timestamp'],
            'dominant_topics': self._extract_topics(self.conversation_history[-10:])[:5],
            'interaction_frequency': len(self.conversation_history) / max(1, 
                (datetime.now() - self.conversation_history[0]['timestamp']).total_seconds() / 3600)
        }
    
    def _get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences from database"""
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT preference_key, preference_value, weight
            FROM user_preferences
            WHERE user_id = ?
        ''', (user_id,))
        
        preferences = {}
        for row in cursor.fetchall():
            preferences[row[0]] = {
                'value': row[1],
                'weight': row[2]
            }
        
        conn.close()
        self.user_preferences[user_id] = preferences
        return preferences
    
    def _calculate_context_score(self, embedding_id: int, query: str, 
                               user_prefs: Dict, similarity: float) -> float:
        """Calculate context-aware relevance score"""
        base_score = similarity
        
        # Recency boost
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT last_accessed, access_count, relevance_score
            FROM context_embeddings
            WHERE id = ?
        ''', (embedding_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            last_accessed = datetime.fromisoformat(result[0])
            access_count = result[1]
            relevance_score = result[2]
            
            # Recency boost (more recent = higher score)
            hours_since_access = (datetime.now() - last_accessed).total_seconds() / 3600
            recency_boost = max(0, 1 - (hours_since_access / 168))  # Decay over a week
            
            # Frequency boost
            frequency_boost = min(1.0, access_count / 10)
            
            # User preference alignment
            preference_boost = self._calculate_preference_alignment(query, user_prefs)
            
            # Combined context score
            context_score = (
                base_score * 0.4 +
                recency_boost * 0.2 +
                frequency_boost * 0.1 +
                preference_boost * 0.2 +
                relevance_score * 0.1
            )
            
            return context_score
        
        return base_score
    
    def _calculate_preference_alignment(self, query: str, user_prefs: Dict) -> float:
        """Calculate how well query aligns with user preferences"""
        if not user_prefs:
            return 0.5
        
        alignment_score = 0.0
        total_weight = 0.0
        
        query_lower = query.lower()
        
        for pref_key, pref_data in user_prefs.items():
            weight = pref_data['weight']
            value = pref_data['value'].lower()
            
            # Simple keyword matching (can be enhanced)
            if value in query_lower or any(word in query_lower for word in value.split()):
                alignment_score += weight
            
            total_weight += weight
        
        return alignment_score / max(1.0, total_weight)
    
    def _get_recent_context(self, user_id: str, hours: int = 24) -> Dict:
        """Get recent context for personalization"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_interactions = [
            interaction for interaction in self.conversation_history
            if interaction['timestamp'] > cutoff_time and 
               interaction['user_id'] == user_id
        ]
        
        return {
            'recent_queries': [item['query'] for item in recent_interactions],
            'recent_topics': self._extract_topics(recent_interactions),
            'interaction_count': len(recent_interactions)
        }
    
    def _apply_personalization(self, base_response: str, user_prefs: Dict, 
                             recent_context: Dict, query: str) -> str:
        """Apply personalization to base response"""
        # Start with base response
        personalized_response = base_response
        
        # Add context-aware enhancements
        if recent_context.get('recent_topics'):
            common_topics = set(recent_context['recent_topics']) & set(query.lower().split())
            if common_topics:
                personalized_response += f"\n\n*Building on our recent discussion about {', '.join(common_topics)}*"
        
        # Add preference-based customizations
        response_style = user_prefs.get('response_style', {}).get('value', 'balanced')
        if response_style == 'detailed':
            personalized_response += "\n\n*Providing detailed explanation based on your preference*"
        elif response_style == 'concise':
            personalized_response += "\n\n*Keeping response concise as preferred*"
        
        return personalized_response
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract pattern from query for learning"""
        # Simple pattern extraction (can be enhanced with NLP)
        words = query.lower().split()
        if len(words) == 0:
            return "empty"
        
        # Identify question types
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in words for word in question_words):
            return f"question_{words[0] if words[0] in question_words else 'general'}"
        
        # Identify commands
        command_words = ['create', 'make', 'build', 'generate', 'write', 'find', 'search']
        if any(word in words for word in command_words):
            return f"command_{next(word for word in words if word in command_words)}"
        
        return "statement"
    
    def _extract_response_pattern(self, response: str) -> str:
        """Extract pattern from response for learning"""
        # Simple response pattern classification
        if len(response) < 50:
            return "short"
        elif len(response) < 200:
            return "medium"
        elif "```" in response:
            return "code_example"
        elif response.count('\n') > 5:
            return "structured"
        else:
            return "long_explanation"

# Skill interface for Jarvis
def get_skill_info():
    return {
        "name": "context_aware_manager",
        "description": "Manages context-aware learning and retrieval with personalized knowledge management",
        "version": "1.0.0",
        "author": "Jarvis Enhanced",
        "commands": {
            "update_context": "Update conversation context with new interaction",
            "get_contextual_knowledge": "Retrieve knowledge with context awareness",
            "learn_preference": "Learn and store user preferences",
            "get_personalized_response": "Get personalized response based on user context"
        }
    }

def process_request(jarvis_instance, command, args):
    """Process context-aware manager requests"""
    try:
        if not hasattr(jarvis_instance, 'context_manager'):
            jarvis_instance.context_manager = ContextAwareManager()
        
        manager = jarvis_instance.context_manager
        
        if command == "update_context":
            query = args.get('query', '')
            response = args.get('response', '')
            user_id = args.get('user_id', 'default')
            metadata = args.get('metadata', {})
            
            manager.update_context(query, response, user_id, metadata)
            return "Context updated successfully"
        
        elif command == "get_contextual_knowledge":
            query = args.get('query', '')
            user_id = args.get('user_id', 'default')
            limit = args.get('limit', 10)
            
            results = manager.get_contextual_knowledge(query, user_id, limit)
            return {
                "status": "success",
                "results": results,
                "count": len(results)
            }
        
        elif command == "learn_preference":
            user_id = args.get('user_id', 'default')
            pref_key = args.get('preference_key', '')
            pref_value = args.get('preference_value', '')
            weight = args.get('weight', 1.0)
            
            manager.learn_user_preference(user_id, pref_key, pref_value, weight)
            return f"User preference learned: {pref_key} = {pref_value}"
        
        elif command == "get_personalized_response":
            query = args.get('query', '')
            base_response = args.get('base_response', '')
            user_id = args.get('user_id', 'default')
            
            personalized = manager.get_personalized_response(query, base_response, user_id)
            return personalized
        
        else:
            return f"Unknown command: {command}"
            
    except Exception as e:
        return f"Error in context-aware manager: {str(e)}"
