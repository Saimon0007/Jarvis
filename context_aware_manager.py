"""
Context-Aware Manager Module
Provides context-aware learning and knowledge management capabilities
"""

import json
import sqlite3
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

class ContextAwareManager:
    """
    Context-aware learning and retrieval manager that maintains conversation context
    and provides intelligent knowledge retrieval based on user interactions.
    """
    
    def __init__(self, db_path="context_memory.db"):
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self.user_contexts = defaultdict(dict)
        self.entity_memory = defaultdict(dict)
        
        # Initialize embedding model if available
        self.embeddings_model = None
        if HAS_EMBEDDINGS:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Context-aware embeddings model loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load embedding model: {e}")
        
        self._init_database()
        
    def _init_database(self):
        """Initialize the context-aware database"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Context memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Entity memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Contextual interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contextual_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context_used TEXT,
                    success_rating REAL DEFAULT 0.5,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text using simple pattern matching
        Returns list of (entity, type) tuples
        """
        entities = []
        
        # Simple patterns for common entities
        patterns = {
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'location': r'\b(?:in|at|from) ([A-Z][a-zA-Z\s]+)\b',
            'organization': r'\b([A-Z][a-zA-Z\s]+ (?:Inc|Corp|LLC|Ltd|Company))\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
            'technology': r'\b(?:Python|JavaScript|AI|machine learning|neural network|algorithm|database|API|web development)\b',
            'topic': r'\b(?:artificial intelligence|data science|programming|software|technology|science|research)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                entities.append((match.strip(), entity_type))
        
        return entities
    
    def update_context(self, user_id: str, query: str, response: str):
        """
        Update user context based on interaction
        """
        try:
            # Extract entities from query and response
            query_entities = self.extract_entities(query)
            response_entities = self.extract_entities(response)
            
            # Store entities in memory
            for entity, entity_type in query_entities + response_entities:
                self._store_entity(user_id, entity, entity_type)
            
            # Store contextual interaction
            self._store_interaction(user_id, query, response, query_entities + response_entities)
            
            # Update user context in memory
            self.user_contexts[user_id].update({
                'last_query': query,
                'last_response': response,
                'last_entities': query_entities + response_entities,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logging.warning(f"Failed to update context for user {user_id}: {e}")
    
    def _store_entity(self, user_id: str, entity_name: str, entity_type: str):
        """Store or update entity in database"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Check if entity exists
            cursor.execute("""
                SELECT id, properties FROM entity_memory 
                WHERE user_id = ? AND entity_name = ? AND entity_type = ?
            """, (user_id, entity_name, entity_type))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entity
                entity_id, properties_json = existing
                properties = json.loads(properties_json)
                properties['mention_count'] = properties.get('mention_count', 0) + 1
                properties['last_mentioned'] = datetime.now().isoformat()
                
                cursor.execute("""
                    UPDATE entity_memory 
                    SET properties = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (json.dumps(properties), entity_id))
            else:
                # Create new entity
                properties = {
                    'mention_count': 1,
                    'first_mentioned': datetime.now().isoformat(),
                    'last_mentioned': datetime.now().isoformat()
                }
                
                cursor.execute("""
                    INSERT INTO entity_memory (user_id, entity_name, entity_type, properties)
                    VALUES (?, ?, ?, ?)
                """, (user_id, entity_name, entity_type, json.dumps(properties)))
            
            conn.commit()
            conn.close()
    
    def _store_interaction(self, user_id: str, query: str, response: str, entities: List[Tuple[str, str]]):
        """Store contextual interaction"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            context_data = {
                'entities': entities,
                'query_length': len(query),
                'response_length': len(response)
            }
            
            cursor.execute("""
                INSERT INTO contextual_interactions (user_id, query, response, context_used)
                VALUES (?, ?, ?, ?)
            """, (user_id, query, response, json.dumps(context_data)))
            
            conn.commit()
            conn.close()
    
    def get_contextual_knowledge(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve contextually relevant knowledge for the user's query
        """
        try:
            context_items = []
            
            # Get recent interactions
            recent_interactions = self._get_recent_interactions(user_id, limit=10)
            
            # Get relevant entities
            query_entities = self.extract_entities(query)
            relevant_entities = self._get_relevant_entities(user_id, query_entities)
            
            # Combine context information
            for interaction in recent_interactions:
                context_items.append({
                    'type': 'interaction',
                    'content': f"Previous Q: {interaction['query']} A: {interaction['response'][:200]}...",
                    'relevance': 0.8,
                    'timestamp': interaction['timestamp']
                })
            
            for entity in relevant_entities:
                context_items.append({
                    'type': 'entity',
                    'content': f"{entity['entity_type'].title()}: {entity['entity_name']} (mentioned {entity['mention_count']} times)",
                    'relevance': min(1.0, entity['mention_count'] * 0.1),
                    'timestamp': entity['last_mentioned']
                })
            
            # Sort by relevance and recency
            context_items.sort(key=lambda x: (x['relevance'], x['timestamp']), reverse=True)
            
            return context_items[:limit]
            
        except Exception as e:
            logging.warning(f"Failed to get contextual knowledge: {e}")
            return []
    
    def _get_recent_interactions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent interactions for the user"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, response, timestamp FROM contextual_interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'query': row[0],
                    'response': row[1],
                    'timestamp': row[2]
                })
            
            conn.close()
            return results
    
    def _get_relevant_entities(self, user_id: str, query_entities: List[Tuple[str, str]]) -> List[Dict]:
        """Get entities relevant to the query"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Get entities that match query entities or are frequently mentioned
            entity_names = [entity[0] for entity in query_entities]
            entity_types = [entity[1] for entity in query_entities]
            
            if entity_names:
                placeholders = ','.join(['?' for _ in entity_names])
                cursor.execute(f"""
                    SELECT entity_name, entity_type, properties FROM entity_memory
                    WHERE user_id = ? AND (entity_name IN ({placeholders}) OR entity_type IN ({placeholders}))
                    ORDER BY updated_at DESC
                    LIMIT 10
                """, [user_id] + entity_names + entity_types)
            else:
                cursor.execute("""
                    SELECT entity_name, entity_type, properties FROM entity_memory
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 5
                """, (user_id,))
            
            results = []
            for row in cursor.fetchall():
                properties = json.loads(row[2])
                results.append({
                    'entity_name': row[0],
                    'entity_type': row[1],
                    'mention_count': properties.get('mention_count', 1),
                    'last_mentioned': properties.get('last_mentioned', '')
                })
            
            conn.close()
            return results
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Get user preferences based on interaction history
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                # Analyze interaction patterns
                cursor.execute("""
                    SELECT context_used, COUNT(*) as frequency FROM contextual_interactions
                    WHERE user_id = ?
                    GROUP BY context_used
                    ORDER BY frequency DESC
                    LIMIT 10
                """, (user_id,))
                
                context_patterns = cursor.fetchall()
                
                # Get most mentioned entity types
                cursor.execute("""
                    SELECT entity_type, COUNT(*) as frequency FROM entity_memory
                    WHERE user_id = ?
                    GROUP BY entity_type
                    ORDER BY frequency DESC
                    LIMIT 5
                """, (user_id,))
                
                interest_areas = cursor.fetchall()
                
                conn.close()
                
                return {
                    'preferred_contexts': [pattern[0] for pattern in context_patterns if pattern[0]],
                    'interest_areas': [area[0] for area in interest_areas],
                    'interaction_count': len(context_patterns)
                }
                
        except Exception as e:
            logging.warning(f"Failed to get user preferences: {e}")
            return {'preferred_contexts': [], 'interest_areas': [], 'interaction_count': 0}
    
    def cleanup_old_data(self, days_old: int = 30):
        """
        Clean up old context data to prevent database bloat
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                # Clean old interactions
                cursor.execute("""
                    DELETE FROM contextual_interactions
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean unused entities
                cursor.execute("""
                    DELETE FROM entity_memory
                    WHERE updated_at < ? AND json_extract(properties, '$.mention_count') < 2
                """, (cutoff_date.isoformat(),))
                
                deleted_interactions = cursor.rowcount
                conn.commit()
                conn.close()
                
                logging.info(f"Cleaned up {deleted_interactions} old context entries")
                
        except Exception as e:
            logging.warning(f"Failed to cleanup old data: {e}")
