"""
Semantic search capability for finding relevant information in the workspace.
Provides natural language search across multiple data sources.
"""

import logging
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        self.entity_types = {
            'programming': ['python', 'javascript', 'java', 'c++', 'code', 'function', 'class', 'api'],
            'technology': ['software', 'hardware', 'database', 'network', 'server', 'cloud'],
            'datetime': ['today', 'yesterday', 'tomorrow', 'date', 'time', 'year', 'month'],
            'action': ['create', 'update', 'delete', 'find', 'search', 'modify', 'run']
        }

    def expand_query(self, query: str) -> List[str]:
        """Expand the query with synonyms and related terms."""
        expanded = [query]
        words = query.lower().split()
        
        # Add variations with different word orders
        if len(words) > 2:
            expanded.append(' '.join(words[::-1]))  # Reverse order
        
        # Add variations with partial matches
        if len(words) > 3:
            expanded.append(' '.join(words[1:]))  # Skip first word
            expanded.append(' '.join(words[:-1]))  # Skip last word
        
        return list(set(expanded))

    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract known entities and their types from the query."""
        entities = defaultdict(list)
        words = query.lower().split()
        
        for word in words:
            for entity_type, keywords in self.entity_types.items():
                if word in keywords:
                    entities[entity_type].append(word)
        
        return {k: ', '.join(v) for k, v in entities.items() if v}

class SearchEngine:
    def __init__(self):
        self.stats = {
            'total_searches': 0,
            'total_results': 0,
            'source_usage': defaultdict(int),
            'recent_queries': []
        }
        self.knowledge_base = []
        self.max_recent_queries = 10

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search across the knowledge base."""
        self.stats['total_searches'] += 1
        
        if not self.knowledge_base:
            return []
        
        # Basic similarity scoring (placeholder for actual semantic search)
        results = []
        query_words = set(query.lower().split())
        
        for entry in self.knowledge_base:
            text = entry['text'].lower()
            text_words = set(text.split())
            
            # Simple Jaccard similarity
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0:
                results.append({
                    'text': entry['text'],
                    'source': entry['source'],
                    'similarity': similarity
                })
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:top_k]
        
        # Update stats
        self.stats['total_results'] += len(results)
        for result in results:
            self.stats['source_usage'][result['source']] += 1
        
        # Update recent queries
        if query not in self.stats['recent_queries']:
            self.stats['recent_queries'].insert(0, query)
            self.stats['recent_queries'] = self.stats['recent_queries'][:self.max_recent_queries]
        
        return results

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get current search statistics."""
        return {
            'total_searches': self.stats['total_searches'],
            'average_results': (
                self.stats['total_results'] / self.stats['total_searches']
                if self.stats['total_searches'] > 0 else 0.0
            ),
            'source_usage': dict(self.stats['source_usage']),
            'recent_queries': self.stats['recent_queries']
        }

    def add_to_knowledge_base(self, text: str, source: str = "unknown", metadata: Optional[Dict] = None) -> bool:
        """Add new text to the knowledge base."""
        try:
            entry = {
                'text': text,
                'source': source,
                'added_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.knowledge_base.append(entry)
            return True
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {e}")
            return False

# Global search engine instance
search_engine = SearchEngine()

def enhanced_semantic_search_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    Enhanced semantic search skill with multi-source integration.
    
    Usage:
    - semantic search <query>
    - search semantics <query>
    - find similar <query>
    - search stats
    - add knowledge <text>
    """
    # Ensure user_input is a string
    if not isinstance(user_input, str):
        return "Invalid input type. Expected string."
        
    user_input = user_input.strip()
    
    # Handle different commands
    if user_input.lower() in ['search stats', 'search statistics']:
        stats = search_engine.get_search_statistics()
        
        if 'message' in stats:
            return stats['message']
        
        response_parts = []
        response_parts.append("📊 **Semantic Search Statistics:**")
        response_parts.append(f"• Total searches: {stats['total_searches']}")
        response_parts.append(f"• Average results per search: {stats['average_results']:.1f}")
        response_parts.append("")
        
        if stats['source_usage']:
            response_parts.append("**Source Usage:**")
            for source, count in stats['source_usage'].items():
                response_parts.append(f"• {source.replace('_', ' ').title()}: {count}")
            response_parts.append("")
        
        if stats['recent_queries']:
            response_parts.append("**Recent Queries:**")
            for query in stats['recent_queries']:
                response_parts.append(f"• {query}")
        
        return "\n".join(response_parts)
    
    elif user_input.lower().startswith('add knowledge '):
        text_to_add = user_input[13:].strip()
        if not text_to_add:
            return "Please provide text to add to the knowledge base."
        
        success = search_engine.add_to_knowledge_base(text_to_add, source="user_added")
        if success:
            return f"✅ Added to knowledge base: {text_to_add[:100]}{'...' if len(text_to_add) > 100 else ''}"
        else:
            return "❌ Failed to add text to knowledge base."
    
    else:
        # Extract query from different command formats
        query = ""
        if user_input.lower().startswith('semantic search '):
            query = user_input[16:].strip()
        elif user_input.lower().startswith('search semantics '):
            query = user_input[17:].strip()
        elif user_input.lower().startswith('find similar '):
            query = user_input[13:].strip()
        else:
            query = user_input
        
        if not query:
            return "Please provide a search query."
        
        try:
            # Process and expand query
            processor = QueryProcessor()
            expanded_queries = processor.expand_query(query)
            entities = processor.extract_entities(query)
            
            # Perform semantic search
            results = search_engine.semantic_search(query, top_k=5)
            
            if not results:
                return f"No semantic matches found for: '{query}'. Try rephrasing or adding more specific terms."
            
            # Build response
            response_parts = []
            response_parts.append(f"🔍 **Semantic Search Results for: '{query}'**")
            
            if entities:
                response_parts.append(f"**Detected entities:** {', '.join([f'{k}: {v}' for k, v in entities.items()])}")
            
            response_parts.append("")
            
            for i, result in enumerate(results, 1):
                similarity = result['similarity']
                source = result['source'].replace('_', ' ').title()
                text = result['text']
                
                response_parts.append(f"**Result {i}** (Similarity: {similarity:.2f}, Source: {source})")
                
                # Truncate long results
                if len(text) > 300:
                    text = text[:300] + "..."
                
                response_parts.append(text)
                response_parts.append("")
            
            # Add query expansion info if helpful
            if len(expanded_queries) > 1:
                response_parts.append("💡 **Alternative queries tried:**")
                for alt_query in expanded_queries[1:3]:  # Show up to 2 alternatives
                    response_parts.append(f"• {alt_query}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return f"I encountered an error during semantic search: {str(e)}"

def auto_ingest_from_conversation(conversation_history: List[Tuple[str, str]]) -> int:
    """Automatically ingest useful information from conversation history."""
    if not conversation_history:
        return 0
    
    ingested_count = 0
    
    for role, message in conversation_history:
        # Ensure message is a string
        if not isinstance(message, str):
            continue
            
        # Only ingest substantial responses from Jarvis
        if role == "Jarvis" and len(message.split()) > 20:
            # Skip error messages and simple acknowledgments
            if not message.lower().startswith(("sorry", "i don't", "error", "failed")):
                metadata = {
                    'source': 'conversation_history',
                    'role': role,
                    'ingested_at': datetime.now().isoformat()
                }
                
                if search_engine.add_to_knowledge_base(message, "conversation", metadata):
                    ingested_count += 1
    
    return ingested_count

def register(jarvis):
    """Register the semantic search skill."""
    jarvis.register_skill("semantic_search", enhanced_semantic_search_skill)
    jarvis.register_skill("find_similar", enhanced_semantic_search_skill)
    jarvis.register_skill("search_semantics", enhanced_semantic_search_skill)
    
    # Make search engine available globally
    jarvis.search_engine = search_engine
    
    # Auto-ingest existing conversation history if available
    try:
        if hasattr(jarvis, 'conversation_history') and jarvis.conversation_history:
            ingested = auto_ingest_from_conversation(jarvis.conversation_history)
            if ingested > 0:
                logger.info(f"Auto-ingested {ingested} conversation entries into semantic search")
    except Exception as e:
        logger.error(f"Error auto-ingesting conversation history: {e}")
