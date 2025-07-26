"""
Advanced Auto-Ingest skill for Jarvis: Intelligent knowledge collection and updates
from multiple sources with scheduled ingestion, topic diversification, and quality filtering.

Usage: 
- auto_ingest(["topic1", "topic2", ...])
- schedule_ingest
- ingest_stats
- add_topic <topic>
- remove_topic <topic>
"""
import os
import json
import requests
import logging
import threading
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
INGEST_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "ingest_config.json")
INGEST_STATS_FILE = os.path.join(os.path.dirname(__file__), "ingest_stats.json")

class IngestSource:
    """Represents a knowledge source for ingestion."""
    
    def __init__(self, name: str, fetch_func: callable, priority: int = 1, quality_threshold: float = 0.5):
        self.name = name
        self.fetch_func = fetch_func
        self.priority = priority
        self.quality_threshold = quality_threshold
        self.success_count = 0
        self.failure_count = 0
        self.last_used = None
    
    def fetch(self, topic: str) -> Optional[Dict[str, Any]]:
        """Fetch content from this source."""
        try:
            content = self.fetch_func(topic)
            if content and self._assess_quality(content):
                self.success_count += 1
                self.last_used = datetime.now().isoformat()
                return {
                    'content': content,
                    'source': self.name,
                    'topic': topic,
                    'timestamp': datetime.now().isoformat(),
                    'quality_score': self._assess_quality(content)
                }
        except Exception as e:
            logger.error(f"Error fetching from {self.name}: {e}")
            self.failure_count += 1
        
        return None
    
    def _assess_quality(self, content: str) -> float:
        """Assess content quality (0-1 score)."""
        if not content or len(content.strip()) < 20:
            return 0.0
        
        # Basic quality indicators
        quality_score = 0.5  # Base score
        
        # Length factor (longer is generally better up to a point)
        length_factor = min(len(content) / 500, 1.0) * 0.2
        quality_score += length_factor
        
        # Sentence structure (presence of periods indicates structured text)
        sentences = content.count('.')
        if sentences > 0:
            quality_score += min(sentences / 10, 0.2)
        
        # Avoid promotional/spam content
        spam_indicators = ['click here', 'buy now', 'limited offer', '!!!', 'FREE!!!']
        spam_penalty = sum(0.1 for indicator in spam_indicators if indicator.lower() in content.lower())
        quality_score -= spam_penalty
        
        return max(0.0, min(1.0, quality_score))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get source statistics."""
        total_attempts = self.success_count + self.failure_count
        success_rate = self.success_count / total_attempts if total_attempts > 0 else 0
        
        return {
            'name': self.name,
            'priority': self.priority,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'last_used': self.last_used
        }

class AdvancedIngestManager:
    """Advanced knowledge ingestion manager with scheduling and quality control."""
    
    def __init__(self):
        self.sources = []
        self.topics = set()
        self.ingestion_history = deque(maxlen=1000)
        self.scheduler_running = False
        self.scheduler_thread = None
        self.config = self.load_config()
        self.stats = self.load_stats()
        
        self._setup_default_sources()
        self._setup_default_topics()
    
    def _setup_default_sources(self):
        """Setup default ingestion sources."""
        # Wikipedia source
        def fetch_wikipedia(topic):
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract")
                if extract and len(extract) > 50:
                    return extract
            return None
        
        # News source
        def fetch_news(topic):
            if not NEWS_API_KEY:
                return None
            url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(topic)}&apiKey={NEWS_API_KEY}&pageSize=3&sortBy=publishedAt"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                if articles:
                    # Combine top articles
                    combined = []
                    for article in articles[:2]:  # Top 2 articles
                        title = article.get("title", "")
                        description = article.get("description", "")
                        if title and description:
                            combined.append(f"{title}: {description}")
                    return " | ".join(combined) if combined else None
            return None
        
        # DuckDuckGo source
        def fetch_duckduckgo(topic):
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(topic)}&format=json&no_redirect=1&no_html=1"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("AbstractText"):
                    return data["AbstractText"]
                elif data.get("RelatedTopics"):
                    topics_data = data["RelatedTopics"][:3]  # Top 3
                    combined = []
                    for topic_data in topics_data:
                        if isinstance(topic_data, dict) and topic_data.get("Text"):
                            combined.append(topic_data["Text"])
                    return " | ".join(combined) if combined else None
            return None
        
        # Web search source (if available)
        def fetch_web_search(topic):
            try:
                from importlib import import_module
                main_mod = import_module("main")
                jarvis_instance = getattr(main_mod, "jarvis", None)
                
                if jarvis_instance and "search" in jarvis_instance.skills:
                    result = jarvis_instance.skills["search"](f"search {topic}")
                    if result and not result.lower().startswith(("sorry", "search failed")):
                        return result
            except Exception:
                pass
            return None
        
        # Register sources with priorities
        self.sources = [
            IngestSource("Wikipedia", fetch_wikipedia, priority=3, quality_threshold=0.7),
            IngestSource("News", fetch_news, priority=2, quality_threshold=0.6),
            IngestSource("DuckDuckGo", fetch_duckduckgo, priority=2, quality_threshold=0.5),
            IngestSource("WebSearch", fetch_web_search, priority=1, quality_threshold=0.4)
        ]
        
        # Sort by priority
        self.sources.sort(key=lambda x: x.priority, reverse=True)
    
    def _setup_default_topics(self):
        """Setup default topics for ingestion."""
        default_topics = {
            # Technology
            "artificial intelligence", "machine learning", "quantum computing", "blockchain",
            "cybersecurity", "cloud computing", "5G technology", "autonomous vehicles",
            "virtual reality", "augmented reality", "robotics", "internet of things",
            
            # Science
            "climate change", "renewable energy", "space exploration", "genetics",
            "neuroscience", "physics discoveries", "medical breakthroughs", "archaeology",
            
            # Current Events
            "world news", "economic trends", "political developments", "social issues",
            "environmental conservation", "global health", "education trends",
            
            # Culture & Society
            "cultural trends", "philosophy", "psychology", "sociology", "literature",
            "art movements", "music trends", "film industry"
        }
        
        # Load additional topics from config
        if 'topics' in self.config:
            default_topics.update(self.config['topics'])
        
        self.topics = default_topics
    
    def load_config(self) -> Dict[str, Any]:
        """Load ingestion configuration."""
        if os.path.exists(INGEST_CONFIG_FILE):
            try:
                with open(INGEST_CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading ingest config: {e}")
        
        return {
            'auto_ingest_enabled': True,
            'ingest_interval_hours': 6,
            'topics_per_session': 10,
            'max_sources_per_topic': 2,
            'min_quality_score': 0.5
        }
    
    def save_config(self):
        """Save ingestion configuration."""
        try:
            config_to_save = self.config.copy()
            config_to_save['topics'] = list(self.topics)
            
            with open(INGEST_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving ingest config: {e}")
    
    def load_stats(self) -> Dict[str, Any]:
        """Load ingestion statistics."""
        if os.path.exists(INGEST_STATS_FILE):
            try:
                with open(INGEST_STATS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_ingestions': 0,
            'successful_ingestions': 0,
            'last_ingest_time': None,
            'topics_ingested': 0,
            'sources_used': defaultdict(int)
        }
    
    def save_stats(self):
        """Save ingestion statistics."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            stats_to_save = dict(self.stats)
            if 'sources_used' in stats_to_save and isinstance(stats_to_save['sources_used'], defaultdict):
                stats_to_save['sources_used'] = dict(stats_to_save['sources_used'])
            
            with open(INGEST_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving ingest stats: {e}")
    
    def add_topic(self, topic: str) -> str:
        """Add a topic to the ingestion list."""
        topic = topic.strip().lower()
        if topic in self.topics:
            return f"Topic '{topic}' is already in the ingestion list."
        
        self.topics.add(topic)
        self.save_config()
        
        return f"✅ Added '{topic}' to ingestion topics. Total topics: {len(self.topics)}"
    
    def remove_topic(self, topic: str) -> str:
        """Remove a topic from the ingestion list."""
        topic = topic.strip().lower()
        if topic not in self.topics:
            return f"Topic '{topic}' is not in the ingestion list."
        
        self.topics.remove(topic)
        self.save_config()
        
        return f"✅ Removed '{topic}' from ingestion topics. Total topics: {len(self.topics)}"
    
    def ingest_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Ingest knowledge for a specific topic."""
        self.stats['total_ingestions'] += 1
        
        best_result = None
        best_quality = 0
        sources_tried = 0
        max_sources = self.config.get('max_sources_per_topic', 2)
        
        for source in self.sources:
            if sources_tried >= max_sources:
                break
            
            result = source.fetch(topic)
            if result and result['quality_score'] > best_quality:
                best_result = result
                best_quality = result['quality_score']
            
            sources_tried += 1
            
            # If we got a high-quality result, we can stop early
            if best_quality > 0.8:
                break
        
        if best_result and best_quality >= self.config.get('min_quality_score', 0.5):
            # Store in knowledge base
            try:
                from importlib import import_module
                main_mod = import_module("main")
                jarvis_instance = getattr(main_mod, "jarvis", None)
                
                if jarvis_instance and hasattr(jarvis_instance, 'enhanced_kb'):
                    jarvis_instance.enhanced_kb.learn(
                        topic,
                        best_result['content'],
                        f"auto_ingest_{best_result['source'].lower()}"
                    )
                elif jarvis_instance and hasattr(jarvis_instance, 'search_engine'):
                    jarvis_instance.search_engine.add_to_knowledge_base(
                        best_result['content'],
                        f"auto_ingest_{best_result['source'].lower()}",
                        {'topic': topic, 'quality_score': best_quality}
                    )
            except Exception as e:
                logger.error(f"Error storing ingested knowledge: {e}")
            
            # Record success
            self.stats['successful_ingestions'] += 1
            self.stats['sources_used'][best_result['source']] += 1
            self.ingestion_history.append(best_result)
            
            return best_result
        
        return None
    
    def run_ingestion_session(self, topic_count: Optional[int] = None) -> Dict[str, Any]:
        """Run a complete ingestion session."""
        if topic_count is None:
            topic_count = self.config.get('topics_per_session', 10)
        
        # Select random topics to keep things diverse
        topics_to_ingest = random.sample(list(self.topics), min(topic_count, len(self.topics)))
        
        session_results = {
            'session_start': datetime.now().isoformat(),
            'topics_attempted': topics_to_ingest,
            'successful_ingestions': [],
            'failed_topics': [],
            'total_topics': len(topics_to_ingest)
        }
        
        logger.info(f"Starting ingestion session with {len(topics_to_ingest)} topics")
        
        for topic in topics_to_ingest:
            try:
                result = self.ingest_topic(topic)
                if result:
                    session_results['successful_ingestions'].append({
                        'topic': topic,
                        'source': result['source'],
                        'quality_score': result['quality_score']
                    })
                    
                    # Small delay to be respectful to APIs
                    time.sleep(1)
                else:
                    session_results['failed_topics'].append(topic)
            except Exception as e:
                logger.error(f"Error ingesting topic '{topic}': {e}")
                session_results['failed_topics'].append(topic)
        
        # Update stats
        self.stats['last_ingest_time'] = datetime.now().isoformat()
        self.stats['topics_ingested'] += len(session_results['successful_ingestions'])
        self.save_stats()
        
        session_results['session_end'] = datetime.now().isoformat()
        
        logger.info(f"Ingestion session completed: {len(session_results['successful_ingestions'])}/{len(topics_to_ingest)} successful")
        
        return session_results
    
    def start_scheduler(self) -> str:
        """Start the automatic ingestion scheduler."""
        if self.scheduler_running:
            return "Scheduler is already running."
        
        def scheduler_loop():
            while self.scheduler_running:
                try:
                    # Run ingestion session
                    results = self.run_ingestion_session()
                    
                    # Wait for next session
                    sleep_hours = self.config.get('ingest_interval_hours', 6)
                    sleep_seconds = sleep_hours * 3600
                    
                    # Sleep in small chunks to allow for clean shutdown
                    for _ in range(int(sleep_seconds / 60)):  # Check every minute
                        if not self.scheduler_running:
                            break
                        time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        return f"✅ Auto-ingestion scheduler started. Will run every {self.config.get('ingest_interval_hours', 6)} hours."
    
    def stop_scheduler(self) -> str:
        """Stop the automatic ingestion scheduler."""
        if not self.scheduler_running:
            return "Scheduler is not running."
        
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        return "✅ Auto-ingestion scheduler stopped."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics."""
        stats = self.stats.copy()
        
        # Add source statistics
        source_stats = [source.get_stats() for source in self.sources]
        stats['source_performance'] = source_stats
        
        # Add configuration info
        stats['configuration'] = self.config.copy()
        stats['active_topics_count'] = len(self.topics)
        stats['scheduler_running'] = self.scheduler_running
        
        # Add recent ingestion history
        stats['recent_ingestions'] = list(self.ingestion_history)[-10:]
        
        return stats

# Global ingest manager
ingest_manager = AdvancedIngestManager()

def enhanced_auto_ingest_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    Enhanced auto-ingest skill with comprehensive management capabilities.
    
    Usage:
    - auto_ingest [topic1, topic2, ...] or auto_ingest <single_topic>
    - schedule_ingest
    - stop_ingest
    - ingest_stats
    - add_topic <topic>
    - remove_topic <topic>
    - ingest_now [count]
    """
    
    user_input = user_input.strip()
    
    # Handle different commands
    if user_input.lower() == 'schedule_ingest':
        return ingest_manager.start_scheduler()
    
    elif user_input.lower() == 'stop_ingest':
        return ingest_manager.stop_scheduler()
    
    elif user_input.lower() in ['ingest_stats', 'ingest statistics']:
        stats = ingest_manager.get_statistics()
        
        response_parts = []
        response_parts.append("📊 **Auto-Ingestion Statistics:**")
        response_parts.append(f"• Total ingestion attempts: {stats['total_ingestions']}")
        response_parts.append(f"• Successful ingestions: {stats['successful_ingestions']}")
        response_parts.append(f"• Success rate: {(stats['successful_ingestions']/max(stats['total_ingestions'], 1)*100):.1f}%")
        response_parts.append(f"• Topics ingested: {stats['topics_ingested']}")
        response_parts.append(f"• Active topics: {stats['active_topics_count']}")
        response_parts.append(f"• Scheduler running: {'✅' if stats['scheduler_running'] else '❌'}")
        
        if stats.get('last_ingest_time'):
            response_parts.append(f"• Last ingestion: {stats['last_ingest_time'][:19]}")
        
        response_parts.append("")
        
        # Source performance
        if stats.get('source_performance'):
            response_parts.append("**Source Performance:**")
            for source in stats['source_performance']:
                rate = f"{source['success_rate']:.1%}" if source['success_rate'] > 0 else "N/A"
                response_parts.append(f"• {source['name']}: {rate} success ({source['success_count']} successful)")
            response_parts.append("")
        
        # Recent ingestions
        if stats.get('recent_ingestions'):
            response_parts.append("**Recent Ingestions:**")
            for ingestion in stats['recent_ingestions'][-5:]:
                topic = ingestion.get('topic', 'Unknown')
                source = ingestion.get('source', 'Unknown')
                quality = ingestion.get('quality_score', 0)
                response_parts.append(f"• {topic} from {source} (quality: {quality:.2f})")
        
        return "\n".join(response_parts)
    
    elif user_input.lower().startswith('add_topic '):
        topic = user_input[10:].strip()
        if not topic:
            return "Please specify a topic to add."
        return ingest_manager.add_topic(topic)
    
    elif user_input.lower().startswith('remove_topic '):
        topic = user_input[13:].strip()
        if not topic:
            return "Please specify a topic to remove."
        return ingest_manager.remove_topic(topic)
    
    elif user_input.lower().startswith('ingest_now'):
        # Extract count if provided
        parts = user_input.split()
        count = None
        if len(parts) > 1:
            try:
                count = int(parts[1])
            except ValueError:
                return "Invalid count. Usage: ingest_now [count]"
        
        results = ingest_manager.run_ingestion_session(count)
        
        response_parts = []
        response_parts.append("🔄 **Ingestion Session Results:**")
        response_parts.append(f"• Topics attempted: {results['total_topics']}")
        response_parts.append(f"• Successful ingestions: {len(results['successful_ingestions'])}")
        response_parts.append(f"• Failed topics: {len(results['failed_topics'])}")
        
        if results['successful_ingestions']:
            response_parts.append("")
            response_parts.append("**Successful Ingestions:**")
            for ingestion in results['successful_ingestions']:
                response_parts.append(f"• {ingestion['topic']} from {ingestion['source']} (quality: {ingestion['quality_score']:.2f})")
        
        if results['failed_topics']:
            response_parts.append("")
            response_parts.append(f"**Failed Topics:** {', '.join(results['failed_topics'])}")
        
        return "\n".join(response_parts)
    
    else:
        # Handle traditional auto_ingest with topics
        # Parse topics from input
        topics = []
        
        # Check if input looks like a list
        if user_input.startswith('[') and user_input.endswith(']'):
            # Parse as list
            try:
                import ast
                topics = ast.literal_eval(user_input)
            except Exception:
                # Fallback: split by comma
                content = user_input[1:-1]  # Remove brackets
                topics = [t.strip().strip('"').strip("'") for t in content.split(',')]
        elif ',' in user_input:
            # Split by comma
            topics = [t.strip() for t in user_input.split(',')]
        else:
            # Single topic
            if user_input.lower().startswith('auto_ingest '):
                topics = [user_input[12:].strip()]
            else:
                topics = [user_input]
        
        if not topics or not any(t.strip() for t in topics):
            return "Please provide topics to ingest. Usage: auto_ingest <topic> or auto_ingest [topic1, topic2, ...]"
        
        # Filter out empty topics
        topics = [t.strip() for t in topics if t.strip()]
        
        # Run ingestion for specified topics
        results = []
        for topic in topics:
            result = ingest_manager.ingest_topic(topic)
            if result:
                results.append(f"✅ {topic}: ingested from {result['source']} (quality: {result['quality_score']:.2f})")
            else:
                results.append(f"❌ {topic}: failed to find quality content")
        
        if results:
            return f"**Ingestion Results:**\n" + "\n".join(results)
        else:
            return "No topics could be successfully ingested."

def multi_llm_search_autoingest(user_input, conversation_history=None, skills=None):
    """
    Query all available LLMs (ask, gemini, llm_plugin, etc.), search, and auto_ingest in parallel.
    Returns a combined, deduplicated, and well-formatted answer.
    """
    if skills is None:
        return "No skills provided."
    
    responses = []
    
    # Query all LLMs
    for llm_name in ["ask", "gemini", "llm_plugin"]:
        if llm_name in skills:
            try:
                resp = skills[llm_name](user_input, conversation_history=conversation_history, search_skill=skills.get("search"))
                if resp and resp not in responses:
                    responses.append(f"**[{llm_name.upper()}]** {resp}")
            except Exception as e:
                responses.append(f"**[{llm_name.upper()}]** Error: {e}")
    
    # Query search
    if "search" in skills:
        try:
            search_resp = skills["search"](f"search {user_input}")
            if search_resp and search_resp not in responses:
                responses.append(f"**[SEARCH]** {search_resp}")
        except Exception as e:
            responses.append(f"**[SEARCH]** Error: {e}")
    
    # Auto-ingest the topic for future reference
    try:
        ingest_result = ingest_manager.ingest_topic(user_input)
        if ingest_result:
            responses.append(f"**[AUTO-LEARNED]** Added '{user_input}' to knowledge base from {ingest_result['source']}")
    except Exception as e:
        logger.error(f"Auto-ingest error: {e}")
    
    # Combine and deduplicate
    if responses:
        combined = "\n\n".join(responses)
        return f"🔍 **Multi-Source Response:**\n\n{combined}"
    else:
        return "No response from any LLM, search, or auto-ingest."

# Legacy function for backward compatibility
def auto_ingest(topics):
    """Legacy auto-ingest function for backward compatibility."""
    if isinstance(topics, str):
        topics = [topics]
    
    results = []
    for topic in topics:
        result = ingest_manager.ingest_topic(topic)
        if result:
            results.append(topic)
    
    return f"Auto-ingested knowledge for {len(results)} topics: {', '.join(results)}"

def register(jarvis):
    """Register the enhanced auto-ingest skill."""
    jarvis.register_skill("auto_ingest", enhanced_auto_ingest_skill)
    jarvis.register_skill("ingest", enhanced_auto_ingest_skill)
    
    # Register the meta-skill for combined LLM/search/ingest
    jarvis.register_skill("multi_llm_search_autoingest", multi_llm_search_autoingest)
    
    # Make ingest manager available globally
    jarvis.ingest_manager = ingest_manager
    
    # Start scheduler if enabled in config
    if ingest_manager.config.get('auto_ingest_enabled', True):
        try:
            ingest_manager.start_scheduler()
            logger.info("Auto-ingestion scheduler started automatically")
        except Exception as e:
            logger.error(f"Failed to start auto-ingestion scheduler: {e}")
