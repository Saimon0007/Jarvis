"""
Advanced Text Processing skill for Jarvis: Handles long text questions, documents,
and complex text analysis with intelligent chunking and summarization.
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """Intelligently chunks long text while preserving semantic meaning."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences while respecting max size."""
        sentences = self.sentence_endings.split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    if self.overlap > 0 and len(current_chunk) > self.overlap:
                        current_chunk = current_chunk[-self.overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    chunks.extend(self._split_long_sentence(sentence))
                    current_chunk = ""
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraphs while respecting max size."""
        paragraphs = self.paragraph_breaks.split(text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    chunks.extend(self.chunk_by_sentences(paragraph))
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that's too long into smaller parts."""
        # Split by commas, semicolons, or other natural breaks
        parts = re.split(r'[,;:\-]+', sentence)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += ", " + part if current_chunk else part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def smart_chunk(self, text: str) -> List[str]:
        """Intelligently choose the best chunking strategy."""
        # First try paragraph-based chunking
        if '\n\n' in text:
            chunks = self.chunk_by_paragraphs(text)
        else:
            chunks = self.chunk_by_sentences(text)
        
        # Filter out chunks that are too small to be meaningful
        meaningful_chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]
        
        return meaningful_chunks if meaningful_chunks else chunks

class TextSummarizer:
    """Provides various text summarization strategies."""
    
    def __init__(self):
        self.key_phrase_patterns = re.compile(r'\b(important|key|main|primary|essential|crucial|significant|major)\b', re.I)
        
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """Extract key sentences based on various heuristics."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Score sentences based on various factors
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position bias (first and last sentences are often important)
            if i == 0 or i == len(sentences) - 1:
                score += 2
            
            # Length bias (medium-length sentences are often more informative)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1
            
            # Key phrase presence
            if self.key_phrase_patterns.search(sentence):
                score += 2
            
            # Question or exclamation (often important)
            if sentence.strip().endswith(('?', '!')):
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sent for score, sent in scored_sentences[:num_sentences]]
    
    def create_abstract_summary(self, text: str, max_length: int = 200) -> str:
        """Create an abstract summary of the text."""
        key_sentences = self.extract_key_sentences(text, 5)
        summary = " ".join(key_sentences)
        
        if len(summary) <= max_length:
            return summary
        
        # Truncate at sentence boundary
        words = summary.split()
        truncated = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                break
            truncated.append(word)
            current_length += len(word) + 1
        
        return " ".join(truncated) + "..."

class LongTextProcessor:
    """Main processor for handling long text questions and analysis."""
    
    def __init__(self):
        self.chunker = TextChunker()
        self.summarizer = TextSummarizer()
        self.processed_cache = {}
    
    def identify_text_type(self, text: str) -> str:
        """Identify the type of text for appropriate processing."""
        text_lower = text.lower()
        
        if any(marker in text_lower for marker in ['abstract:', 'introduction:', 'conclusion:']):
            return 'academic_paper'
        elif text.count('\n') > 10 and any(marker in text_lower for marker in ['chapter', 'section']):
            return 'document'
        elif '```' in text or 'def ' in text or 'class ' in text:
            return 'code'
        elif text.count('?') > 2:
            return 'qa_text'
        elif len(text.split()) > 500:
            return 'long_article'
        else:
            return 'general_text'
    
    def process_long_text(self, text: str, question: str = None) -> Dict[str, Any]:
        """Process long text with appropriate strategy based on content type."""
        # Create a hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.processed_cache:
            logger.info("Using cached analysis")
            cached_result = self.processed_cache[text_hash]
        else:
            # Analyze text
            text_type = self.identify_text_type(text)
            chunks = self.chunker.smart_chunk(text)
            summary = self.summarizer.create_abstract_summary(text)
            
            cached_result = {
                'type': text_type,
                'chunks': chunks,
                'summary': summary,
                'word_count': len(text.split()),
                'chunk_count': len(chunks),
                'processed_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.processed_cache[text_hash] = cached_result
        
        # If there's a specific question, focus analysis on relevant chunks
        if question:
            relevant_chunks = self._find_relevant_chunks(cached_result['chunks'], question)
            cached_result['relevant_chunks'] = relevant_chunks
        
        return cached_result
    
    def _find_relevant_chunks(self, chunks: List[str], question: str) -> List[Tuple[int, str, float]]:
        """Find chunks most relevant to the question."""
        question_words = set(question.lower().split())
        relevant_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            # Simple relevance score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            relevance_score = overlap / max(len(question_words), 1)
            
            if relevance_score > 0.1:  # Threshold for relevance
                relevant_chunks.append((i, chunk, relevance_score))
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x[2], reverse=True)
        return relevant_chunks[:5]  # Return top 5 relevant chunks

def process_long_text_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    Process long text questions and provide comprehensive analysis.
    
    Usage:
    - process <long text>
    - analyze text <long text>
    - summarize <long text>
    - question: <question> text: <long text>
    """
    
    processor = LongTextProcessor()
    
    try:
        # Parse input to extract text and optional question
        text = ""
        question = ""
        
        # Handle different input formats
        if user_input.lower().startswith(("process ", "analyze text ", "summarize ")):
            command_parts = user_input.split(" ", 1)
            if len(command_parts) > 1:
                text = command_parts[1].strip()
        elif "question:" in user_input.lower() and "text:" in user_input.lower():
            # Format: question: <question> text: <text>
            parts = user_input.lower().split("text:", 1)
            if len(parts) == 2:
                question_part = parts[0].replace("question:", "").strip()
                text = parts[1].strip()
                question = question_part
        else:
            # Assume the entire input is text to be processed
            text = user_input.strip()
        
        if not text:
            return "Please provide text to process. Usage: process <text> or question: <question> text: <text>"
        
        # Check if text is long enough to warrant processing
        if len(text.split()) < 50:
            return "Text appears to be short enough for standard processing. For long text analysis, please provide text with at least 50 words."
        
        # Process the text
        analysis = processor.process_long_text(text, question)
        
        # Build response
        response_parts = []
        response_parts.append("📄 **Long Text Analysis Results:**")
        response_parts.append(f"• Text Type: {analysis['type'].replace('_', ' ').title()}")
        response_parts.append(f"• Word Count: {analysis['word_count']:,}")
        response_parts.append(f"• Chunks Created: {analysis['chunk_count']}")
        response_parts.append("")
        
        # Add summary
        response_parts.append("📝 **Summary:**")
        response_parts.append(analysis['summary'])
        response_parts.append("")
        
        # If there was a question, show relevant chunks
        if question and 'relevant_chunks' in analysis:
            response_parts.append(f"🎯 **Relevant Sections for: '{question}'**")
            for i, (chunk_idx, chunk, score) in enumerate(analysis['relevant_chunks'][:3], 1):
                response_parts.append(f"**Section {i} (Relevance: {score:.2f}):**")
                response_parts.append(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                response_parts.append("")
            
            # Try to get an answer using LLM with relevant context
            if analysis['relevant_chunks']:
                context = "\n\n".join([chunk for _, chunk, _ in analysis['relevant_chunks'][:3]])
                llm_response = get_llm_answer_with_context(question, context)
                if llm_response:
                    response_parts.append("🤖 **AI Analysis:**")
                    response_parts.append(llm_response)
        
        # Processing strategy recommendations
        response_parts.append("💡 **Processing Recommendations:**")
        if analysis['type'] == 'academic_paper':
            response_parts.append("• Focus on abstract, introduction, and conclusion sections")
            response_parts.append("• Look for methodology and results sections for key findings")
        elif analysis['type'] == 'long_article':
            response_parts.append("• Break into thematic sections for better comprehension")
            response_parts.append("• Identify main arguments and supporting evidence")
        elif analysis['type'] == 'code':
            response_parts.append("• Analyze functions and classes separately")
            response_parts.append("• Focus on documentation and comments for understanding")
        else:
            response_parts.append("• Process in logical chunks for better understanding")
            response_parts.append("• Identify key themes and main points")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error processing long text: {e}")
        return f"I encountered an error while processing the text: {str(e)}"

def get_llm_answer_with_context(question: str, context: str) -> Optional[str]:
    """Get LLM response with provided context."""
    try:
        from importlib import import_module
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        
        if not jarvis_instance:
            return None
        
        # Create a prompt with context
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the provided context."
        
        # Try different LLM skills
        llm_skills = ["gemini", "ask", "llm_plugin"]
        
        for skill_name in llm_skills:
            if skill_name in jarvis_instance.skills:
                try:
                    if skill_name == "ask":
                        response = jarvis_instance.skills[skill_name](f"ask {prompt}")
                    else:
                        response = jarvis_instance.skills[skill_name](prompt)
                    
                    if response and not response.lower().startswith(("sorry", "error", "failed")):
                        return response
                except Exception as e:
                    logger.warning(f"LLM skill {skill_name} failed: {e}")
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting LLM response with context: {e}")
        return None

def register(jarvis):
    """Register the long text processing skill."""
    jarvis.register_skill("process", process_long_text_skill)
    jarvis.register_skill("analyze_text", process_long_text_skill)
    jarvis.register_skill("summarize_long", process_long_text_skill)
