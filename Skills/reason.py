"""
Advanced Reasoning skill for Jarvis: Provides comprehensive step-by-step explanations
for complex questions using multi-layered analysis and reasoning chains.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplexQuestionProcessor:
    """Processes complex questions by breaking them into manageable components."""
    
    def __init__(self):
        self.question_patterns = {
            'comparison': r'\b(compare|contrast|difference|vs|versus|better|worse)\b',
            'causal': r'\b(why|how|because|cause|reason|result|effect|due to)\b',
            'analytical': r'\b(analyze|analysis|examine|evaluate|assess|critique)\b',
            'definitional': r'\b(what is|define|definition|meaning|explain)\b',
            'procedural': r'\b(how to|steps|process|method|procedure|way to)\b',
            'conditional': r'\b(if|when|suppose|assume|given that|provided)\b',
            'temporal': r'\b(when|before|after|during|timeline|history|future)\b',
            'quantitative': r'\b(how much|how many|calculate|count|measure|amount)\b'
        }
        
        self.complexity_indicators = [
            'multiple', 'various', 'several', 'both', 'either', 'neither',
            'however', 'although', 'despite', 'nevertheless', 'furthermore',
            'moreover', 'additionally', 'consequently', 'therefore', 'hence'
        ]
    
    def analyze_question_type(self, question: str) -> List[str]:
        """Identify the types of reasoning required for a question."""
        question_lower = question.lower()
        identified_types = []
        
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, question_lower):
                identified_types.append(q_type)
        
        return identified_types
    
    def assess_complexity(self, question: str) -> int:
        """Assess question complexity on a scale of 1-10."""
        complexity_score = 1
        question_lower = question.lower()
        
        # Length factor
        complexity_score += min(len(question.split()) // 10, 3)
        
        # Complexity indicators
        for indicator in self.complexity_indicators:
            if indicator in question_lower:
                complexity_score += 1
        
        # Multiple question marks or sub-questions
        complexity_score += question.count('?') - 1
        complexity_score += question.count(';')
        complexity_score += question.count(',')
        
        # Nested clauses
        complexity_score += question.count('(')
        complexity_score += question.count('[')
        
        return min(complexity_score, 10)
    
    def decompose_question(self, question: str) -> List[str]:
        """Break down complex questions into simpler sub-questions."""
        sub_questions = []
        
        # Split by conjunctions and punctuation
        splits = re.split(r'[;,]|\band\b|\bor\b|\bbut\b|\bhowever\b', question)
        
        for split in splits:
            split = split.strip()
            if split and len(split.split()) > 3:
                sub_questions.append(split)
        
        # If no meaningful splits, return original
        if not sub_questions:
            sub_questions = [question]
        
        return sub_questions

class ReasoningChain:
    """Manages reasoning chains for complex problem solving."""
    
    def __init__(self):
        self.reasoning_templates = {
            'comparison': [
                "What are the key characteristics of each item being compared?",
                "What are the similarities between them?",
                "What are the differences between them?",
                "Which criteria should be used for comparison?",
                "What is the conclusion based on the comparison?"
            ],
            'causal': [
                "What is the main phenomenon or event in question?",
                "What are the potential causes or contributing factors?",
                "What evidence supports each potential cause?",
                "How do these causes interact with each other?",
                "What is the most likely explanation?"
            ],
            'analytical': [
                "What is the subject being analyzed?",
                "What are its main components or aspects?",
                "How do these components relate to each other?",
                "What patterns or trends can be identified?",
                "What conclusions can be drawn from this analysis?"
            ],
            'procedural': [
                "What is the desired outcome or goal?",
                "What resources or prerequisites are needed?",
                "What are the main steps in the process?",
                "What potential challenges or obstacles exist?",
                "How can success be measured?"
            ]
        }
    
    def generate_reasoning_steps(self, question_types: List[str], question: str) -> List[str]:
        """Generate reasoning steps based on question types."""
        if not question_types:
            return ["Let me analyze this question step by step:", question]
        
        # Use the most specific template available
        primary_type = question_types[0] if question_types else 'analytical'
        template = self.reasoning_templates.get(primary_type, self.reasoning_templates['analytical'])
        
        reasoning_steps = [f"Let me approach this {primary_type} question systematically:"]
        reasoning_steps.extend([f"Step {i+1}: {step}" for i, step in enumerate(template)])
        
        return reasoning_steps

def enhanced_reasoning_skill(user_input, conversation_history=None, search_skill=None, **kwargs):
    """
    Enhanced reasoning skill that provides comprehensive step-by-step explanations
    for complex questions using advanced analysis and reasoning chains.
    
    Usage: reason <question or problem>
    """
    # Extract question
    if user_input.lower().startswith("reason "):
        question = user_input[7:].strip()
    else:
        question = user_input.strip()
    
    if not question:
        return "Please provide a question or problem to reason about."
    
    # Initialize processors
    question_processor = ComplexQuestionProcessor()
    reasoning_chain = ReasoningChain()
    
    try:
        # Analyze the question
        question_types = question_processor.analyze_question_type(question)
        complexity = question_processor.assess_complexity(question)
        sub_questions = question_processor.decompose_question(question)
        
        logger.info(f"Question types: {question_types}, Complexity: {complexity}")
        
        # Generate reasoning framework
        reasoning_steps = reasoning_chain.generate_reasoning_steps(question_types, question)
        
        # Build comprehensive response
        response_parts = []
        response_parts.append(f"📊 **Question Analysis:**")
        response_parts.append(f"• Complexity Level: {complexity}/10")
        response_parts.append(f"• Question Types: {', '.join(question_types) if question_types else 'General inquiry'}")
        response_parts.append(f"• Sub-components: {len(sub_questions)}")
        response_parts.append("")
        
        # Add reasoning framework
        response_parts.append("🧠 **Reasoning Framework:**")
        for step in reasoning_steps:
            response_parts.append(f"• {step}")
        response_parts.append("")
        
        # Process each sub-question if multiple exist
        if len(sub_questions) > 1:
            response_parts.append("🔍 **Detailed Analysis:**")
            for i, sub_q in enumerate(sub_questions, 1):
                response_parts.append(f"**Sub-question {i}:** {sub_q}")
                
                # Try to get answer for each sub-question
                sub_answer = get_llm_response(sub_q, conversation_history, search_skill)
                if sub_answer:
                    response_parts.append(f"**Answer {i}:** {sub_answer}")
                response_parts.append("")
        
        # Get comprehensive answer for the main question
        main_answer = get_llm_response(question, conversation_history, search_skill)
        if main_answer:
            response_parts.append("💡 **Comprehensive Answer:**")
            response_parts.append(main_answer)
            response_parts.append("")
        
        # Add synthesis if multiple sub-questions
        if len(sub_questions) > 1:
            synthesis_prompt = f"Synthesize the following sub-answers into a coherent response for: {question}"
            synthesis = get_llm_response(synthesis_prompt, conversation_history, search_skill)
            if synthesis:
                response_parts.append("🔗 **Synthesis:**")
                response_parts.append(synthesis)
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error in enhanced reasoning: {e}")
        return f"I encountered an issue while analyzing your question. Let me try a simpler approach: {get_llm_response(question, conversation_history, search_skill)}"

def get_llm_response(question: str, conversation_history=None, search_skill=None) -> Optional[str]:
    """Get response from available LLM services."""
    try:
        from importlib import import_module
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        
        if not jarvis_instance:
            return None
        
        # Try different LLM skills in order of preference
        llm_skills = ["gemini", "ask", "llm_plugin"]
        
        for skill_name in llm_skills:
            if skill_name in jarvis_instance.skills:
                try:
                    if skill_name == "ask":
                        response = jarvis_instance.skills[skill_name](
                            f"ask {question}", 
                            conversation_history=conversation_history, 
                            search_skill=search_skill
                        )
                    else:
                        response = jarvis_instance.skills[skill_name](
                            question, 
                            conversation_history=conversation_history, 
                            search_skill=search_skill
                        )
                    
                    if response and not response.lower().startswith(("sorry", "error", "failed")):
                        return response
                except Exception as e:
                    logger.warning(f"LLM skill {skill_name} failed: {e}")
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return None

def register(jarvis):
    """Register the enhanced reasoning skill."""
    jarvis.register_skill("reason", enhanced_reasoning_skill)
    # Also register aliases
    jarvis.register_skill("analyze", enhanced_reasoning_skill)
    jarvis.register_skill("think", enhanced_reasoning_skill)
