"""
Enhanced Error Handling and User Feedback system for Jarvis: Provides intelligent
error recovery, detailed diagnostics, and helpful user guidance.
"""

import re
import sys
import traceback
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict, deque
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorClassifier:
    """Classifies different types of errors for appropriate handling."""
    
    def __init__(self):
        self.error_patterns = {
            'api_error': [
                r'api.+?error',
                r'request.+?failed',
                r'connection.+?error',
                r'timeout',
                r'rate.+?limit',
                r'authentication.+?failed',
                r'unauthorized'
            ],
            'skill_error': [
                r'skill.+?not.+?found',
                r'no.+?such.+?skill',
                r'command.+?not.+?recognized',
                r'invalid.+?skill'
            ],
            'input_error': [
                r'invalid.+?input',
                r'missing.+?parameter',
                r'format.+?error',
                r'syntax.+?error',
                r'parsing.+?error'
            ],
            'system_error': [
                r'import.+?error',
                r'module.+?not.+?found',
                r'permission.+?denied',
                r'file.+?not.+?found',
                r'memory.+?error'
            ],
            'llm_error': [
                r'llm.+?request.+?failed',
                r'model.+?error',
                r'openai.+?error',
                r'gemini.+?error',
                r'context.+?too.+?long'
            ]
        }
    
    def classify_error(self, error_message: str, exception_type: str = None) -> str:
        """Classify error based on message content and exception type."""
        error_lower = error_message.lower()
        
        # First check exception type
        if exception_type:
            type_mapping = {
                'ConnectionError': 'api_error',
                'TimeoutError': 'api_error',
                'ImportError': 'system_error',
                'ModuleNotFoundError': 'system_error',
                'FileNotFoundError': 'system_error',
                'PermissionError': 'system_error',
                'ValueError': 'input_error',
                'TypeError': 'input_error',
                'SyntaxError': 'input_error',
                'KeyError': 'input_error'
            }
            
            if exception_type in type_mapping:
                return type_mapping[exception_type]
        
        # Then check message patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return error_type
        
        return 'unknown_error'

class ErrorRecoverySystem:
    """Provides recovery strategies for different types of errors."""
    
    def __init__(self):
        self.recovery_strategies = {
            'api_error': self._recover_api_error,
            'skill_error': self._recover_skill_error,
            'input_error': self._recover_input_error,
            'system_error': self._recover_system_error,
            'llm_error': self._recover_llm_error,
            'unknown_error': self._recover_unknown_error
        }
        
        self.alternative_skills = {
            'ask': ['gemini', 'llm_plugin', 'search'],
            'gemini': ['ask', 'llm_plugin', 'search'],
            'search': ['ask', 'gemini', 'wikipedia'],
            'translate': ['gtranslate'],
            'solve': ['ask', 'gemini']
        }
    
    def recover_from_error(self, error_type: str, original_input: str, error_context: Dict) -> Dict[str, Any]:
        """Attempt to recover from an error."""
        recovery_func = self.recovery_strategies.get(error_type, self._recover_unknown_error)
        return recovery_func(original_input, error_context)
    
    def _recover_api_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from API-related errors."""
        failed_skill = context.get('failed_skill')
        
        recovery_plan = {
            'strategy': 'fallback_to_alternatives',
            'alternatives': [],
            'user_message': "I'm having trouble connecting to some services. Let me try alternative approaches:",
            'success': False
        }
        
        # Try alternative skills
        if failed_skill in self.alternative_skills:
            recovery_plan['alternatives'] = self.alternative_skills[failed_skill]
        else:
            recovery_plan['alternatives'] = ['ask', 'search', 'reason']
        
        return recovery_plan
    
    def _recover_skill_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from skill-related errors."""
        return {
            'strategy': 'suggest_similar_skills',
            'user_message': "I couldn't find that exact command. Here are some similar options:",
            'suggestions': self._find_similar_skills(original_input),
            'success': False
        }
    
    def _recover_input_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from input-related errors."""
        return {
            'strategy': 'provide_guidance',
            'user_message': "I had trouble understanding your request. Here's how you can rephrase it:",
            'guidance': self._generate_input_guidance(original_input, context),
            'success': False
        }
    
    def _recover_system_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from system-related errors."""
        return {
            'strategy': 'graceful_degradation',
            'user_message': "I'm experiencing some technical difficulties. Let me try a simpler approach:",
            'fallback_response': self._generate_fallback_response(original_input),
            'success': False
        }
    
    def _recover_llm_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from LLM-related errors."""
        recovery_plan = {
            'strategy': 'llm_fallback_chain',
            'user_message': "I'm having trouble with my AI processing. Trying backup systems:",
            'alternatives': ['gemini', 'ask', 'search', 'reason'],
            'success': False
        }
        
        # If context is too long, suggest chunking
        if 'context too long' in str(context.get('error_message', '')).lower():
            recovery_plan['user_message'] = "Your input is quite long. Let me break it down into smaller parts:"
            recovery_plan['strategy'] = 'chunk_input'
        
        return recovery_plan
    
    def _recover_unknown_error(self, original_input: str, context: Dict) -> Dict[str, Any]:
        """Recover from unknown errors."""
        return {
            'strategy': 'best_effort',
            'user_message': "Something unexpected happened, but let me try my best to help:",
            'alternatives': ['ask', 'search', 'reason'],
            'success': False
        }
    
    def _find_similar_skills(self, input_text: str) -> List[str]:
        """Find skills similar to the input."""
        from difflib import get_close_matches
        
        # Get available skills (this would be passed from the main system)
        try:
            from importlib import import_module
            main_mod = import_module("main")
            jarvis_instance = getattr(main_mod, "jarvis", None)
            
            if jarvis_instance:
                available_skills = list(jarvis_instance.skills.keys())
                words = input_text.lower().split()
                
                suggestions = []
                for word in words:
                    matches = get_close_matches(word, available_skills, n=3, cutoff=0.6)
                    suggestions.extend(matches)
                
                return list(set(suggestions))[:5]
        except Exception:
            pass
        
        return ['ask', 'search', 'help']
    
    def _generate_input_guidance(self, original_input: str, context: Dict) -> List[str]:
        """Generate helpful guidance for fixing input errors."""
        guidance = []
        
        error_msg = context.get('error_message', '').lower()
        
        if 'missing parameter' in error_msg:
            guidance.append("• Make sure to include all required information")
            guidance.append("• Try: 'command parameter1 parameter2'")
        
        if 'format error' in error_msg:
            guidance.append("• Check the format of your input")
            guidance.append("• Use quotes for multi-word parameters")
        
        if 'syntax error' in error_msg:
            guidance.append("• Check for typos in your command")
            guidance.append("• Try simpler phrasing")
        
        if not guidance:
            guidance = [
                "• Try rephrasing your question",
                "• Use simpler language",
                "• Break complex requests into smaller parts",
                "• Type 'help' to see available commands"
            ]
        
        return guidance
    
    def _generate_fallback_response(self, original_input: str) -> str:
        """Generate a basic fallback response."""
        if '?' in original_input:
            return f"I understand you're asking about: {original_input}. While I can't process this normally right now, I can tell you this is an interesting question that would benefit from web search or expert consultation."
        else:
            return f"I see you mentioned: {original_input}. This seems like something I should be able to help with once my systems are fully operational."

class UserFeedbackSystem:
    """Provides intelligent user feedback and guidance."""
    
    def __init__(self):
        self.feedback_history = deque(maxlen=100)
        self.success_patterns = defaultdict(int)
        self.failure_patterns = defaultdict(int)
    
    def generate_feedback(self, error_type: str, recovery_result: Dict, original_input: str) -> str:
        """Generate user-friendly feedback message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        feedback_parts = []
        
        # Add empathetic opening
        feedback_parts.append("🔧 **System Status Update:**")
        
        # Explain what happened
        error_explanations = {
            'api_error': "I encountered a connectivity issue with external services.",
            'skill_error': "I couldn't find the specific command you requested.",
            'input_error': "There was an issue with how your request was formatted.",
            'system_error': "I experienced a technical difficulty.",
            'llm_error': "My AI processing encountered an issue.",
            'unknown_error': "Something unexpected occurred."
        }
        
        explanation = error_explanations.get(error_type, "I encountered an issue.")
        feedback_parts.append(f"• **Issue:** {explanation}")
        
        # Add recovery information
        if recovery_result.get('user_message'):
            feedback_parts.append(f"• **Action:** {recovery_result['user_message']}")
        
        # Add specific guidance based on strategy
        strategy = recovery_result.get('strategy', '')
        
        if strategy == 'suggest_similar_skills':
            suggestions = recovery_result.get('suggestions', [])
            if suggestions:
                feedback_parts.append("• **Suggestions:**")
                for suggestion in suggestions[:3]:
                    feedback_parts.append(f"  - Try: `{suggestion}`")
        
        elif strategy == 'provide_guidance':
            guidance = recovery_result.get('guidance', [])
            if guidance:
                feedback_parts.append("• **How to fix:**")
                feedback_parts.extend([f"  {item}" for item in guidance])
        
        elif strategy in ['fallback_to_alternatives', 'llm_fallback_chain']:
            alternatives = recovery_result.get('alternatives', [])
            if alternatives:
                feedback_parts.append(f"• **Trying alternatives:** {', '.join(alternatives[:3])}")
        
        # Add helpful tips
        feedback_parts.append("")
        feedback_parts.append("💡 **Quick Tips:**")
        
        if error_type == 'skill_error':
            feedback_parts.append("• Type `help` or `skills` to see available commands")
            feedback_parts.append("• Try rephrasing your request in simpler terms")
        elif error_type == 'api_error':
            feedback_parts.append("• This is usually temporary - try again in a moment")
            feedback_parts.append("• I'll attempt to use backup services")
        elif error_type == 'input_error':
            feedback_parts.append("• Check your spelling and formatting")
            feedback_parts.append("• Try breaking complex requests into smaller parts")
        else:
            feedback_parts.append("• Try rephrasing your question")
            feedback_parts.append("• Use the `help` command for assistance")
        
        # Record feedback for learning
        self._record_feedback(error_type, recovery_result, original_input)
        
        return "\n".join(feedback_parts)
    
    def _record_feedback(self, error_type: str, recovery_result: Dict, original_input: str):
        """Record feedback for system learning."""
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'original_input': original_input,
            'recovery_strategy': recovery_result.get('strategy'),
            'success': recovery_result.get('success', False)
        }
        
        self.feedback_history.append(feedback_record)
        
        # Update patterns for learning
        pattern_key = f"{error_type}_{recovery_result.get('strategy', 'unknown')}"
        if recovery_result.get('success'):
            self.success_patterns[pattern_key] += 1
        else:
            self.failure_patterns[pattern_key] += 1

class EnhancedErrorHandler:
    """Main error handling system that orchestrates all components."""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.recovery_system = ErrorRecoverySystem()
        self.feedback_system = UserFeedbackSystem()
        self.error_log = deque(maxlen=1000)
    
    def handle_error(self, error: Exception, original_input: str, context: Dict = None) -> str:
        """Main error handling method."""
        if context is None:
            context = {}
        
        try:
            # Extract error information
            error_message = str(error)
            error_type_name = type(error).__name__
            
            # Log the error
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type_name,
                'error_message': error_message,
                'original_input': original_input,
                'context': context,
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_record)
            logger.error(f"Error handled: {error_type_name} - {error_message}")
            
            # Classify the error
            error_category = self.classifier.classify_error(error_message, error_type_name)
            
            # Add error info to context
            context.update({
                'error_message': error_message,
                'error_type': error_type_name,
                'error_category': error_category
            })
            
            # Attempt recovery
            recovery_result = self.recovery_system.recover_from_error(error_category, original_input, context)
            
            # Generate user feedback
            feedback = self.feedback_system.generate_feedback(error_category, recovery_result, original_input)
            
            # Try to execute recovery if possible
            if recovery_result.get('alternatives'):
                recovery_attempt = self._attempt_recovery(original_input, recovery_result['alternatives'], context)
                if recovery_attempt['success']:
                    feedback += f"\n\n✅ **Success!** I was able to process your request using an alternative method:\n{recovery_attempt['result']}"
                    recovery_result['success'] = True
            
            return feedback
            
        except Exception as meta_error:
            # If error handling itself fails, provide basic feedback
            logger.critical(f"Error handler failed: {meta_error}")
            return f"I encountered a critical error while trying to help you. Please try rephrasing your request or contact support. Error: {str(error)[:100]}..."
    
    def _attempt_recovery(self, original_input: str, alternatives: List[str], context: Dict) -> Dict[str, Any]:
        """Attempt to recover by trying alternative skills."""
        try:
            from importlib import import_module
            main_mod = import_module("main")
            jarvis_instance = getattr(main_mod, "jarvis", None)
            
            if not jarvis_instance:
                return {'success': False, 'result': None}
            
            for skill_name in alternatives:
                if skill_name in jarvis_instance.skills:
                    try:
                        # Try the alternative skill
                        if skill_name == "ask":
                            result = jarvis_instance.skills[skill_name](f"ask {original_input}")
                        else:
                            result = jarvis_instance.skills[skill_name](original_input)
                        
                        if result and not result.lower().startswith(("sorry", "error", "failed")):
                            return {'success': True, 'result': result, 'used_skill': skill_name}
                    
                    except Exception as e:
                        logger.warning(f"Alternative skill {skill_name} also failed: {e}")
                        continue
            
            return {'success': False, 'result': None}
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return {'success': False, 'result': None}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        if not self.error_log:
            return {'message': 'No errors recorded yet'}
        
        stats = {
            'total_errors': len(self.error_log),
            'error_types': defaultdict(int),
            'error_categories': defaultdict(int),
            'recent_errors': list(self.error_log)[-5:]  # Last 5 errors
        }
        
        for error_record in self.error_log:
            stats['error_types'][error_record['error_type']] += 1
            stats['error_categories'][error_record.get('context', {}).get('error_category', 'unknown')] += 1
        
        return dict(stats)

# Global error handler instance
error_handler = EnhancedErrorHandler()

def enhanced_error_handling_skill(user_input: str, **kwargs) -> str:
    """
    Error handling skill that can be called directly to get error information.
    
    Usage:
    - errors
    - error stats
    - error log
    """
    
    if user_input.lower().strip() in ['errors', 'error stats', 'error statistics']:
        stats = error_handler.get_error_statistics()
        
        if 'message' in stats:
            return stats['message']
        
        response_parts = []
        response_parts.append("📊 **Error Statistics:**")
        response_parts.append(f"• Total errors handled: {stats['total_errors']}")
        response_parts.append("")
        
        if stats['error_categories']:
            response_parts.append("**Error Categories:**")
            for category, count in stats['error_categories'].items():
                response_parts.append(f"• {category.replace('_', ' ').title()}: {count}")
            response_parts.append("")
        
        if stats['recent_errors']:
            response_parts.append("**Recent Errors:**")
            for error in stats['recent_errors']:
                timestamp = error['timestamp'][:19]  # Remove microseconds
                response_parts.append(f"• {timestamp}: {error['error_type']} - {error['original_input'][:50]}...")
        
        return "\n".join(response_parts)
    
    elif user_input.lower().strip() in ['error log', 'error history']:
        if not error_handler.error_log:
            return "No errors recorded in the current session."
        
        response_parts = []
        response_parts.append("📜 **Error Log (Last 10 entries):**")
        response_parts.append("")
        
        for error in list(error_handler.error_log)[-10:]:
            timestamp = error['timestamp'][:19]
            response_parts.append(f"**{timestamp}**")
            response_parts.append(f"Type: {error['error_type']}")
            response_parts.append(f"Input: {error['original_input']}")
            response_parts.append(f"Message: {error['error_message'][:100]}...")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    else:
        return "Error handling commands: 'errors', 'error stats', 'error log'"

def register(jarvis):
    """Register the error handling skill."""
    jarvis.register_skill("errors", enhanced_error_handling_skill)
    jarvis.register_skill("error_stats", enhanced_error_handling_skill)
    
    # Make the error handler available globally
    jarvis.error_handler = error_handler
