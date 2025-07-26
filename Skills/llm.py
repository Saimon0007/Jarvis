"""
LLM skill for Jarvis: Uses OpenAI API (or compatible) for advanced Q&A and conversation.
Usage: ask <your question>
"""

import os
import re
import time
import math
import json
import requests
import logging
import threading
from importlib import import_module
from datetime import datetime
from typing import Optional, Any, Callable, List, Tuple, Dict, Union

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

class TokenManager:
    """Manages token counting and context window handling for different LLM providers."""
    
    # Default token limits for different models
    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "claude-2": 100000,
        "llama3": 4096,
        "default": 4096
    }
    
    # Approximate tokens per character for different languages
    TOKENS_PER_CHAR = {
        "english": 0.25,  # ~4 chars per token
        "chinese": 1.0,   # ~1 char per token
        "japanese": 0.8,  # ~1.25 chars per token
        "korean": 0.8,    # ~1.25 chars per token
        "default": 0.25
    }
    
    def __init__(self, model: str = "default", language: str = "english"):
        self.model = model
        self.language = language.lower()
        self.token_limit = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["default"])
        self.tokens_per_char = self.TOKENS_PER_CHAR.get(language, self.TOKENS_PER_CHAR["default"])
        self._tiktoken_encoder = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self._tiktoken_encoder = tiktoken.encoding_for_model(model)
            except Exception:
                try:
                    self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken if available, else estimate."""
        if self._tiktoken_encoder:
            return len(self._tiktoken_encoder.encode(text))
        return math.ceil(len(text) * self.tokens_per_char)
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of chat messages."""
        total = 0
        for msg in messages:
            # Add overhead for message format
            total += 4  # Every message has a base overhead
            for key, value in msg.items():
                total += self.count_tokens(value)
                total += 1  # Key overhead
        return total + 2  # Add overhead for chat format
    
    def trim_messages_to_fit(self, messages: List[Dict[str, str]], max_new_tokens: int = 1000) -> List[Dict[str, str]]:
        """Trim messages to fit within context window while reserving space for response."""
        available_tokens = self.token_limit - max_new_tokens
        total_tokens = self.count_messages_tokens(messages)
        
        if total_tokens <= available_tokens:
            return messages
        
        # Always keep system message and last user message
        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
        
        required_tokens = (self.count_messages_tokens([system_msg]) if system_msg else 0) + \
                         (self.count_messages_tokens([last_user_msg]) if last_user_msg else 0)
        
        # Keep as much context as possible
        result = []
        if system_msg:
            result.append(system_msg)
        
        remaining_tokens = available_tokens - required_tokens
        for msg in reversed(messages[1:-1] if last_user_msg else messages[1:]):
            msg_tokens = self.count_messages_tokens([msg])
            if remaining_tokens - msg_tokens >= 0:
                result.insert(1, msg)  # Insert after system message
                remaining_tokens -= msg_tokens
            else:
                break
        
        if last_user_msg:
            result.append(last_user_msg)
        
        return result
    
    def ensure_response_fits(self, response: str, max_tokens: int) -> str:
        """Ensure response fits within token limit, trimming if necessary."""
        tokens = self.count_tokens(response)
        if tokens <= max_tokens:
            return response
            
        if self._tiktoken_encoder:
            # Precise trimming with tiktoken
            encoded = self._tiktoken_encoder.encode(response)
            trimmed = self._tiktoken_encoder.decode(encoded[:max_tokens])
            # Try to trim at sentence boundary
            last_period = trimmed.rfind('.')
            if last_period > len(trimmed) * 0.75:  # Only trim at period if we keep most of the content
                trimmed = trimmed[:last_period + 1]
            return trimmed.strip()
        
        # Approximate trimming without tiktoken
        char_limit = math.floor(max_tokens / self.tokens_per_char)
        trimmed = response[:char_limit]
        # Try to trim at sentence boundary
        last_period = trimmed.rfind('.')
        if last_period > len(trimmed) * 0.75:
            trimmed = trimmed[:last_period + 1]
        return trimmed.strip()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    config = import_module("config")
except ImportError:
    config = None

def get_config(attr: str, default: Optional[Any] = None) -> Any:
    if config and hasattr(config, attr):
        return getattr(config, attr)
    return default

def validate_api_keys() -> Dict[str, str]:
    """Check for presence of required API keys and return missing info for reporting."""
    missing = {}
    provider = get_config("LLM_PROVIDER", "auto")
    
    # Only check OpenAI if it's the specified provider or in auto mode
    if provider in ["auto", "openai"]:
        if not os.getenv("OPENAI_API_KEY"):
            missing["OpenAI"] = "OPENAI_API_KEY"
    
    # Check for provider-specific requirements
    if provider == "huggingface" and not os.getenv("HUGGINGFACE_API_KEY"):
        missing["HuggingFace"] = "HUGGINGFACE_API_KEY"
    
    # Don't require API keys for local providers
    if provider in ["ollama", "lmstudio", "huggingface_local"]:
        return {}
    
    return missing

def format_error(err: Exception) -> str:
    """Mask exception details for security, include timestamp for tracking."""
    return f"An error occurred during LLM processing at {datetime.utcnow().isoformat()}Z. Please check logs or configuration."

def load_conversation_history(conversation_history: Optional[List[Tuple[str, str]]]) -> List[Dict[str, str]]:
    """Load and process conversation history with smart context retention."""
    if not conversation_history:
        return []

    MAX_TOKENS = 3000  # Approximate token limit for context
    CHAR_PER_TOKEN = 4  # Approximate chars per token
    MAX_CHARS = MAX_TOKENS * CHAR_PER_TOKEN
    
    messages = []
    total_chars = 0
    important_markers = {"question:", "answer:", "summary:", "conclusion:", "problem:", "solution:"}
    
    # First pass: Mark importance and calculate sizes
    history_with_importance = []
    for role, msg in conversation_history:
        importance = sum(1 for marker in important_markers if marker in msg.lower()) + 1
        chars = len(msg)
        history_with_importance.append((role, msg, importance, chars))
    
    # Process most recent messages first
    for role, msg, importance, chars in reversed(history_with_importance):
        # Always include the last 2 exchanges regardless of size
        if len(messages) < 4:  # 2 exchanges = 4 messages (2 user + 2 assistant)
            messages.insert(0, {
                "role": "user" if role.lower() not in {"jarvis", "assistant", "system"} else "assistant",
                "content": msg
            })
            total_chars += chars
            continue
        
        # For older messages, be selective based on importance and size
        if total_chars + chars <= MAX_CHARS:
            # Include if important or not too long
            if importance > 1 or chars < 500:
                messages.insert(0, {
                    "role": "user" if role.lower() not in {"jarvis", "assistant", "system"} else "assistant",
                    "content": msg
                })
                total_chars += chars
        
        # Stop if we're getting too big
        if total_chars >= MAX_CHARS:
            break
    
    return messages

def get_system_prompt(search_context: str = "", custom: Optional[str] = None,
                   conversation_type: str = "general") -> str:
    """Generate a dynamic system prompt based on conversation context and type.
    
    Args:
        search_context: Additional context from search results
        custom: Custom override prompt
        conversation_type: Type of conversation (general, technical, creative, etc.)
    """
    if custom:
        return custom + search_context
        
    # Base personality and principles
    base_prompt = (
        "You are Jarvis, an expert AI assistant with a helpful and professional demeanor. "
        "Always maintain a consistent personality that is:"
        "\n- Knowledgeable yet approachable"
        "\n- Direct but polite"
        "\n- Precise while remaining conversational"
        "\n\nCore principles:"
        "\n- Provide accurate, well-structured information"
        "\n- Admit uncertainty when appropriate"
        "\n- Be concise but thorough"
        "\n- Use clear formatting for readability"
    )
    
    # Context-specific additions based on conversation type
    type_specific = {
        "technical": (
            "\n\nFor technical discussions:"
            "\n- Use code blocks with syntax highlighting when sharing code"
            "\n- Include brief explanations with examples"
            "\n- Break down complex concepts into digestible parts"
            "\n- Reference relevant documentation when applicable"
        ),
        "creative": (
            "\n\nFor creative discussions:"
            "\n- Encourage exploration of ideas"
            "\n- Offer multiple perspectives or approaches"
            "\n- Use analogies to explain complex concepts"
            "\n- Balance creativity with practicality"
        ),
        "analytical": (
            "\n\nFor analytical discussions:"
            "\n- Present structured, logical arguments"
            "\n- Include relevant data or examples"
            "\n- Consider multiple factors in analysis"
            "\n- Highlight key assumptions and limitations"
        ),
        "educational": (
            "\n\nFor educational discussions:"
            "\n- Break down concepts into clear learning steps"
            "\n- Provide examples and analogies"
            "\n- Encourage understanding over memorization"
            "\n- Include practice suggestions when relevant"
        )
    }
    
    # Add type-specific instructions
    full_prompt = base_prompt + type_specific.get(conversation_type, "")
    
    # Add formatting guidelines
    formatting_guide = (
        "\n\nFormatting guidelines:"
        "\n- Use markdown for structure and emphasis"
        "\n- Break long responses into sections with headers"
        "\n- Use bullet points or numbered lists for multiple items"
        "\n- Include code blocks with proper syntax highlighting"
    )
    
    # Combine everything with search context
    return full_prompt + formatting_guide + search_context

def try_search_context(prompt: str, search_skill: Optional[Callable], logger: logging.Logger) -> Tuple[str, str]:
    """Enhanced RAG implementation with better context gathering and conversation type detection.
    
    Returns:
        Tuple[str, str]: (search_context, conversation_type)
    """
    if not search_skill:
        return ""

    # Enhanced trigger patterns for different types of queries
    patterns = {
        'factual': [
            "what", "who", "when", "where", "which", "how many", "how much",
            "explain", "define", "describe", "tell me about", "give me information"
        ],
        'comparison': [
            "compare", "difference between", "similarities between", "versus", "vs",
            "better than", "worse than", "pros and cons"
        ],
        'historical': [
            "history of", "origin of", "background of", "development of",
            "evolution of", "timeline", "in the past"
        ],
        'technical': [
            "how to", "how do", "process of", "steps to", "method for",
            "technique", "implementation", "working of"
        ]
    }

    def should_search(text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        for category, triggers in patterns.items():
            if any(trigger in text_lower for trigger in triggers):
                return True, category
        return False, ""

    try:
        needs_search, query_type = should_search(prompt)
        
        # Map query types to conversation types
        conversation_type_mapping = {
            'technical': "technical",
            'factual': "analytical",
            'comparison': "analytical",
            'historical': "educational",
            'creative': "creative"
        }
        conversation_type = conversation_type_mapping.get(query_type, "general")
        
        if not needs_search:
            return "", conversation_type

        # Prepare focused search queries based on query type
        search_queries = []
        cleaned_prompt = prompt.strip("?!.,").lower()

        if query_type == 'comparison':
            # Split comparison queries to search both sides
            for split_word in ["versus", "vs", "compared to", "or"]:
                if split_word in cleaned_prompt:
                    parts = cleaned_prompt.split(split_word)
                    search_queries.extend(parts)
                    break
            if not search_queries:
                search_queries = [cleaned_prompt]
        
        elif query_type == 'technical':
            # Add specific context for technical queries
            search_queries = [
                cleaned_prompt,
                f"step by step {cleaned_prompt}",
                f"tutorial {cleaned_prompt}"
            ]
        else:
            search_queries = [cleaned_prompt]

        # Gather and combine results
        all_results = []
        for query in search_queries[:2]:  # Limit to top 2 queries for performance
            try:
                result = search_skill(f"search {query}")
                if result and not result.lower().startswith("sorry"):
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Individual search failed for '{query}': {e}")

        if not all_results:
            return ""

        # Format results based on query type
        if query_type == 'comparison':
            context = "\n\n[Comparison information:\n"
            for idx, result in enumerate(all_results):
                context += f"• {result}\n"
            context += "]"
        elif query_type == 'technical':
            context = "\n\n[Technical information:\n"
            context += "\n".join(f"• {result}" for result in all_results)
            context += "]"
        else:
            # Combine factual results with bullet points
            context = "\n\n[Relevant information:\n"
            context += "\n".join(f"• {result}" for result in all_results)
            context += "]"

        return context, conversation_type

    except Exception as e:
        logger.warning(f"Search context fetch failed: {e}")
        return "", "general"

class ProviderError(Exception):
    """Base class for provider-related errors."""
    pass

class AuthenticationError(ProviderError):
    """Raised when authentication fails (e.g., invalid API key)."""
    pass

class RateLimitError(ProviderError):
    """Raised when hitting rate limits."""
    pass

class ContextLengthError(ProviderError):
    """Raised when input exceeds model's context length."""
    pass

class ServiceUnavailableError(ProviderError):
    """Raised when the service is temporarily unavailable."""
    pass

class InvalidRequestError(ProviderError):
    """Raised when the request is malformed or invalid."""
    pass

class LLMProviderBase:
    name: str
    def __init__(self, model: Optional[str], logger: logging.Logger):
        self.model = model
        self.logger = logger
        # Default timeout and retry settings
        self.timeout = get_config("PROVIDER_TIMEOUT", 60)  # 60 second default timeout
        self.max_retries = get_config("PROVIDER_MAX_RETRIES", 3)
        self.retry_delay = get_config("PROVIDER_RETRY_DELAY", 2)
        self.backoff_factor = get_config("PROVIDER_BACKOFF_FACTOR", 2)
        
        # Token management
        self.token_manager = TokenManager(
            model=model or "default",
            language=get_config("LANGUAGE", "english")
        )
        self.max_output_tokens = get_config(f"{self.name.upper()}_MAX_OUTPUT_TOKENS", 1000)

    def classify_error(self, exception: Exception) -> Tuple[type, bool]:
        """Classify the error and determine if it's retryable."""
        if isinstance(exception, requests.exceptions.RequestException):
            resp = getattr(exception, 'response', None)
            if resp is not None:
                status_code = resp.status_code
                
                # Authentication errors (non-retryable)
                if status_code in {401, 403}:
                    return AuthenticationError, False
                    
                # Rate limiting (retryable with backoff)
                if status_code == 429:
                    return RateLimitError, True
                    
                # Service unavailable (retryable)
                if status_code in {500, 502, 503, 504}:
                    return ServiceUnavailableError, True
                    
                # Timeout (retryable)
                if status_code == 408:
                    return ServiceUnavailableError, True
                    
            # Connection issues (retryable)
            if isinstance(exception, (requests.exceptions.Timeout,
                                   requests.exceptions.ConnectionError)):
                return ServiceUnavailableError, True
                
        # Check response content for specific error types
        if isinstance(exception, Exception):
            error_msg = str(exception).lower()
            
            if "api key" in error_msg or "authentication" in error_msg:
                return AuthenticationError, False
                
            if "rate limit" in error_msg:
                return RateLimitError, True
                
            if any(phrase in error_msg for phrase in ["context length", "too long", "maximum length"]):
                return ContextLengthError, False
                
            if "invalid request" in error_msg:
                return InvalidRequestError, False
                
        return ProviderError, True

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Enhanced retry decision logic with error classification."""
        if attempt >= self.max_retries:
            return False
            
        error_class, is_retryable = self.classify_error(exception)
        
        if not is_retryable:
            return False
            
        # Adjust retry behavior based on error type
        if error_class == RateLimitError:
            # Use longer backoff for rate limits
            self.retry_delay = max(self.retry_delay, 5)
            self.backoff_factor = max(self.backoff_factor, 3)
            
        elif error_class == ServiceUnavailableError:
            # Use moderate backoff for service issues
            self.retry_delay = max(self.retry_delay, 2)
            self.backoff_factor = max(self.backoff_factor, 2)
            
        return True

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self.should_retry(e, attempt):
                    break
                    
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.backoff_factor ** attempt)
                self.logger.warning(
                    f"{self.name} request failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying in {delay:.1f}s: {str(e)}")
                time.sleep(delay)
        
        raise last_exception

    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        raise NotImplementedError("Provider must implement chat()")

class OpenAIProvider(LLMProviderBase):
    name = "openai"
    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        
        def make_request():
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model or "gpt-3.5-turbo",
                "messages": messages
            }
            url = "https://api.openai.com/v1/chat/completions"
            resp = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
            
        return self.execute_with_retry(make_request)

class OllamaProvider(LLMProviderBase):
    name = "ollama"
    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        def make_request():
            url = get_config("OLLAMA_URL", "http://localhost:11434/api/chat")
            data = {
                "model": self.model or "llama3",
                "messages": messages
            }
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if "message" in result:
                return result["message"]["content"].strip()
            elif "choices" in result:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return str(result)
        
        return self.execute_with_retry(make_request)

class LMStudioProvider(LLMProviderBase):
    name = "lmstudio"
    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        def make_request():
            url = get_config("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")
            data = {
                "model": self.model or "gpt-3.5-turbo",
                "messages": messages
            }
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
            
        return self.execute_with_retry(make_request)

class HuggingFaceProvider(LLMProviderBase):
    name = "huggingface"
    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        def make_request():
            url = get_config("HUGGINGFACE_URL", "http://localhost:5000/generate")
            data = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 256, "return_full_text": False},
                "model": self.model or "HuggingFaceH4/zephyr-7b-beta"
            }
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
            elif isinstance(result, list) and result and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            elif isinstance(result, dict) and "choices" in result and result["choices"]:
                return result["choices"][0].get("text", "").strip()
            else:
                return str(result)
                
        return self.execute_with_retry(make_request)

class HFTransformersLocalProvider(LLMProviderBase):
    name = "huggingface_local"
    def chat(self, messages: List[Dict[str, str]], prompt: str) -> str:
        def make_request():
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            model_dir = get_config("HUGGINGFACE_MODEL", "finetune/results_peft")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            full_prompt = f"<|user|> {prompt}\n<|assistant|>"
            output = pipe(full_prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
            if "<|assistant|>" in output:
                reply = output.split("<|assistant|>", 1)[-1].strip()
                return reply if reply else output.strip()
            return output.strip()
            
        return self.execute_with_retry(make_request)

def get_provider_registry(logger: logging.Logger) -> Dict[str, Callable[..., LLMProviderBase]]:
    return {
        "openai": lambda model: OpenAIProvider(model, logger),
        "ollama": lambda model: OllamaProvider(model, logger),
        "lmstudio": lambda model: LMStudioProvider(model, logger),
        "huggingface": lambda model: HuggingFaceProvider(model, logger),
        "huggingface_local": lambda model: HFTransformersLocalProvider(model, logger)
    }

def get_preferred_providers(provider: str, logger: logging.Logger) -> List[LLMProviderBase]:
    registry = get_provider_registry(logger)
    
    # Check for local providers first
    local_providers = {
        "ollama": "llama3",
        "lmstudio": "gpt-4",
        "huggingface_local": get_config("HUGGINGFACE_MODEL", None)
    }
    
    # Then cloud providers
    cloud_providers = {
        "openai": "gpt-4",
        "huggingface": "meta-llama/Llama-3-70b-chat-hf"
    }
    
    # Combine all providers with priority to local ones
    preferred = [(k, v) for k, v in local_providers.items()] + \
               [(k, v) for k, v in cloud_providers.items()]
    if provider != "auto":
        return [registry[provider](get_config(f"{provider.upper()}_MODEL", None))]
    result = []
    for prov, model in preferred:
        if prov in registry:
            result.append(registry[prov](model))
    # Add configurable fallbacks
    for prov in ["openai", "ollama", "lmstudio", "huggingface"]:
        if prov in registry:
            result.append(registry[prov](get_config(f"{prov.upper()}_MODEL", None)))
    # HuggingFace transformers local mode
    if get_config("HUGGINGFACE_URL") == "transformers":
        result.append(registry["huggingface_local"](get_config("HUGGINGFACE_MODEL", None)))
    return result

def do_provider_request(provider: LLMProviderBase, messages: List[Dict[str, str]], prompt: str, 
                      result_dict: dict, lock: threading.Lock, end_time: float) -> None:
    """Execute a provider request with timeout awareness."""
    try:
        # Calculate remaining time for this provider
        remaining_time = max(0.1, end_time - time.time())
        
        # Start a timer thread to monitor provider execution
        timer_expired = threading.Event()
        def timeout_monitor():
            time.sleep(remaining_time)
            timer_expired.set()
        
        timer = threading.Thread(target=timeout_monitor)
        timer.daemon = True
        timer.start()
        
        # Execute provider chat with timeout awareness
        response = None
        try:
            response = provider.chat(messages, prompt)
        finally:
            # If we got a response before timeout, cancel the timer
            if not timer_expired.is_set():
                timer_expired.set()
                timer.join(0.1)  # Give timer a chance to clean up
        
        # Store result if we got one and haven't already found a faster response
        if response:
            with lock:
                if "result" not in result_dict or not result_dict["result"]:
                    result_dict["result"] = response
                    result_dict["responded_provider"] = provider.name
                    
    except Exception as e:
        with lock:
            if "errors" not in result_dict:
                result_dict["errors"] = []
            result_dict["errors"].append(f"{provider.name} ({provider.model}): {str(e)}")

def llm_skill(
    user_input: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
    search_skill: Optional[Callable[[str], str]] = None,
    skills: Optional[dict] = None,
    system_prompt_override: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    LLM skill for Jarvis. Can be used with or without 'ask' prefix.
    Uses conversation history for context and can include search results for RAG.
    Now supports extensible provider registry, better error handling, and async fallback.
    """
    logger = logger or logging.getLogger("llm_skill")
    
    # Handle input with or without "ask" prefix
    prompt = user_input.strip()
    if prompt.lower().startswith("ask"):
        prompt = prompt[len("ask"):].strip()
    
    if not prompt:
        return "Please provide a question or prompt."

    # Check for missing API keys and inform user early
    missing_keys = validate_api_keys()
    if missing_keys:
        key_info = ", ".join(f"{k} ({v})" for k, v in missing_keys.items())
        return f"Missing required API keys: {key_info}. Please check your configuration."

    # Fetch web/search results and determine conversation type
    search_context, conversation_type = try_search_context(prompt, search_skill, logger)

    # Prepare and validate messages with token management
    base_messages = [{
        "role": "system", 
        "content": get_system_prompt(
            search_context=search_context,
            custom=system_prompt_override,
            conversation_type=conversation_type
        )
    }]
    base_messages.extend(load_conversation_history(conversation_history))
    base_messages.append({"role": "user", "content": prompt})

    # Provider registry and preference
    provider = get_config("LLM_PROVIDER", "auto")
    providers = get_preferred_providers(provider, logger)
    
    # Create token-managed message sets for each provider
    provider_messages = {}
    for prov in providers:
        try:
            # Trim messages to fit each provider's context window
            managed_messages = prov.token_manager.trim_messages_to_fit(
                base_messages,
                max_new_tokens=prov.max_output_tokens
            )
            provider_messages[prov.name] = managed_messages
        except Exception as e:
            logger.warning(f"Token management failed for {prov.name}: {e}")
            provider_messages[prov.name] = base_messages  # Fallback to base messages

    # Provider registry and preference
    provider = get_config("LLM_PROVIDER", "auto")
    providers = get_preferred_providers(provider, logger)

    # Try all providers in parallel with smart timeout management
    result_dict = {"result": None, "errors": [], "responded_provider": None}
    threads = []
    lock = threading.Lock()

    # Get timeout settings
    default_timeout = get_config("PARALLEL_TIMEOUT", 30)  # 30 second default for parallel execution
    provider_timeouts = {
        "openai": get_config("OPENAI_TIMEOUT", default_timeout),
        "ollama": get_config("OLLAMA_TIMEOUT", default_timeout),
        "lmstudio": get_config("LMSTUDIO_TIMEOUT", default_timeout),
        "huggingface": get_config("HUGGINGFACE_TIMEOUT", default_timeout),
        "huggingface_local": get_config("HUGGINGFACE_LOCAL_TIMEOUT", default_timeout * 2)  # Local models may need more time
    }
    
    # Calculate end time for the entire parallel execution
    max_timeout = max(provider_timeouts.values())
    end_time = time.time() + max_timeout
    
    # Start all providers in parallel with their specific message sets
    for prov in providers:
        timeout = provider_timeouts.get(prov.name, default_timeout)
        provider_specific_messages = provider_messages.get(prov.name, base_messages)
        t = threading.Thread(
            target=do_provider_request,
            args=(prov, provider_specific_messages, prompt, result_dict, lock, end_time)
        )
        t.daemon = True  # Allow thread to be terminated when main thread ends
        threads.append(t)
        t.start()
    
    # Wait for results with periodic checks
    check_interval = 0.5  # Check every 500ms
    start_time = time.time()
    
    while time.time() < end_time and not result_dict.get("result"):
        # Check if any thread is still alive
        active_threads = [t for t in threads if t.is_alive()]
        if not active_threads:
            break
            
        # Wait a bit before checking again
        time.sleep(check_interval)
        
        # Log progress for long-running requests
        elapsed = time.time() - start_time
        if elapsed > 5 and elapsed % 5 < check_interval:  # Log every 5 seconds
            active_providers = [p.name for p, t in zip(providers, threads) if t.is_alive()]
            logger.info(f"Still waiting for providers after {elapsed:.1f}s: {', '.join(active_providers)}")
    
    # If we got a result, return it along with provider info
    if result_dict.get("result"):
        provider_info = f" (from {result_dict['responded_provider']})" if result_dict.get('responded_provider') else ""
        return result_dict["result"] + provider_info
    logger.warning(f"All providers failed. Errors: {result_dict.get('errors')}")

    # Fallback: try search skill directly if all LLMs fail
    try:
        main_mod = import_module("main")
        jarvis_instance = getattr(main_mod, "jarvis", None)
        if jarvis_instance and "search" in jarvis_instance.skills:
            search_result = jarvis_instance.skills["search"](f"search {prompt}")
            if search_result and not search_result.lower().startswith("sorry"):
                return f"(From web search): {search_result}"
    except Exception as e:
        logger.warning(f"Web search fallback failed: {e}")

    # Enhanced math fallback with better expression parsing
    import re
    from decimal import Decimal
    
    def evaluate_expression(expr: str) -> Optional[str]:
        try:
            # Replace word operators with symbols
            word_to_symbol = {
                'plus': '+', 'minus': '-', 
                'times': '*', 'multiplied by': '*',
                'divided by': '/', 'over': '/',
                'x': '*'  # Common multiplication symbol
            }
            cleaned = expr.lower()
            for word, symbol in word_to_symbol.items():
                cleaned = cleaned.replace(word, symbol)
            
            # Extract numbers and operators
            numbers = re.findall(r'-?\d*\.?\d+', cleaned)
            operators = re.findall(r'[+\-*/]', cleaned)
            
            if not numbers:
                return None
                
            # Convert to decimal for precise calculation
            result = Decimal(numbers[0])
            for i, op in enumerate(operators):
                if i + 1 >= len(numbers):
                    break
                num = Decimal(numbers[i + 1])
                if op == '+':
                    result += num
                elif op == '-':
                    result -= num
                elif op == '*':
                    result *= num
                elif op == '/':
                    if num == 0:
                        return "Cannot divide by zero"
                    result /= num
            
            # Format result to remove trailing zeros after decimal
            return str(result.normalize())
        except Exception as e:
            logger.warning(f"Enhanced math evaluation failed: {e}")
            return None
    
    # Try to evaluate any mathematical expression in the prompt
    math_result = evaluate_expression(prompt)
    if math_result is not None:
        return math_result

    # Advanced math fallback: use solve skill for complex math if available
    if skills and "solve" in skills:
        try:
            solve_resp = skills["solve"](f"solve {prompt}")
            if solve_resp and not solve_resp.lower().startswith("sorry"):
                return solve_resp
        except Exception as e:
            logger.warning(f"Solve skill fallback failed: {e}")

    # Enhanced error handling and fallback system
    if result_dict.get("errors"):
        error_types = {
            "authentication": [],
            "rate_limit": [],
            "context_length": [],
            "service": [],
            "timeout": [],
            "other": []
        }
        
        # Categorize errors
        for error in result_dict.get("errors", []):
            if "API key" in error.lower() or "authentication" in error.lower():
                error_types["authentication"].append(error)
            elif "rate limit" in error.lower() or "too many requests" in error.lower():
                error_types["rate_limit"].append(error)
            elif "context length" in error.lower() or "too long" in error.lower():
                error_types["context_length"].append(error)
            elif "unavailable" in error.lower() or "connection" in error.lower():
                error_types["service"].append(error)
            elif "timeout" in error.lower():
                error_types["timeout"].append(error)
            else:
                error_types["other"].append(error)

        # Generate helpful response based on error types
        if error_types["authentication"]:
            return ("I'm having trouble accessing some services due to authentication issues. "
                   "Please check the API keys configuration.")
        
        if error_types["rate_limit"]:
            return ("I've hit the rate limit with some providers. Please try again in a few minutes, "
                   "or consider adjusting the rate limits in the configuration.")
        
        if error_types["context_length"]:
            return ("Your question is a bit too long for processing. Could you try breaking it down "
                   "into smaller parts? I'll be happy to help with each part separately.")
        
        if error_types["service"]:
            service_errors = len(error_types["service"])
            total_providers = len(providers)
            if service_errors == total_providers:
                return ("I'm currently experiencing connectivity issues with all providers. "
                       "Please check your internet connection or try again later.")
            else:
                return ("Some services are currently unavailable, but I'm still working on getting "
                       "you an answer. Please try again, and I'll attempt to use available providers.")
        
        if error_types["timeout"]:
            timeout_errors = len(error_types["timeout"])
            if timeout_errors > 1:
                return ("The request is taking longer than expected. This might be due to high complexity "
                       "or current system load. Would you like to:\n"
                       "1. Try again with a simpler question\n"
                       "2. Break down your question into smaller parts\n"
                       "3. Use a different approach (like search or mathematical computation)")
        
        # Detailed error report for debugging or when no specific category matches
        err_msgs = []
        for category, errors in error_types.items():
            if errors:
                err_msgs.extend(errors)
        
        if logger.getEffectiveLevel() <= logging.DEBUG:
            return f"Request failed. Detailed errors:\n" + "\n".join(err_msgs)
        else:
            return ("I encountered some issues processing your request. You might want to:\n"
                   "1. Rephrase your question\n"
                   "2. Try again in a few moments\n"
                   "3. Use a different skill or approach")

    return "No provider was able to process your request. Please try again or rephrase your question."

def register(jarvis):
    # Register with skills argument using a wrapper
    def llm_skill_with_skills(user_input, conversation_history=None, search_skill=None):
        return llm_skill(user_input, conversation_history=conversation_history, search_skill=search_skill, skills=jarvis.skills)
    jarvis.register_skill("ask", llm_skill_with_skills)