"""
Configuration for Jarvis LLM provider selection and endpoints.
"""

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()
import os

# LLM provider: 'openai', 'ollama', 'lmstudio', 'huggingface', or 'auto'
LLM_PROVIDER = os.getenv("JARVIS_LLM_PROVIDER", "auto").lower()

# Optional: custom endpoints for local providers
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")
# Use transformers pipeline directly for local fine-tuned model
HUGGINGFACE_URL = os.getenv("HUGGINGFACE_URL", "transformers")

# OpenAI API key (for cloud or compatible endpoints)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model names (can be customized)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "gpt-3.5-turbo")
# Use the local PEFT/LoRA fine-tuned model for HuggingFace provider
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "finetune/results_peft")
