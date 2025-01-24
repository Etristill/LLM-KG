# src/config.py

from typing import Dict, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for language models"""
    name: str
    provider: str  # "anthropic" or "openai" for now
    api_key_env: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None

class Config:
    """Central configuration for the entire projedn"""
    
    # Default model configurations
    MODEL_CONFIGS = {
        "claude-3-opus-20240229": ModelConfig(
            name="claude-3-opus-20240229",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.7
        ),
        "claude-3-sonnet-20240229": ModelConfig(
            name="claude-3-sonnet-20240229",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.7
        ),
        "gpt-4-turbo-preview": ModelConfig(
            name="gpt-4-turbo-preview",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7
        )
    }

    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        """Initialize configuration with specified model"""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        self.model_config = self.MODEL_CONFIGS[model_name]
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Ensure required API key is available"""
        api_key = os.getenv(self.model_config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {self.model_config.api_key_env} environment variable not found")
    
    @property
    def model_name(self) -> str:
        """Get current model name"""
        return self.model_config.name
    
    @property
    def is_anthropic(self) -> bool:
        """Check if current model is from Anthropic"""
        return self.model_config.provider == "anthropic"
    
    @property
    def is_openai(self) -> bool:
        """Check if current model is from OpenAI"""
        return self.model_config.provider == "openai"
    
    @property
    def api_key(self) -> str:
        """Get API key for current model"""
        return os.getenv(self.model_config.api_key_env, "")
    
    def get_model_kwargs(self) -> Dict:
        """Get model-specific keyword arguments"""
        kwargs = {
            "temperature": self.model_config.temperature
        }
        
        if self.model_config.max_tokens:
            kwargs["max_tokens"] = self.model_config.max_tokens
        if self.model_config.top_p:
            kwargs["top_p"] = self.model_config.top_p
        if self.model_config.top_k:
            kwargs["top_k"] = self.model_config.top_k
            
        return kwargs