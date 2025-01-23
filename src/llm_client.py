# src/llm_client.py

from typing import List, Dict, Any, Optional
import anthropic
import openai
from .config import Config

class UnifiedLLMClient:
    """Unified client for interacting with different LLM providers"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        self.config = Config(model_name)
        
        if self.config.is_anthropic:
            self.client = anthropic.Client(api_key=self.config.api_key)
        else:
            self.client = openai.Client(api_key=self.config.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the model"""
        try:
            if self.config.is_anthropic:
                return await self._generate_anthropic(messages, **kwargs)
            else:
                return await self._generate_openai(messages, **kwargs)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    async def _generate_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic's Claude"""
        # Convert messages to Anthropic format!!!
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                formatted_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": msg["content"]})
            elif role == "system":
                # !!!!Anthropic handles system messages differently
                formatted_messages.insert(0, {"role": "user", "content": msg["content"]})
        
        model_kwargs = self.config.get_model_kwargs()
        model_kwargs.update(kwargs)
        
        response = await self.client.messages.create(
            model=self.config.model_name,
            messages=formatted_messages,
            **model_kwargs
        )
        
        return response.content[0].text
    
    async def _generate_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI models"""
        # OpenAI already accepts messages in the correct format
        model_kwargs = self.config.get_model_kwargs()
        model_kwargs.update(kwargs)
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **model_kwargs
        )
        
        return response.choices[0].message.content
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "name": self.config.model_name,
            "provider": self.config.model_config.provider,
            "temperature": self.config.model_config.temperature,
            "max_tokens": self.config.model_config.max_tokens,
        }