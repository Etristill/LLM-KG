from typing import List, Dict, Any, Optional
import anthropic
import openai
from .config import Config

class UnifiedLLMClient:
    """Unified client for interacting with different LLM providers"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        self.config = Config(model_name)
        
        if self.config.is_anthropic:
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        else:
            self.client = openai.OpenAI(api_key=self.config.api_key)
    
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
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        # Convert to Anthropic format
        anthropic_messages = [{
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        } for msg in user_messages]

        # Use create_sync for synchronous call since Anthropic client doesn't support async
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=1000,
            system=system_message,
            messages=anthropic_messages
        )
        
        return response.content[0].text
    
    async def _generate_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI models"""
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