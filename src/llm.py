# src/llm.py:

# - Handles communication with the language model
# - Formats prompts and parses responses

import os
from typing import Dict, List, Optional, Set
from anthropic import Anthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
from .config import Config

load_dotenv()

class EnhancedLLMInterface:
    """Enhanced LLM interface with sophisticated knowledge graph integration"""
    def __init__(self, model_name: str = "claude-3-opus-20240229", use_mock: bool = False):
        self.use_mock = use_mock
        self.kg = CognitiveKnowledgeGraph()
        
        if not use_mock:
            # Initialize configuration
            self.config = Config(model_name)
            
            # Initialize appropriate client based on provider
            if self.config.is_anthropic:
                self.client = Anthropic(api_key=self.config.api_key)
            else:
                self.client = AsyncOpenAI(api_key=self.config.api_key)
        
        # Cache for frequent KG queries
        self.mechanism_cache: Dict[str, Dict] = {}
        
        # Track successful generations
        self.successful_patterns: Set[str] = set()
    
    async def generate(self, state: ModelState, context: Optional[Dict] = None) -> ModelState:
        """Generate new model variations using LLM and knowledge graph"""
        if self.use_mock:
            return self._mock_generate(state)
        else:
            return await self._llm_generate(state, context)

    async def _llm_generate(self, state: ModelState, context: Optional[Dict]) -> ModelState:
        """Use LLM API with enhanced knowledge graph context"""
        try:
            # Extract mechanisms from current state
            mechanisms = self._extract_mechanisms(state)
            
            # Build comprehensive knowledge context
            knowledge_context = await self._build_knowledge_context(state, mechanisms)
            
            # Create messages
            system_content = self._create_system_prompt(knowledge_context)
            user_content = self._create_detailed_prompt(state, knowledge_context)
            
            if self.config.is_anthropic:
                response = await self.client.messages.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    **self.config.get_model_kwargs()
                )
                response_content = response.content[0].text
            else:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    **self.config.get_model_kwargs()
                )
                response_content = response.choices[0].message.content
            
            new_state = self._parse_llm_response(response_content, state)
            
            # Update successful patterns if generation was valid
            if new_state != state:
                self._update_successful_patterns(new_state)
            
            return new_state
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._mock_generate(state)

    def _create_system_prompt(self, knowledge_context: Dict) -> str:
        """Create enhanced system prompt using knowledge context"""
        constraints = knowledge_context.get('constraints', [])
        best_practices = knowledge_context.get('best_practices', [])
        
        return f"""You are an expert in cognitive modeling and reinforcement learning, 
        specializing in developing theoretically sound and mathematically valid models.
        
        Key constraints to consider:
        {self._format_list(constraints)}
        
        Best practices to follow:
        {self._format_list(best_practices)}
        
        Focus on generating models that:
        1. Are mathematically precise and implementable
        2. Follow established theoretical principles
        3. Use parameters within valid ranges
        4. Can be empirically tested
        5. Build upon known successful patterns"""

    def _create_detailed_prompt(self, state: ModelState, knowledge_context: Dict) -> str:
        """Create detailed prompt with comprehensive knowledge integration"""
        mechanisms = knowledge_context.get('mechanisms', [])
        examples = knowledge_context.get('examples', [])
        relationships = knowledge_context.get('relationships', [])
        successful_patterns = list(self.successful_patterns)[:3]
        
        return f"""
        Current cognitive model equation(s):
        {state.equations[0]}
        
        Current parameters:
        {self._format_parameters(state.parameters)}
        
        Known mechanisms:
        {self._format_list(mechanisms)}
        
        Related examples:
        {self._format_list(examples)}
        
        Mechanism relationships:
        {self._format_list(relationships)}
        
        Previously successful patterns:
        {self._format_list(successful_patterns)}
        
        Generate a variation of this model that:
        1. Incorporates these known cognitive mechanisms
        2. Uses validated parameter ranges
        3. Maintains mathematical precision
        4. Builds on successful patterns
        5. Could explain human learning in a two-armed bandit task
        
        YOU MUST FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        EQUATION: [your equation]
        PARAMETERS: [parameter1: value1, parameter2: value2, ...]
        THEORETICAL_BASIS: [brief explanation of theoretical justification]
        MECHANISM_MAPPING: [how each mechanism is represented]
        """

    async def _build_knowledge_context(self, state: ModelState, mechanisms: List[str]) -> Dict:
        """Build comprehensive knowledge context from KG"""
        context = {
            'mechanisms': [],
            'constraints': [],
            'examples': [],
            'relationships': [],
            'best_practices': []
        }
        
        for mechanism in mechanisms:
            # Use cached information if available
            if mechanism not in self.mechanism_cache:
                info = self.kg.query_mechanism(mechanism)
                related = self.kg.get_related_mechanisms(mechanism)
                self.mechanism_cache[mechanism] = {
                    'info': info,
                    'related': related
                }
            
            cached_data = self.mechanism_cache[mechanism]
            
            # Add mechanism information
            if cached_data['info']:
                context['mechanisms'].append(f"{mechanism}: {cached_data['info'].get('description', '')}")
                context['constraints'].extend(cached_data['info'].get('constraints', []))
                context['examples'].extend(cached_data['info'].get('base_equations', []))
            
            # Add relationship information
            if cached_data['related']:
                for rel in cached_data['related']:
                    context['relationships'].append(
                        f"{mechanism} {rel.get('relationship', 'relates to')} {rel.get('mechanism', '')}"
                    )
        
        return context

    def _parse_llm_response(self, response_content: str, original_state: ModelState) -> ModelState:
        """Parse LLM response with enhanced validation"""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            equation = None
            parameters = {}
            theoretical_basis = None
            mechanism_mapping = None
            
            # Parse response sections
            current_section = None
            section_content = []
            
            for line in lines:
                if line.startswith('EQUATION:'):
                    if section_content and current_section == 'PARAMETERS':
                        parameters = self._parse_parameters(''.join(section_content))
                    current_section = 'EQUATION'
                    equation = line.replace('EQUATION:', '').strip()
                    section_content = []
                elif line.startswith('PARAMETERS:'):
                    current_section = 'PARAMETERS'
                    section_content = []
                elif line.startswith('THEORETICAL_BASIS:'):
                    if section_content and current_section == 'PARAMETERS':
                        parameters = self._parse_parameters(''.join(section_content))
                    current_section = 'THEORETICAL_BASIS'
                    theoretical_basis = line.replace('THEORETICAL_BASIS:', '').strip()
                    section_content = []
                elif line.startswith('MECHANISM_MAPPING:'):
                    current_section = 'MECHANISM_MAPPING'
                    mechanism_mapping = line.replace('MECHANISM_MAPPING:', '').strip()
                    section_content = []
                else:
                    section_content.append(line)
            
            # Handle last section
            if section_content and current_section == 'PARAMETERS':
                parameters = self._parse_parameters(''.join(section_content))
            
            if equation and parameters:
                new_state = ModelState(
                    equations=[equation],
                    parameters=parameters
                )
                if self.validate_equation(new_state):
                    return new_state
            
            print("Failed to generate valid model")
            return original_state.copy()
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return original_state.copy()

    def validate_equation(self, state: ModelState) -> bool:
        """Enhanced validation of generated equations"""
        equation = state.equations[0]
        try:
            # Structural validation
            if equation.count('(') != equation.count(')'):
                return False
            
            # Required components check
            required_components = ['Q(t', 'R(t)']
            if not all(comp in equation for comp in required_components):
                return False
            
            # Parameter validation
            equation_params = set()
            for param in ['alpha', 'beta', 'gamma', 'temp']:
                if param in equation:
                    equation_params.add(param)
            
            if not equation_params.issubset(set(state.parameters.keys())):
                return False
            
            # Parameter range validation
            param_ranges = {
                'alpha': (0, 1),
                'gamma': (0, 1),
                'temp': (0, float('inf')),
                'beta': (0, 2)
            }
            
            for param, value in state.parameters.items():
                if param in param_ranges:
                    min_val, max_val = param_ranges[param]
                    if value < min_val or value > max_val:
                        return False
            
            # Theoretical validation using KG
            mechanisms = self._extract_mechanisms(state)
            for mechanism in mechanisms:
                info = self.mechanism_cache.get(mechanism, {}).get('info', {})
                if info and 'constraints' in info:
                    for constraint in info['constraints']:
                        if not self._check_constraint(state, constraint):
                            return False
            
            return True
            
        except Exception as e:
            print(f"Error in equation validation: {e}")
            return False

    def _mock_generate(self, state: ModelState) -> ModelState:
        """Enhanced mock generation with theoretical components"""
        new_state = state.copy()
        
        # Enhanced modifications with theoretical justification
        modifications = [
            (" + alpha * (R(t) - Q(t))", {"alpha": 0.1}, "Standard RL update"),
            (" * (1 - gamma)", {"gamma": 0.1}, "Memory decay"),
            (" / temp", {"temp": 1.0}, "Action selection"),
            (" + beta * abs(R(t) - Q(t))", {"beta": 0.2}, "Surprise modulation"),
            (" * (1 + alpha * PE)", {"alpha": 0.1}, "Dynamic learning rate"),
        ]
        
        # Select modification based on current equation complexity
        current_terms = len(new_state.equations[0].split('+'))
        if current_terms < 3:
            # Prefer simpler modifications for simple equations
            valid_mods = [m for m in modifications if len(m[0].split('+')) <= 2]
        else:
            # Prefer more complex modifications for complex equations
            valid_mods = [m for m in modifications if len(m[0].split('+')) > 1]
        
        if valid_mods:
            mod, params, _ = np.random.choice(valid_mods)
            new_state.equations[0] += mod
            new_state.parameters.update(params)
        
        return new_state

    def _extract_mechanisms(self, state: ModelState) -> List[str]:
        """Extract cognitive mechanisms from model state"""
        mechanisms = []
        equation = state.equations[0].lower()
        
        # Mechanism detection patterns
        patterns = {
            'reinforcement_learning': ['q(t)', 'r(t)', 'pe'],
            'working_memory': ['wm', 'memory', 'gamma'],
            'prediction_error': ['pe', 'r(t)-q(t)'],
            'exploration': ['temp', 'beta'],
            'surprise': ['abs', 'surprise', 'unexpected']
        }
        
        for mechanism, keywords in patterns.items():
            if any(keyword in equation for keyword in keywords):
                mechanisms.append(mechanism)
        
        return mechanisms

    def _format_parameters(self, parameters: Dict) -> str:
        """Format parameters with descriptions"""
        param_descriptions = {
            'alpha': 'learning rate',
            'gamma': 'discount factor',
            'temp': 'temperature',
            'beta': 'exploration factor'
        }
        
        formatted = []
        for param, value in parameters.items():
            desc = param_descriptions.get(param, '')
            formatted.append(f"{param}: {value} ({desc})")
        
        return '\n'.join(formatted)

    def _format_list(self, items: List) -> str:
        """Format list items with bullet points"""
        if not items:
            return "None available"
        return '\n'.join(f"• {item}" for item in items)

    def _parse_parameters(self, param_str: str) -> Dict[str, float]:
        """Parse parameters with enhanced error handling"""
        parameters = {}
        try:
            pairs = param_str.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':')
                    key = key.strip()
                    try:
                        value = float(value.split('(')[0].strip())
                        parameters[key] = value
                    except ValueError:
                        print(f"Error parsing parameter value: {pair}")
        except Exception as e:
            print(f"Error parsing parameters: {e}")
        return parameters

    def _check_constraint(self, state: ModelState, constraint: str) -> bool:
        """Check if state satisfies a theoretical constraint"""
        try:
            # Basic constraint checking - can be expanded
            if "positive_learning_rate" in constraint:
                return state.parameters.get('alpha', 0) > 0
            if "bounded_parameters" in constraint:
                return all(0 <= v <= 1 for v in state.parameters.values())
            return True
        except Exception:
            return True

    def _update_successful_patterns(self, state: ModelState):
        """Update tracking of successful generation patterns"""
        equation = state.equations[0]
        for pattern in ['Q(t)', 'R(t)', 'alpha', 'temp']:
            if pattern in equation:
                self.successful_patterns.add(pattern)