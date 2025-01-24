# src/graph_workflow.py

from typing import Dict, List, TypedDict, Optional
import logging
import networkx as nx
import numpy as np
import ast
import re
from dataclasses import dataclass

from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
from .transformations import ThoughtTransformations
from .evaluation import SimpleEvaluator
from .llm_client import UnifiedLLMClient  

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State maintained in the graph workflow."""
    current_model: ModelState
    knowledge: Dict
    messages: List[Dict[str, str]]  # Changed from BaseMessage to Dict
    next_step: str
    metrics: Dict
    thought_history: List[ModelState]  # Track sequence of thoughts
    active_thoughts: List[ModelState]  # Currently active thoughts

class ModelDiscoveryGraph:
    """Manages the discovery process combining MCTS exploration with Graph of Thoughts."""
    def __init__(
        self, 
        knowledge_graph: CognitiveKnowledgeGraph, 
        test_data: Dict, 
        model_name: str = "claude-3-opus-20240229"
    ):
        # Core components initialization
        self.kg = knowledge_graph
        
        # Initialize the unified LLM client
        self.llm = UnifiedLLMClient(model_name=model_name)
        
        # Pass the same LLM client to transformations
        self.transformations = ThoughtTransformations(self.llm)
        self.evaluator = SimpleEvaluator(test_data)
        
        # Graph structures
        self.workflow_graph = nx.DiGraph()  # For workflow steps
        self.thought_graph = nx.DiGraph()   # For thought relationships
        
        # Configuration
        self.max_active_thoughts = 10
        self.aggregation_threshold = 0.2
        self.refinement_threshold = 0.05
        
        # Performance tracking
        self.performance_cache = {}

        logger.info(f"Initialized ModelDiscoveryGraph with model: {model_name}")

    async def run_workflow(self, state: AgentState) -> AgentState:
        """Execute the complete workflow with enhanced error handling."""
        try:
            # Initialize thought tracking if first run
            if 'thought_history' not in state:
                state['thought_history'] = []
            if 'active_thoughts' not in state:
                state['active_thoughts'] = []

            # Execute workflow steps sequentially
            state = await self.query_knowledge_node(state)
            state = await self.generate_hypothesis_node(state)
            
            # Try thought aggregation if enough good thoughts
            good_thoughts = []
            for t in state['active_thoughts']:
                if hasattr(t, 'score') and t.score is not None and t.score > self.aggregation_threshold:
                    good_thoughts.append(t)
                    
            if len(good_thoughts) >= 3:
                state = await self.aggregate_thoughts_node(state, good_thoughts)
            
            # Try thought refinement if promising
            current_score = getattr(state['current_model'], 'score', None)
            if current_score is not None and current_score > self.refinement_threshold:
                state = await self.refine_thought_node(state)
            
            # Standard evaluation and updates
            # -- Call evaluate_model_node (it calls the evaluator synchronously now)
            state = await self.evaluate_model_node(state)
            state = await self.update_knowledge_node(state)
            
            # Update thought tracking and check convergence
            self._update_thought_tracking(state)
            state = await self.check_convergence_node(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            return state

    async def evaluate_model_node(self, state: AgentState) -> AgentState:
        """Evaluate model with comprehensive metrics."""
        try:
            # Make sure we have a valid model state
            if not state["current_model"]:
                logger.error("No current model to evaluate")
                return state
                
            # >>>>>>>>>> IMPORTANT CHANGE! <<<<<<<<<<
            # Call synchronously. Do *not* await this.
            score = self.evaluator.evaluate_model(state["current_model"])
            
            # Assign score to the model state
            state["current_model"].score = score
            
            # Cache performance
            model_id = state["current_model"].id
            if model_id in self.thought_graph.nodes:
                creation_time = self.thought_graph.nodes[model_id].get('creation_time', 0)
            else:
                creation_time = 0

            self.performance_cache[model_id] = {
                'score': score,
                'creation_time': creation_time
            }
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluate_model_node: {e}")
            return state

    async def query_knowledge_node(self, state: AgentState) -> AgentState:
        """Query knowledge graph with enhanced caching."""
        try:
            mechanisms = self._extract_mechanisms(state["current_model"])
            knowledge = {}
            
            for mechanism in mechanisms:
                # Get mechanism knowledge
                mech_info = self.kg.query_mechanism(mechanism)
                if mech_info:
                    knowledge[mechanism] = mech_info
                    
                # Get performance data
                perf_info = self.kg.get_mechanism_performance(mechanism)
                if perf_info:
                    knowledge[f"{mechanism}_performance"] = perf_info
            
            state["knowledge"] = knowledge
            return state
            
        except Exception as e:
            logger.error(f"Error in query_knowledge_node: {e}")
            return state

    async def generate_hypothesis_node(self, state: AgentState) -> AgentState:
        """Generate new hypotheses with theoretical guidance."""
        try:
            system_message = (
                "You are an expert in cognitive modeling. "
                "Generate new hypotheses based on known mechanisms and constraints."
            )
            
            prompt = self._create_hypothesis_prompt(state)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Use unified client (async call to LLM)
            response_text = await self.llm.generate(messages)
            
            logger.debug(f"LLM Response:\n{response_text}")
            
            new_model = self._parse_llm_response(response_text)
            
            # If we successfully parsed a new model, update current_model
            if new_model:
                # Preserve old score if available
                if state["current_model"].score is not None:
                    new_model.score = state["current_model"].score
                state["current_model"] = new_model
                self._add_to_thought_graph(new_model, "generation")
            else:
                # If parsing failed, keep the current model
                logger.warning("Failed to parse LLM response, keeping current model")
                new_model = state["current_model"].copy()
                
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_hypothesis_node: {e}")
            return state

    async def aggregate_thoughts_node(self, state: AgentState, thoughts: List[ModelState]) -> AgentState:
        """Node for combining multiple thoughts."""
        try:
            aggregated_model = await self.transformations.aggregate_thoughts(thoughts)
            if aggregated_model:
                state["current_model"] = aggregated_model
                self._add_to_thought_graph(
                    aggregated_model, 
                    "aggregation", 
                    parent_thoughts=thoughts
                )
                
                # Update metrics
                if 'aggregation_counts' not in state['metrics']:
                    state['metrics']['aggregation_counts'] = []
                state['metrics']['aggregation_counts'].append(len(thoughts))
            
            return state
            
        except Exception as e:
            logger.error(f"Error in aggregate_thoughts_node: {e}")
            return state

    async def refine_thought_node(self, state: AgentState) -> AgentState:
        """Node for refining thoughts."""
        try:
            old_model = state["current_model"]
            refined_model = await self.transformations.refine_thought(old_model)
            if refined_model:
                state["current_model"] = refined_model
                self._add_to_thought_graph(
                    refined_model, 
                    "refinement",
                    parent_thoughts=[old_model]
                )
                
                # Update metrics
                if 'refinement_counts' not in state['metrics']:
                    state['metrics']['refinement_counts'] = []
                state['metrics']['refinement_counts'].append(1)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in refine_thought_node: {e}")
            return state

    async def update_knowledge_node(self, state: AgentState) -> AgentState:
        """Update knowledge graph with enhanced model tracking."""
        try:
            if state["current_model"].score is not None:
                mechanisms = self._extract_mechanisms(state["current_model"])
                for mechanism in mechanisms:
                    self.kg.add_model_knowledge(mechanism, state["current_model"])
                    logger.info(f"Added model to KG for mechanism: {mechanism}")
                    
                # Update performance cache
                self.performance_cache[state["current_model"].id] = {
                    'score': state["current_model"].score,
                    'mechanisms': mechanisms
                }
                
            return state
            
        except Exception as e:
            logger.error(f"Error in update_knowledge_node: {e}")
            return state

    async def check_convergence_node(self, state: AgentState) -> AgentState:
        """Check convergence with enhanced criteria."""
        try:
            # Check if we have enough history
            if len(state['thought_history']) > 5:
                # Get recent scores
                recent_scores = []
                for t in state['thought_history'][-5:]:
                    if hasattr(t, 'score') and t.score is not None:
                        recent_scores.append(t.score)
                
                # Check for score convergence
                if recent_scores and (max(recent_scores) - min(recent_scores) < 0.01):
                    # Check for mechanism diversity
                    recent_mechanisms = set()
                    for thought in state['thought_history'][-5:]:
                        mechanisms = self._extract_mechanisms(thought)
                        recent_mechanisms.update(mechanisms)
                    
                    # If we've explored multiple mechanisms and scores converged
                    if len(recent_mechanisms) >= 2:
                        state['next_step'] = 'complete'
                        logger.info("Convergence detected")
                    
            return state
            
        except Exception as e:
            logger.error(f"Error checking convergence: {e}")
            return state

    def _create_hypothesis_prompt(self, state: AgentState) -> str:
        """Create prompt encouraging creative exploration."""
        current_model = state["current_model"]
        
        return f"""Explore creative mathematical models for learning and decision-making.

Current model:
{current_model.equations[0]}

Current parameters:
{current_model.parameters}

You can:
- Combine multiple learning mechanisms
- Add new parameters with any values
- Try non-linear interactions
- Experiment with temporal dependencies
- Add memory effects
- Consider additional mechanisms
- Explore adaptation and meta-learning
KEY FORMAT REQUIREMENTS:
- All parameters must be plain numeric values. For example, do not write "2 * pi / 25", just approximate it to 0.2513274.
Use PLAIN text notation (not LaTeX)!!!!!!!!!
   - Write 'alpha' not '\\alpha'
   - Use simple functions like 'exp(x)' not 'e^x'
   - No \\[ or \\] or \\frac notation

Just ensure your equation uses Q(t) and R(t) terms.

Response format:
EQUATION: [your equation]
PARAMETERS: [your parameters]
THEORETICAL_BASIS: [your idea]

Be creative and explore interesting mathematical structures!"""

    def _parse_llm_response(self, text: str) -> Optional[ModelState]:
        """Parse LLM response into a ModelState."""
        try:
            logger.info(f"Raw LLM response:\n{text}")
            
            lines = text.strip().split('\n')
            equation = None
            parameters = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse equation
                if line.startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                    logger.info(f"Found equation: {equation}")
                    
                # Parse parameters
                elif line.startswith('PARAMETERS:'):
                    param_text = line.replace('PARAMETERS:', '').strip()
                    try:
                        # Try to evaluate as literal Python dict
                        if param_text.startswith('{') and param_text.endswith('}'):
                            parameters = ast.literal_eval(param_text)
                        else:
                            # If not in dict format, try to parse key-value pairs
                            pairs = param_text.split(',')
                            for pair in pairs:
                                if ':' in pair:
                                    key, value = pair.split(':')
                                    key = key.strip()
                                    try:
                                        value = float(value.strip())
                                        parameters[key] = value
                                    except ValueError:
                                        continue
                    except Exception as e:
                        logger.error(f"Error parsing parameters: {e}")
                        logger.info(f"Problem parameter text: {param_text}")
            
            # Add debug logs
            logger.info(f"Parsed equation: {equation}")
            logger.info(f"Parsed parameters: {parameters}")
            
            if equation and parameters:
                new_state = ModelState(
                    equations=[equation],
                    parameters=parameters
                )
                if self.validate_equation(new_state):
                    return new_state
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.info(f"Problematic text:\n{text}")
            return None

    def validate_equation(self, state: ModelState) -> bool:
        """Minimal validation to allow creative exploration."""
        try:
            equation = state.equations[0]
            
            # Only check that we have Q(t) and R(t) and balanced parentheses
            basic_terms_present = 'Q(t' in equation and 'R(t' in equation
            balanced_parens = equation.count('(') == equation.count(')')
            
            return basic_terms_present and balanced_parens
            
        except Exception as e:
            logger.error(f"Error validating equation: {e}")
            return False
        
    def _extract_mechanisms(self, state: ModelState) -> List[str]:
        """Extract cognitive mechanisms from model state."""
        try:
            mechanisms = []
            if not state.equations:
                return mechanisms
            
            equation = state.equations[0].lower()
            mechanism_patterns = {
                "reinforcement_learning": ["q(t)", "r(t)", "reward"],
                "working_memory": ["wm", "memory", "gamma"],
                "prediction_error": ["pe", "error", "r(t)-q(t)"]
            }
            for mechanism, patterns in mechanism_patterns.items():
                if any(pattern in equation for pattern in patterns):
                    mechanisms.append(mechanism)
                    
            return mechanisms
            
        except Exception as e:
            logger.error(f"Error extracting mechanisms: {e}")
            return []

    def _update_thought_tracking(self, state: AgentState):
        """Update thought tracking with performance data."""
        try:
            # Add to history
            state['thought_history'].append(state['current_model'])
            
            # Update active thoughts
            state['active_thoughts'].append(state['current_model'])
            if len(state['active_thoughts']) > self.max_active_thoughts:
                # Keep best thoughts based on score + novelty
                state['active_thoughts'].sort(
                    key=lambda x: (
                        x.score if x.score is not None else float('-inf')
                    ) + self._compute_novelty(x),
                    reverse=True
                )
                state['active_thoughts'] = state['active_thoughts'][:self.max_active_thoughts]
                
        except Exception as e:
            logger.error(f"Error updating thought tracking: {e}")

    def _add_to_thought_graph(self, model: ModelState, operation_type: str, parent_thoughts: List[ModelState] = None):
        """Add thought to graph with enhanced metadata."""
        try:
            if not self.thought_graph.has_node(model.id):
                # Add node with comprehensive metadata
                self.thought_graph.add_node(
                    model.id,
                    state=model,
                    creation_time=len(self.thought_graph),
                    operation_type=operation_type,
                    score=model.score,
                    mechanisms=self._extract_mechanisms(model)
                )
                
            # Add edges from parent thoughts
            if parent_thoughts:
                for parent in parent_thoughts:
                    if self.thought_graph.has_node(parent.id):
                        self.thought_graph.add_edge(
                            parent.id,
                            model.id,
                            operation=operation_type
                        )
            
            logger.debug(f"Added thought {model.id} to graph with operation {operation_type}")
                
        except Exception as e:
            logger.error(f"Error adding to thought graph: {e}")

    def compute_thought_metrics(self, model: ModelState) -> Dict:
        """Compute comprehensive thought metrics."""
        try:
            if not self.thought_graph.has_node(model.id):
                return {"volume": 0, "latency": 0, "influence": 0, "novelty": 0}
            
            # Volume = number of predecessor thoughts
            volume = len(nx.ancestors(self.thought_graph, model.id))
            
            # Latency = longest path to this thought
            root_nodes = [
                n for n in self.thought_graph.nodes() 
                if self.thought_graph.in_degree(n) == 0
            ]
            
            max_path_length = 0
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, model.id):
                    paths = list(nx.all_simple_paths(self.thought_graph, root, model.id))
                    if paths:
                        max_path_length = max(
                            max_path_length, 
                            max(len(path) for path in paths)
                        )
            
            metrics = {
                "volume": volume,
                "latency": max_path_length,
                "influence": self._compute_influence(model),
                "novelty": self._compute_novelty(model)
            }
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing thought metrics: {e}")
            return {"volume": 0, "latency": 0, "influence": 0, "novelty": 0}

    def _compute_influence(self, model: ModelState) -> float:
        """Compute model's influence in thought graph."""
        try:
            if not self.thought_graph.has_node(model.id):
                return 0.0
            
            try:
                pagerank = nx.pagerank(self.thought_graph, alpha=0.85)
                return pagerank[model.id]
            except ImportError:
                # Fallback to simpler centrality
                return nx.degree_centrality(self.thought_graph)[model.id]
                
        except Exception as e:
            logger.error(f"Error computing influence: {e}")
            return 0.0

    def _compute_novelty(self, model: ModelState) -> float:
        """Compute model's novelty compared to existing thoughts."""
        try:
            if not model.equations:
                return 0.0
                
            equation = model.equations[0]
            similarities = []
            
            for node_id in self.thought_graph.nodes():
                node_data = self.thought_graph.nodes[node_id].get('state')
                if node_data and node_data.equations:
                    other_eq = node_data.equations[0]
                    similarity = self._compute_equation_similarity(equation, other_eq)
                    similarities.append(similarity)
            
            if not similarities:
                return 1.0
            
            return 1.0 - np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Error computing novelty: {e}")
            return 0.0

    def _compute_equation_similarity(self, eq1: str, eq2: str) -> float:
        """Compute similarity between equations."""
        try:
            terms1 = set(eq1.split())
            terms2 = set(eq2.split())
            if terms1 and terms2:
                return len(terms1 & terms2) / len(terms1 | terms2)
            return 0.0
        except Exception as e:
            logger.error(f"Error computing equation similarity: {e}")
            return 0.0
