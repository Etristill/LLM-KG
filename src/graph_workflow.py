# src/graph_workflow.py

# - Manages the step-by-step process of discovering models
# - Controls when to generate, evaluate, and modify models
# - Includes steps for combining promising models
# - Tracks how models relate to each other


from typing import Dict, List, TypedDict, Optional
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import operator
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

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State maintained in the graph workflow."""
    current_model: ModelState
    knowledge: Dict
    messages: List[BaseMessage]
    next_step: str
    metrics: Dict
    thought_history: List[ModelState]  # Track sequence of thoughts
    active_thoughts: List[ModelState]  # Currently active thoughts

class ModelDiscoveryGraph:
    """Manages the discovery process combining MCTS exploration with Graph of Thoughts."""
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph, test_data: Dict):
        # Core components initialization
        self.kg = knowledge_graph
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
        )
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
            good_thoughts = [
                t for t in state['active_thoughts'] 
                if getattr(t, 'score', 0) > self.aggregation_threshold
            ]
            if len(good_thoughts) >= 3:
                state = await self.aggregate_thoughts_node(state, good_thoughts)
            
            # Try thought refinement if promising
            if state['current_model'].score and state['current_model'].score > self.refinement_threshold:
                state = await self.refine_thought_node(state)
            
            # Standard evaluation and updates
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
            
            response = await self.llm.agenerate([messages])
            new_model = self._parse_llm_response(response.generations[0][0].text)
            
            # If we successfully parsed a new model, update current_model
            if new_model:
                # Preserve old score if available
                if state["current_model"].score is not None:
                    new_model.score = state["current_model"].score
                state["current_model"] = new_model
                self._add_to_thought_graph(new_model, "generation")
                
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

    def _create_hypothesis_prompt(self, state: AgentState) -> str:
        """Create detailed prompt for hypothesis generation."""
        current_model = state["current_model"]
        knowledge = state["knowledge"]
        
        # Extract mechanism information
        mechanism_info = ""
        for mech, info in knowledge.items():
            if not mech.endswith('_performance'):
                mechanism_info += f"\n{mech}:\n"
                if 'description' in info:
                    mechanism_info += f"Description: {info['description']}\n"
                if 'base_equations' in info:
                    mechanism_info += f"Base equations: {info['base_equations']}\n"
                if 'parameters' in info:
                    mechanism_info += f"Parameters: {info['parameters']}\n"
        
        return f"""
        Current cognitive model equation(s):
        {current_model.equations[0]}
        
        Current parameters:
        {current_model.parameters}
        
        Known mechanism information:
        {mechanism_info}
        
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
        """

    def _parse_llm_response(self, text: str) -> Optional[ModelState]:
        """
        Parse LLM response into a ModelState.
        - Fixes the 'unexpected keyword argument "metadata"' by NOT passing metadata.
        - Embeds the theoretical_basis into parameters if it exists.
        - Cleans up unquoted keys or special characters in parameters.
        """
        try:
            lines = text.strip().split('\n')
            equation = None
            parameters = {}
            theoretical_basis = None
            
            for line in lines:
                if line.startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                
                elif line.startswith('PARAMETERS:'):
                    params_str = line.replace('PARAMETERS:', '').strip()
                    parameters = self._safe_parse_parameters(params_str)
                
                elif line.startswith('THEORETICAL_BASIS:'):
                    theoretical_basis = line.replace('THEORETICAL_BASIS:', '').strip()
            
            if equation:
                # If there's a theoretical_basis, store it in parameters for reference
                if theoretical_basis:
                    parameters["theoretical_basis"] = theoretical_basis
                
                return ModelState(
                    equations=[equation],
                    parameters=parameters
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None

    def _safe_parse_parameters(self, params_str: str) -> Dict[str, float]:
        """
        Safely parse the PARAMETERS line, handling cases like:
          {learning_rate: 0.2, temperature: 1.5, β: 5.0, Q(ref): 0.0}
        by:
          1. Adding braces if missing.
          2. Quoting unquoted keys (including exotic ones like β or Q(ref)).
          3. Using ast.literal_eval to parse the cleaned string.
          4. Converting values to float if possible.
        """
        candidate = params_str.strip()
        
        # If missing braces, add them
        if not candidate.startswith("{"):
            candidate = "{" + candidate
        if not candidate.endswith("}"):
            candidate += "}"
        
        # Now attempt to quote unquoted keys:
        candidate = self._quote_unquoted_keys(candidate)
        
        # Try to parse
        param_dict = {}
        try:
            param_dict = ast.literal_eval(candidate)
        except Exception as parse_err:
            logger.error(
                f"Error parsing parameter value with ast.literal_eval: {candidate} | {parse_err}"
            )
            param_dict = {}
        
        # Convert each value to float if possible
        parsed_params = {}
        for k, v in param_dict.items():
            try:
                parsed_params[str(k).strip()] = float(v)
            except (ValueError, TypeError):
                logger.error(f"Could not convert param: {k}={v} to float")
        return parsed_params

    def _quote_unquoted_keys(self, text: str) -> str:
        """
        Use a regex to enclose any unquoted key (which can include letters, digits, 
        underscore, parentheses, Greek letters, etc.) in double quotes, 
        so ast.literal_eval or JSON can parse them.
        
        Example:
          {learning_rate: 0.2, temperature: 1.5, β: 5.0, Q(ref): 0.0}
        becomes
          {"learning_rate": 0.2, "temperature": 1.5, "β": 5.0, "Q(ref)": 0.0}
        """
        # Regex explanation:
        # - Look for a group that starts with optional spaces or braces/commas,
        #   then capture a sequence of characters (excluding quotes) up to a colon.
        # - We assume that keys do not contain colons themselves except for the key:value boundary.
        # - We skip if there's already a quote in front of the key.
        #
        # STILL FAILING ;<<<< but I'm close
        
        # Remove any stray whitespace around colons
        text = re.sub(r'\s*:\s*', ': ', text)

        # Now quote the keys if they are not already quoted:
        # pattern: something like  { or , or ^  followed by (not " or ' or space) repeated, up to colon
        # We'll capture the group after the brace or comma, then wrap in quotes.
        # Explanation:
        #   ([{,]\s*) - group 1: a brace or comma plus optional spaces
        #   ([^\s"\'{,]+) - group 2: one or more chars that are NOT whitespace, quote, brace, or comma
        #   (?=\s*:) - a lookahead ensuring there's a colon ahead

        # replacement: \1 "group2"
        # then we add a trailing quote, so effectively: { group -> { "group    -> something like that

        def replacer(match):
            g1 = match.group(1)
            g2 = match.group(2)
            return f'{g1}"{g2}"'

        text_quoted = re.sub(pattern, replacer, text)

        return text_quoted

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

    async def check_convergence_node(self, state: AgentState) -> AgentState:
        """Check convergence with enhanced criteria."""
        try:
            # Check if we have enough history
            if len(state['thought_history']) > 5:
                # Get recent scores
                recent_scores = [
                    t.score for t in state['thought_history'][-5:]
                    if t.score is not None
                ]
                
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
            
            #  volume ==nnumber of predecessor thoughts
            volume = len(nx.ancestors(self.thought_graph, model.id))
            
            #  latency == longest path to this thought
            root_nodes = [
                n for n in self.thought_graph.nodes() 
                if self.thought_graph.in_degree(n) == 0
            ]
            
            max_path_length = 0
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, model.id):
                    paths = list(nx.all_simple_paths(self.thought_graph, root, model.id))
                    if paths:
                        max_path_length = max(max_path_length, max(len(path) for path in paths))
            
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
                # Fallk to simpler centrality
                return nx.degree_centrality(self.thought_graph)[model.id]
                
        except Exception as e:
            logger.error(f"Error computing influence: {e}")
            return 0.0

    def _compute_novelty(self, model: ModelState) -> float:
        """Compute model's novelty compared to existing thoughts."""
        try:
            if not self.thought_graph or self.thought_graph.number_of_nodes() == 0:
                return 1.0
                
            similarities = []
            for node_id in self.thought_graph.nodes():
                if node_id != model.id:
                    node_data = self.thought_graph.nodes[node_id]
                    if 'state' in node_data:
                        eq1 = model.equations[0] if model.equations else ""
                        eq2 = node_data['state'].equations[0] if node_data['state'].equations else ""
                        similarity = self._compute_equation_similarity(eq1, eq2)
                        similarities.append(similarity)
                        
            return 1.0 - (np.mean(similarities) if similarities else 0.0)
            
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

    def get_current_performance_metrics(self, state: AgentState) -> Dict:
        """Get current performance metrics for monitoring."""
        try:
            metrics = {
                'current_score': state['current_model'].score,
                'active_thoughts': len(state['active_thoughts']),
                'thought_history': len(state['thought_history']),
                'unique_mechanisms': len(set(
                    mech for thought in state['thought_history']
                    for mech in self._extract_mechanisms(thought)
                ))
            }
            
            # Adding thought graph metrics if available
            if self.thought_graph:
                metrics.update({
                    'graph_nodes': self.thought_graph.number_of_nodes(),
                    'graph_edges': self.thought_graph.number_of_edges(),
                    'avg_node_degree': np.mean([
                        d for _, d in self.thought_graph.degree()
                    ]) if self.thought_graph.number_of_nodes() > 0 else 0.0
                })
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
