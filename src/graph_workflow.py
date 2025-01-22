# src/graph_workflow.py

# - Manages the step-by-step process of discovering models
# - Controls when to generate, evaluate, and modify models
# - Includes steps for combining promising models
# - Tracks how models relate to each other


from typing import Dict, List, TypedDict, Annotated, Optional
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import operator
from enum import Enum
import logging
import networkx as nx
from dataclasses import dataclass
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
from .transformations import ThoughtTransformations

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State maintained in the graph"""
    current_model: ModelState
    knowledge: Dict
    messages: List[BaseMessage]
    next_step: str
    metrics: Dict
    thought_history: List[ModelState]  # Track sequence of thoughts
    active_thoughts: List[ModelState]  # Currently active thoughts

class ModelDiscoveryGraph:
    """Manages the discovery process combining MCTS exploration with Graph of Thoughts"""
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph):
        # Core components initialization
        self.kg = knowledge_graph
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
        )
        self.transformations = ThoughtTransformations(self.llm)
        
        # Graph structures
        self.workflow_graph = nx.DiGraph()  # For workflow steps
        self.thought_graph = nx.DiGraph()   # For thought relationships
        
        # Configuration
        self.max_active_thoughts = 10
        self.aggregation_threshold = 0.2
        self.refinement_threshold = 0.05 #forcing more changes, idk?
        
        # Initialize metrics structure
        self.default_metrics = {
            'scores': [],
            'model_complexity': [],
            'iterations': [],
            'exploration_paths': [],
            'thought_volumes': [],    # GoT metric
            'thought_latencies': [],  # GoT metric
            'aggregation_counts': [], # Track thought combinations
            'refinement_counts': []   # Track thought improvements
        }

    async def run_workflow(self, state: AgentState) -> AgentState:
        """Execute the complete workflow with GoT enhancements"""
        try:
            # Initialize thought tracking if first run
            if 'thought_history' not in state:
                state['thought_history'] = []
            if 'active_thoughts' not in state:
                state['active_thoughts'] = []

            # Execute workflow steps sequentially
            state = await self.query_knowledge_node(state)
            state = await self.generate_hypothesis_node(state)
            
            # GoT: Try thought aggregation if enough good thoughts
            good_thoughts = [t for t in state['active_thoughts'] 
                           if getattr(t, 'score', 0) > self.aggregation_threshold]
            if len(good_thoughts) >= 3:
                state = await self.aggregate_thoughts_node(state, good_thoughts)
            
            # GoT: Try thought refinement if promising
            if state['current_model'].score and state['current_model'].score > self.refinement_threshold:
                state = await self.refine_thought_node(state)
            
            # Standard evaluation and updates
            state = await self.evaluate_model_node(state)
            state = await self.update_knowledge_node(state)
            state = await self.check_convergence_node(state)
            
            # Update thought tracking
            self._update_thought_tracking(state)
            
            # Check if we should continue or end
            next_step = self.decide_next_step(state)
            if next_step == "complete":
                state = await self.end_workflow_node(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            return state

    def compute_thought_metrics(self, model: ModelState) -> Dict:
        """Compute metrics for a thought, including volume and latency"""
        try:
            if not self.thought_graph.has_node(model.id):
                return {"volume": 0, "latency": 0}
            
            # Calculate volume (number of predecessor thoughts)
            volume = len(nx.ancestors(self.thought_graph, model.id))
            
            # Calculate latency (longest path to this thought)
            root_nodes = [n for n in self.thought_graph.nodes() 
                         if self.thought_graph.in_degree(n) == 0]
            
            max_path_length = 0
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, model.id):
                    paths = list(nx.all_simple_paths(self.thought_graph, root, model.id))
                    if paths:
                        max_path_length = max(max_path_length, max(len(path) for path in paths))
            
            return {
                "volume": volume,
                "latency": max_path_length
            }
            
        except Exception as e:
            logger.error(f"Error computing thought metrics: {e}")
            return {"volume": 0, "latency": 0}

    async def query_knowledge_node(self, state: AgentState) -> AgentState:
        """Node for querying knowledge graph"""
        try:
            current_mechanism = self._extract_mechanism(state["current_model"])
            knowledge = self.kg.query_mechanism(current_mechanism)
            state["knowledge"] = knowledge
            
            # Add to thought graph
            self._add_to_thought_graph(state["current_model"], "knowledge_query")
        except Exception as e:
            logger.error(f"Error in query_knowledge_node: {e}")
        return state
    
    async def generate_hypothesis_node(self, state: AgentState) -> AgentState:
        """Node for generating new model hypotheses"""
        try:
            system_message = "You are an expert in cognitive modeling. Generate new hypotheses based on known mechanisms."
            prompt = self._create_hypothesis_prompt(state)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm.agenerate([messages])
            new_model = self._parse_llm_response(response.generations[0][0].text)
            if new_model:
                state["current_model"] = new_model
                self._add_to_thought_graph(new_model, "generation")
        except Exception as e:
            logger.error(f"Error in generate_hypothesis_node: {e}")
        return state

    async def aggregate_thoughts_node(self, state: AgentState, thoughts: List[ModelState]) -> AgentState:
        """Node for combining multiple thoughts"""
        try:
            aggregated_model = await self.transformations.aggregate_thoughts(thoughts)
            if aggregated_model:
                state["current_model"] = aggregated_model
                self._add_to_thought_graph(aggregated_model, "aggregation", parent_thoughts=thoughts)
                state["metrics"]["aggregation_counts"].append(len(thoughts))
        except Exception as e:
            logger.error(f"Error in aggregate_thoughts_node: {e}")
        return state

    async def refine_thought_node(self, state: AgentState) -> AgentState:
        """Node for refining existing thoughts"""
        try:
            refined_model = await self.transformations.refine_thought(state["current_model"])
            if refined_model:
                state["current_model"] = refined_model
                self._add_to_thought_graph(refined_model, "refinement", 
                                         parent_thoughts=[state["current_model"]])
                state["metrics"]["refinement_counts"].append(1)
        except Exception as e:
            logger.error(f"Error in refine_thought_node: {e}")
        return state

    async def evaluate_model_node(self, state: AgentState) -> AgentState:
        """Node for evaluating models"""
        try:
            score = self._compute_model_score(state["current_model"], state["knowledge"])
            state["current_model"].score = score
        except Exception as e:
            logger.error(f"Error in evaluate_model_node: {e}")
        return state
    
    async def update_knowledge_node(self, state: AgentState) -> AgentState:
        """Node for updating knowledge graph"""
        try:
            if state["current_model"].score:
                self.kg.add_model_knowledge(
                    self._extract_mechanism(state["current_model"]),
                    state["current_model"]
                )
        except Exception as e:
            logger.error(f"Error in update_knowledge_node: {e}")
        return state
    
    async def check_convergence_node(self, state: AgentState) -> AgentState:
        """Node for checking convergence and updating metrics"""
        try:
            if "metrics" not in state:
                state["metrics"] = self.default_metrics.copy()
            
            # Update standard metrics
            state["metrics"]["scores"].append(state["current_model"].score)
            state["metrics"]["model_complexity"].append(len(state["current_model"].equations[0].split()))
            state["metrics"]["iterations"].append(len(state["metrics"]["scores"]))
            
            # Update GoT metrics
            if state["current_model"]:
                thought_metrics = self.compute_thought_metrics(state["current_model"])
                state["metrics"]["thought_volumes"].append(thought_metrics["volume"])
                state["metrics"]["thought_latencies"].append(thought_metrics["latency"])
            
            if state["current_model"].equations:
                state["metrics"]["exploration_paths"].append(state["current_model"].equations[0])
                
        except Exception as e:
            logger.error(f"Error in check_convergence_node: {e}")
        return state
    
    async def end_workflow_node(self, state: AgentState) -> AgentState:
        """Final node in the workflow"""
        state["next_step"] = "complete"
        return state

    def _update_thought_tracking(self, state: AgentState):
        """Manage thought history and active thoughts"""
        # Add current thought to history
        state['thought_history'].append(state['current_model'])
        
        # Update active thoughts
        state['active_thoughts'].append(state['current_model'])
        if len(state['active_thoughts']) > self.max_active_thoughts:
            # Keep best thoughts based on score
            state['active_thoughts'] = sorted(
                state['active_thoughts'],
                key=lambda x: x.score if x.score is not None else float('-inf'),
                reverse=True
            )[:self.max_active_thoughts]

    def _add_to_thought_graph(self, model: ModelState, operation_type: str, 
                             parent_thoughts: List[ModelState] = None):
        """Add thought to graph with relationships"""
        self.thought_graph.add_node(model.id, state=model)
        if parent_thoughts:
            for parent in parent_thoughts:
                self.thought_graph.add_edge(parent.id, model.id, 
                                          operation=operation_type)

    def decide_next_step(self, state: AgentState) -> str:
        """Decide whether to continue or complete the workflow"""
        try:
            if len(state["metrics"]["scores"]) >= 50:
                return "complete"
            if state["current_model"].score > 0.9:
                return "complete"
            if len(state["metrics"]["scores"]) > 5:
                recent_scores = state["metrics"]["scores"][-5:]
                if max(recent_scores) - min(recent_scores) < 0.01:
                    return "refine"
            return "continue"
        except Exception as e:
            logger.error(f"Error in decide_next_step: {e}")
            return "continue"

    def _create_hypothesis_prompt(self, state: AgentState) -> str:
        """Create prompt for hypothesis generation"""
        return f"""
        Current model:
        {state["current_model"].equations[0]}
        
        Known mechanisms:
        {state["knowledge"]}
        
        Generate a new cognitive model that:
        1. Builds on successful aspects of previous models
        2. Incorporates relevant mechanisms
        3. Is mathematically precise
        4. Is theoretically sound
        
        RESPONSE FORMAT:
        EQUATION: [equation]
        PARAMETERS: [param1: value1, param2: value2, ...]
        THEORETICAL_BASIS: [brief explanation]
        """

    def _parse_llm_response(self, text: str) -> Optional[ModelState]:
        """Parse LLM response into a ModelState"""
        try:
            lines = text.strip().split('\n')
            equation = None
            parameters = {}
            
            for line in lines:
                if line.startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                elif line.startswith('PARAMETERS:'):
                    params_str = line.replace('PARAMETERS:', '').strip()
                    for pair in params_str.split(','):
                        if ':' in pair:
                            key, value_str = pair.split(':')
                            key = key.strip()
                            try:
                                value = float(value_str.strip())
                                parameters[key] = value
                            except ValueError:
                                continue
            
            if equation and parameters:
                return ModelState(equations=[equation], parameters=parameters)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
        return None

    def _compute_model_score(self, model: ModelState, knowledge: Dict) -> float:
        """Compute model score"""
        try:
            # Simple scoring based on equation complexity and parameter count
            complexity_penalty = len(model.equations[0].split()) * 0.01
            param_penalty = len(model.parameters) * 0.05
            base_score = 0.5  # Base score
            
            return base_score - complexity_penalty - param_penalty
            
        except Exception as e:
            logger.error(f"Error computing model score: {e}")
            return 0.0

    def _extract_mechanism(self, model: ModelState) -> str: 
        """Extract mechanism type from model, totally stupid now"""
        if "Q(t)" in model.equations[0]:
            return "reinforcement_learning"
        if "WM(t)" in model.equations[0]:
            return "working_memory"
        return "unknown_mechanism"