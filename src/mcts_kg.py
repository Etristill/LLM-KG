# mcts_kg.py

# - Search algorithm that explores different model variations
# - Uses knowledge graph to guide search
# - !!!! has graph-based tracking of model relationships !!!
# - Can combine and refine models based on their relationships



# the key idea is that while the MCTS tree represents exploration paths, the GoT graph represents thought relationships. 
# We need these to work together seamlessly.




# mcts_kg.py

from typing import Dict, Optional, List, Set
import math
import networkx as nx
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search, enhanced with Graph of Thoughts capabilities.
    Each node represents both a position in the search tree and a thought in the graph.
    """
    def __init__(self, state: ModelState, parent=None):
        # Basic MCTS attributes for tree search
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [
            "add_learning_rate",
            "add_forgetting",
            "add_temperature",
            "add_bias"
        ]
        
        # Graph of Thoughts attributes for tracking relationships
        self.thought_parents: List[MCTSNode] = []  # Track thought lineage
        self.refinement_count = 0  # Number of times this thought has been refined
        self.creation_time = 0  # For tracking thought age
        
        # Node classification and tracking
        self.node_type = 'exploration'  # Can be 'exploration', 'refinement', or 'aggregation'
        self.aggregation_sources: List[str] = []  # IDs of thoughts that were combined

class EnhancedMCTS:
    """
    Monte Carlo Tree Search enhanced with Graph of Thoughts capabilities.
    Manages both tree exploration and thought relationships.
    """
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph, exploration_constant: float = 1.414):
        # Basic MCTS setup
        self.kg = knowledge_graph
        self.c = exploration_constant
        
        # Graph of Thoughts structure
        self.thought_graph = nx.DiGraph()
        self.current_time = 0  # For tracking thought age
        
        # Configuration parameters
        self.volume_weight = 0.1
        self.latency_weight = 0.05
        self.refinement_threshold = 0.5
        self.max_refinements = 5
        
        # Active thoughts management
        self.active_thoughts: Set[str] = set()
        self.max_active_thoughts = 50

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using combined MCTS and GoT criteria"""
        visited_nodes = set()
        self.current_time += 1  # Track time for thought age
        
        while node.children and not node.untried_actions:
            if node in visited_nodes:
                logger.warning("Cycle detected in MCTS tree. Breaking loop.")
                break
                
            visited_nodes.add(node)
            node = self._select_knowledge_uct(node)
            self._update_thought_graph(node)
            
        return node

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node and track in both tree and graph structures"""
        if not node.untried_actions:
            return None

        # Select and apply action
        action = self._select_action_with_knowledge(node)
        node.untried_actions.remove(action)
        new_state = self._apply_action(node.state, action)
        
        # Create new node
        child = MCTSNode(new_state, parent=node)
        child.creation_time = self.current_time
        child.thought_parents.append(node)
        node.children.append(child)
        
        # Update thought graph
        self._update_thought_graph(child)
        self._manage_active_thoughts(child)
        
        return child

    def _select_knowledge_uct(self, node: MCTSNode) -> MCTSNode:
        """Enhanced UCT selection using both tree and graph information"""
        def combined_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            
            # Standard MCTS score
            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
            
            # Knowledge graph score
            kg_score = self._compute_kg_prior(child.state)
            
            # Graph of Thoughts metrics
            volume_bonus = self.volume_weight * self._compute_volume_score(child.state)
            latency_penalty = self.latency_weight * self._compute_latency_penalty(child.state)
            
            # Combine all scores
            return exploitation + exploration + kg_score + volume_bonus - latency_penalty

        return max(node.children, key=combined_score)

    def _update_thought_graph(self, node: MCTSNode):
        """Maintain thought graph structure"""
        # Add node if not present
        if not self.thought_graph.has_node(node.state.id):
            self.thought_graph.add_node(
                node.state.id,
                state=node.state,
                creation_time=node.creation_time,
                node_type=node.node_type
            )
            
        # Add relationships
        for parent in node.thought_parents:
            if self.thought_graph.has_node(parent.state.id):
                self.thought_graph.add_edge(
                    parent.state.id,
                    node.state.id,
                    relationship_type=node.node_type
                )

    def _manage_active_thoughts(self, node: MCTSNode):
        """Maintain set of active thoughts for potential aggregation"""
        self.active_thoughts.add(node.state.id)
        if len(self.active_thoughts) > self.max_active_thoughts:
            # Remove oldest thoughts
            sorted_thoughts = sorted(
                self.active_thoughts,
                key=lambda x: self.thought_graph.nodes[x]['creation_time']
            )
            self.active_thoughts = set(sorted_thoughts[-self.max_active_thoughts:])

    def aggregate_thoughts(self, nodes: List[MCTSNode]) -> Optional[MCTSNode]:
        """Combine multiple thoughts into a new enhanced thought"""
        if not nodes:
            return None
            
        try:
            # Select base thought (most complex one)
            base_node = max(nodes, key=lambda n: len(n.state.equations[0].split()))
            new_state = base_node.state.copy()
            
            # Create aggregated node
            new_node = MCTSNode(new_state, parent=None)
            new_node.creation_time = self.current_time
            new_node.node_type = 'aggregation'
            new_node.thought_parents = nodes
            new_node.aggregation_sources = [n.state.id for n in nodes]
            
            # Update thought graph
            self._update_thought_graph(new_node)
            self._manage_active_thoughts(new_node)
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error aggregating thoughts: {e}")
            return None

    def refine_thought(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Iteratively improve a thought"""
        if node.refinement_count >= self.max_refinements:
            return None
            
        try:
            # Create refined state
            new_state = self._apply_refinement(node.state)
            new_node = MCTSNode(new_state, parent=node)
            new_node.creation_time = self.current_time
            new_node.node_type = 'refinement'
            new_node.thought_parents = [node]
            new_node.refinement_count = node.refinement_count + 1
            
            # Update thought graph
            self._update_thought_graph(new_node)
            self._manage_active_thoughts(new_node)
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error refining thought: {e}")
            return None

    def _compute_volume_score(self, state: ModelState) -> float:
        """Calculate score based on number of predecessor thoughts"""
        try:
            if not self.thought_graph.has_node(state.id):
                return 0.0
            return len(nx.ancestors(self.thought_graph, state.id))
        except Exception as e:
            logger.error(f"Error computing volume score: {e}")
            return 0.0

    def _compute_latency_penalty(self, state: ModelState) -> float:
        """Calculate penalty based on path length to thought"""
        try:
            if not self.thought_graph.has_node(state.id):
                return 0.0
                
            # Find longest path to this thought
            root_nodes = [n for n in self.thought_graph.nodes() 
                         if self.thought_graph.in_degree(n) == 0]
            
            max_length = 0
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, state.id):
                    paths = nx.all_simple_paths(self.thought_graph, root, state.id)
                    max_length = max(max_length, max(len(path) for path in paths))
                    
            return max_length
            
        except Exception as e:
            logger.error(f"Error computing latency penalty: {e}")
            return 0.0

    def _compute_kg_prior(self, state: ModelState) -> float:
        """Get prior knowledge score from knowledge graph"""
        try:
            mechanism = self._extract_mechanism(state)
            info = self.kg.query_mechanism(mechanism)
            
            # Compare with best known model
            similarity = self._compute_model_similarity(
                state, info.get('best_model', None)
            )
            mechanism_score = info.get('best_score', 0.0)
            
            return 0.3 * similarity + 0.2 * mechanism_score
            
        except Exception as e:
            logger.error(f"Error computing KG prior: {e}")
            return 0.0

    def _compute_model_similarity(self, model: ModelState, reference: Optional[Dict]) -> float:
        """Compare model similarity using equation terms"""
        if not reference:
            return 0.0
            
        try:
            model_terms = set(model.equations[0].split())
            ref_terms = set(reference['equations'][0].split())
            
            return len(model_terms & ref_terms) / len(model_terms | ref_terms)
            
        except Exception as e:
            logger.error(f"Error computing model similarity: {e}")
            return 0.0

    def _select_action_with_knowledge(self, node: MCTSNode) -> str:
        """Select action using knowledge graph guidance"""
        mechanism = self._extract_mechanism(node.state)
        info = self.kg.query_mechanism(mechanism)
        
        # Score each action
        action_scores = {}
        for action in node.untried_actions:
            score = self._compute_action_score(action, info)
            action_scores[action] = score
            
        # Select probabilistically
        total_score = sum(action_scores.values())
        if total_score == 0:
            return np.random.choice(node.untried_actions)
            
        probs = [score / total_score for score in action_scores.values()]
        return np.random.choice(list(action_scores.keys()), p=probs)

    def _compute_action_score(self, action: str, mechanism_info: Dict) -> float:
        """Score actions based on mechanism knowledge"""
        if not mechanism_info:
            return 1.0
            
        if 'parameters' in mechanism_info:
            if action == "add_learning_rate" and 'learning_rate' in mechanism_info['parameters']:
                return 2.0
            if action == "add_temperature" and 'temperature' in mechanism_info['parameters']:
                return 1.5
                
        return 1.0

    def _apply_action(self, state: ModelState, action: str) -> ModelState:
        """Apply an action to create a new state"""
        new_state = state.copy()
        
        actions = {
            "add_learning_rate": {
                "param": "alpha",
                "value": 0.1,
                "equation": lambda eq: f"({eq}) * alpha"
            },
            "add_forgetting": {
                "param": "gamma",
                "value": 0.1,
                "equation": lambda eq: f"({eq}) * (1 - gamma)"
            },
            "add_temperature": {
                "param": "temp",
                "value": 1.0,
                "equation": lambda eq: f"({eq}) / temp"
            },
            "add_bias": {
                "param": "bias",
                "value": 0.0,
                "equation": lambda eq: f"({eq}) + bias"
            }
        }
        
        if action in actions:
            action_info = actions[action]
            new_state.equations = [
                action_info["equation"](new_state.equations[0])
            ]
            new_state.parameters[action_info["param"]] = action_info["value"]
            
        return new_state

    def _apply_refinement(self, state: ModelState) -> ModelState:
        """Apply refinement transformation"""
        new_state = state.copy()
        for param in new_state.parameters:
            current_value = new_state.parameters[param]
            # Small random adjustment (±10%)
            adjustment = current_value * (1 + np.random.uniform(-0.1, 0.1))
            new_state.parameters[param] = adjustment
        return new_state

    def _extract_mechanism(self, state: ModelState) -> str:
        """Identify mechanism type from equation"""
        if "Q(t)" in state.equations[0]:
            return "reinforcement_learning"
        if "WM(t)" in state.equations[0]:
            return "working_memory"
        return "unknown_mechanism"

    def get_thought_metrics(self, state: ModelState) -> Dict:
        """Get GoT metrics for a thought"""
        return {
            'volume': self._compute_volume_score(state),
            'latency': self._compute_latency_penalty(state),
            'creation_time': self.thought_graph.nodes[state.id]['creation_time'] 
                if self.thought_graph.has_node(state.id) else 0
        }