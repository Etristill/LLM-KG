# mcts_kg.py

# - Search algorithm that explores different model variations
# - Uses knowledge graph to guide search
# - !!!! has graph-based tracking of model relationships !!!
# - Can combine and refine models based on their relationships

# mcts_kg.py

from typing import Dict, Optional, List, Set, Tuple
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
        # Basic MCTS attributes
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
        
        # Enhanced Graph of Thoughts attributes
        self.thought_parents: List["MCTSNode"] = []
        self.refinement_count = 0
        self.creation_time = 0
        self.node_type = 'exploration'  # 'exploration', 'refinement', or 'aggregation'
        self.aggregation_sources: List[str] = []
        
        # Scoring components
        self.theoretical_score = 0.0
        self.empirical_score = 0.0
        self.complexity_score = 0.0
        self.influence_score = 0.0

class EnhancedMCTS:
    """Monte Carlo Tree Search enhanced with Graph of Thoughts and Knowledge Graph capabilities"""
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph, exploration_constant: float = 1.414):
        # Basic setup
        self.kg = knowledge_graph
        self.c = exploration_constant
        self.exploration_constant = exploration_constant
        
        # Graph structures
        self.thought_graph = nx.DiGraph()
        self.current_time = 0
        
        # Configuration
        self.volume_weight = 0.1
        self.latency_weight = 0.05
        self.refinement_threshold = 0.5
        self.max_refinements = 5
        
        # Caching and history
        self.mechanism_cache: Dict[str, Dict] = {}
        self.successful_patterns: Set[str] = set()
        
        # Metrics tracking
        self.metrics = {
            'kg_influence': [],
            'got_coverage': [],
            'theoretical_alignment': [],
            'empirical_performance': []
        }

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using combined criteria"""
        visited_nodes = set()
        self.current_time += 1
        
        while node.children and not node.untried_actions:
            if node in visited_nodes:
                logger.warning("Cycle detected in MCTS tree. Breaking loop.")
                break
                
            visited_nodes.add(node)
            node = self._select_knowledge_uct(node)
            self._update_thought_graph(node)  # <-- Now implemented
            
        return node

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node with enhanced knowledge guidance"""
        if not node.untried_actions:
            return None
            
        try:
            # Select and apply action
            action = self._select_action_with_knowledge(node)
            node.untried_actions.remove(action)
            new_state = self._apply_action_with_knowledge(node.state, action)
            
            # Create child node with enhanced attributes
            child = MCTSNode(new_state, parent=node)
            child.creation_time = self.current_time
            child.thought_parents.append(node)
            
            # Calculate initial scores
            child.theoretical_score = self._compute_theoretical_alignment(child.state)
            child.empirical_score = self._compute_empirical_alignment(child.state)
            child.complexity_score = self._compute_complexity_score(child.state)
            
            node.children.append(child)
            
            self._update_thought_graph(child)  # <-- Now implemented
            self._manage_active_thoughts(child)
            
            return child
            
        except Exception as e:
            logger.error(f"Error in expand: {str(e)}")
            return None

    def _select_knowledge_uct(self, node: MCTSNode) -> MCTSNode:
        """Enhanced UCT selection"""
        def combined_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            
            # Base UCT
            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
            
            # Graph position
            centrality = self._compute_centrality(child)
            influence = self._compute_influence(child)
            
            # Knowledge components
            kg_score = self._compute_kg_prior(child.state)
            theoretical = child.theoretical_score
            empirical = child.empirical_score
            
            # Weighted combination
            return (
                0.3 * exploitation + 
                0.2 * exploration +
                0.1 * centrality +
                0.1 * influence +
                0.1 * kg_score +
                0.1 * theoretical +
                0.1 * empirical
            )

        return max(node.children, key=combined_score)

    def _select_action_with_knowledge(self, node: MCTSNode) -> str:
        """Select action using enhanced knowledge guidance"""
        try:
            # Get mechanism info for current state
            extracted = self._extract_mechanisms(node.state)
            mechanism = extracted[0] if extracted else "unknown"
            info = self.kg.query_mechanism(mechanism)
            
            # Score each available action
            action_scores = {}
            for action in node.untried_actions:
                # Base score from knowledge graph
                base_score = self._compute_action_score(action, info)
                
                # Add complexity consideration
                complexity_penalty = self._compute_complexity_penalty(node.state, action)
                
                # Add success history bonus
                success_bonus = 0.2 if action in self.successful_patterns else 0.0
                
                # Final score
                action_scores[action] = base_score - complexity_penalty + success_bonus
            
            # Select probabilistically
            total_score = sum(action_scores.values())
            if total_score == 0:
                return np.random.choice(node.untried_actions)
            
            probs = [score / total_score for score in action_scores.values()]
            return np.random.choice(list(action_scores.keys()), p=probs)
            
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            return np.random.choice(node.untried_actions) if node.untried_actions else ""

    def _compute_action_score(self, action: str, mechanism_info: Dict) -> float:
        """Compute score for an action based on mechanism knowledge"""
        try:
            base_score = 1.0
            if not mechanism_info:
                return base_score
            
            # Score based on parameter presence
            param_map = {
                'add_learning_rate': 'alpha',
                'add_forgetting': 'gamma',
                'add_temperature': 'temp',
                'add_bias': 'bias'
            }
            
            if action in param_map:
                param = param_map[action]
                if 'parameters' in mechanism_info and param in mechanism_info['parameters']:
                    base_score *= 1.5
            
            # Score based on mechanism requirements
            if 'requirements' in mechanism_info:
                requirements = mechanism_info['requirements']
                if 'needs_learning_rate' in requirements and action == 'add_learning_rate':
                    base_score *= 2.0
                if 'needs_temperature' in requirements and action == 'add_temperature':
                    base_score *= 2.0
            
            # Score based on successful patterns
            if action in self.successful_patterns:
                base_score *= 1.3
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error computing action score: {str(e)}")
            return 1.0

    def _compute_theoretical_alignment(self, state: ModelState) -> float:
        """Compute theoretical alignment score for a state"""
        try:
            mechanisms = self._extract_mechanisms(state)
            total_score = 0.0
            mechanism_count = 0
            
            for mechanism in mechanisms:
                # Get mechanism info from cache or KG
                if mechanism in self.mechanism_cache:
                    info = self.mechanism_cache[mechanism]['info']
                else:
                    info = self.kg.query_mechanism(mechanism)
                    self.mechanism_cache[mechanism] = {'info': info}
                
                if not info:
                    continue
                
                mechanism_count += 1
                mechanism_score = 0.0
                
                # Check equation structure
                if 'base_equations' in info:
                    structure_score = self._compute_equation_similarity(
                        state.equations[0],
                        info['base_equations'][0]
                    )
                    mechanism_score += 0.6 * structure_score
                
                # Check parameter validity
                if 'parameters' in info:
                    param_score = self._compute_parameter_validity(
                        state.parameters,
                        info['parameters']
                    )
                    mechanism_score += 0.4 * param_score
                
                total_score += mechanism_score
            
            # Update metrics
            final_score = total_score / max(mechanism_count, 1)
            self.metrics['theoretical_alignment'].append(final_score)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing theoretical alignment: {str(e)}")
            return 0.0

    def _compute_empirical_alignment(self, state: ModelState) -> float:
        """Compute empirical alignment score"""
        try:
            if state.score is None:
                return 0.0
                
            mechanisms = self._extract_mechanisms(state)
            total_score = 0.0
            
            for mechanism in mechanisms:
                # Get performance data from cache or KG
                if mechanism in self.mechanism_cache:
                    performance = self.mechanism_cache[mechanism].get('performance')
                else:
                    performance = self.kg.get_mechanism_performance(mechanism)
                    if mechanism not in self.mechanism_cache:
                        self.mechanism_cache[mechanism] = {}
                    self.mechanism_cache[mechanism]['performance'] = performance
                
                if performance and 'best_score' in performance:
                    mechanism_score = state.score / max(performance['best_score'], 0.001)
                    total_score += min(mechanism_score, 1.0)
            
            # Update metrics
            final_score = total_score / max(len(mechanisms), 1)
            self.metrics['empirical_performance'].append(final_score)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing empirical alignment: {str(e)}")
            return 0.0

    def _compute_complexity_score(self, state: ModelState) -> float:
        """Compute complexity score for a state"""
        try:
            # Base complexity from equation length
            equation = state.equations[0]
            term_count = len(equation.split())
            param_count = len(state.parameters)
            
            # Penalize excessive complexity
            complexity_penalty = max(0, term_count - 10) * 0.1
            param_penalty = max(0, param_count - 4) * 0.1
            
            # Additional penalties for specific terms
            special_terms = ['exp', 'log', 'sqrt']
            special_term_penalty = sum(0.05 for term in special_terms if term in equation)
            
            total_penalty = complexity_penalty + param_penalty + special_term_penalty
            
            return max(0, 1.0 - total_penalty)
            
        except Exception as e:
            logger.error(f"Error computing complexity score: {str(e)}")
            return 0.0

    def _compute_kg_prior(self, state: ModelState) -> float:
        """Enhanced knowledge graph prior computation"""
        try:
            mechanisms = self._extract_mechanisms(state)
            total_score = 0.0
            mechanism_count = 0
            
            for mechanism in mechanisms:
                # Cache handling
                if mechanism not in self.mechanism_cache:
                    info = self.kg.query_mechanism(mechanism)
                    performance = self.kg.get_mechanism_performance(mechanism)
                    self.mechanism_cache[mechanism] = {
                        'info': info if info else {},
                        'performance': performance if performance else {}
                    }
                    logger.debug(f"KG cache updated for mechanism: {mechanism}")
                
                cached_data = self.mechanism_cache[mechanism]
                mechanism_count += 1
                
                # Theoretical alignment
                if cached_data.get('info', {}).get('base_equations'):
                    structure_score = self._compute_equation_similarity(
                        state.equations[0],
                        cached_data['info']['base_equations'][0]
                    )
                    total_score += 0.4 * structure_score
                
                # Parameter validity
                if cached_data.get('info', {}).get('parameters'):
                    param_score = self._compute_parameter_validity(
                        state.parameters,
                        cached_data['info']['parameters']
                    )
                    total_score += 0.3 * param_score
                
                # Historical performance
                perf_data = cached_data.get('performance', {})
                if perf_data and perf_data.get('best_score'):
                    perf_score = self._compute_performance_alignment(
                        state,
                        perf_data
                    )
                    total_score += 0.3 * perf_score
            
            final_score = total_score / max(mechanism_count, 1)
            self.metrics['kg_influence'].append(final_score)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing KG prior: {str(e)}")
            return 0.0

    def _compute_centrality(self, node: MCTSNode) -> float:
        """Compute node's centrality in thought graph"""
        try:
            if not self.thought_graph.has_node(node.state.id):
                return 0.0
            
            # Combine different centrality metrics
            try:
                degree_cent = nx.degree_centrality(self.thought_graph)[node.state.id]
                between_cent = nx.betweenness_centrality(self.thought_graph)[node.state.id]
                return 0.6 * degree_cent + 0.4 * between_cent
            except ImportError:
                # Fallback to simpler metric if NetworkX features unavailable
                return len(list(self.thought_graph.neighbors(node.state.id))) / self.thought_graph.number_of_nodes()
                
        except Exception as e:
            logger.error(f"Error computing centrality: {str(e)}")
            return 0.0

    def _compute_influence(self, node: MCTSNode) -> float:
        """Compute node's influence in thought graph"""
        try:
            if not self.thought_graph.has_node(node.state.id):
                return 0.0
            
            try:
                # Try PageRank first>>?
                pagerank = nx.pagerank(self.thought_graph, alpha=0.85)
                return pagerank[node.state.id]
            except ImportError:
                # Fallback to simpler centrality
                return nx.degree_centrality(self.thought_graph)[node.state.id]
                
        except Exception as e:
            logger.error(f"Error computing influence: {str(e)}")
            return 0.0

    def _compute_equation_similarity(self, eq1: str, eq2: str) -> float:
        """Compute similarity between equations"""
        try:
            terms1 = set(eq1.split())
            terms2 = set(eq2.split())
            return len(terms1 & terms2) / len(terms1 | terms2)
        except Exception as e:
            logger.error(f"Error computing equation similarity: {str(e)}")
            return 0.0

    def _compute_parameter_validity(self, params: Dict, valid_params: Dict) -> float:
        """Compute parameter validity score"""
        try:
            if not params or not valid_params:
                return 0.0
                
            valid_count = 0
            total_params = len(params)
            
            for param, value in params.items():
                if param in valid_params:
                    param_range = valid_params.get(param, {}).get('typical_range', [0, 1])
                    if param_range[0] <= value <= param_range[1]:
                        valid_count += 1
            
            return valid_count / max(total_params, 1)
        except Exception as e:
            logger.error(f"Error in parameter validity: {str(e)}")
            return 0.0

    def _compute_performance_alignment(self, state: ModelState, performance: Dict) -> float:
        """Compute alignment with historical performance"""
        try:
            if not performance or not performance.get('best_score'):
                return 0.0
            
            if state.score is None:
                return 0.0
                
            return min(state.score / max(performance['best_score'], 0.001), 1.0)
        except Exception as e:
            logger.error(f"Error computing performance alignment: {str(e)}")
            return 0.0

    def _compute_complexity_penalty(self, state: ModelState, action: str) -> float:
        """Compute complexity penalty for an action"""
        try:
            # Count current terms in equation
            current_terms = len(state.equations[0].split())
            
            # Penalize if already complex
            if current_terms > 10:
                return 0.2
            
            # Additional penalty for certain actions
            if action in ['add_temperature', 'add_bias']:
                return 0.1 * (current_terms / 10)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error computing complexity penalty: {str(e)}")
            return 0.0

    def _apply_action_with_knowledge(self, state: ModelState, action: str) -> ModelState:
        """Apply action with theoretical guidance"""
        try:
            new_state = state.copy()
            
            # Enhanced action definitions
            actions = {
                "add_learning_rate": {
                    "param": "alpha",
                    "value": 0.1,
                    "equation": lambda eq: f"({eq}) + alpha * (R(t) - Q(t))"
                },
                "add_forgetting": {
                    "param": "gamma",
                    "value": 0.9,
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
                
                # Use the helper method to get a parameter value
                param_value = self._get_optimal_parameter_value(action_info["param"])
                
                # Apply transformation
                new_state.equations = [action_info["equation"](new_state.equations[0])]
                new_state.parameters[action_info["param"]] = param_value
                
                # Track successful pattern
                if action not in self.successful_patterns:
                    self.successful_patterns.add(action)
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error applying action: {str(e)}")
            return state.copy()

    def _manage_active_thoughts(self, node: MCTSNode):
        """Manage active thoughts with cleanup"""
        try:
            # Update node attributes in the graph
            if self.thought_graph.has_node(node.state.id):
                self.thought_graph.nodes[node.state.id].update({
                    'last_update_time': self.current_time,
                    'visit_count': node.visits,
                    'value': node.value
                })
            
            # Periodic cleanup
            if self.current_time % 50 == 0:  # Cleanup every 50 steps
                self._cleanup_thought_graph()
            
        except Exception as e:
            logger.error(f"Error managing active thoughts: {str(e)}")

    def _cleanup_thought_graph(self):
        """Remove old or low-value nodes from thought graph"""
        try:
            current_nodes = list(self.thought_graph.nodes())
            for node_id in current_nodes:
                node_data = self.thought_graph.nodes[node_id]
                
                # Remove if too old and low value
                if (
                    self.current_time - node_data.get('last_update_time', 0) > 100 and
                    node_data.get('value', 0) < self.refinement_threshold
                ):
                    self.thought_graph.remove_node(node_id)
                    
        except Exception as e:
            logger.error(f"Error cleaning up thought graph: {str(e)}")

    def _extract_mechanisms(self, state: ModelState) -> List[str]:
        """Extract cognitive mechanisms from model state"""
        try:
            mechanisms = []
            equation = state.equations[0].lower()
            
            # Enhanced mechanism detection patterns
            mechanism_patterns = {
                "reinforcement_learning": [
                    "q(t)", "r(t)", "reward", "alpha",
                    "learning_rate"
                ],
                "working_memory": [
                    "wm", "memory", "gamma", "decay",
                    "capacity"
                ],
                "prediction_error": [
                    "pe", "error", "r(t)-q(t)", "delta"
                ],
                "exploration": [
                    "temp", "temperature", "beta",
                    "explore", "exploit"
                ]
            }
            
            for mechanism, patterns in mechanism_patterns.items():
                if any(pattern in equation for pattern in patterns):
                    mechanisms.append(mechanism)
                    
            return mechanisms
            
        except Exception as e:
            logger.error(f"Error extracting mechanisms: {str(e)}")
            return []

    def get_metrics(self) -> Dict:
        """Get current metrics with safety checks"""
        try:
            metrics = {}
            
            for key in ['kg_influence', 'got_coverage', 'theoretical_alignment', 'empirical_performance']:
                if self.metrics[key]:
                    metrics[key] = np.mean(self.metrics[key][-50:])
                else:
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {
                'kg_influence': 0.0,
                'got_coverage': 0.0,
                'theoretical_alignment': 0.0,
                'empirical_performance': 0.0
            }

    def _update_thought_graph(self, node: MCTSNode):
        """
        Add or update the given node in the thought_graph. 
        Creates edges from the node's parent(s) to this node.
        """
        try:
            # Each ModelState should have a unique ID;!!!!!!!!!! using state.id for now.
            state_id = node.state.id
            if not self.thought_graph.has_node(state_id):
                self.thought_graph.add_node(
                    state_id,
                    node_type=node.node_type,
                    creation_time=node.creation_time,
                    refinement_count=node.refinement_count,
                    theoretical_score=node.theoretical_score,
                    empirical_score=node.empirical_score,
                    complexity_score=node.complexity_score,
                    influence_score=node.influence_score
                )

            #  edge from parent
            if node.parent is not None:
                parent_id = node.parent.state.id
                if not self.thought_graph.has_edge(parent_id, state_id):
                    self.thought_graph.add_edge(parent_id, state_id)

            # Also:link 'thought_parents' if using them in addition to 'parent'
            for tparent in node.thought_parents:
                tparent_id = tparent.state.id
                if not self.thought_graph.has_edge(tparent_id, state_id):
                    self.thought_graph.add_edge(tparent_id, state_id)
                    
        except Exception as e:
            logger.error(f"Error in _update_thought_graph: {str(e)}")

    # -------------------------------------------------------------------------
    def _get_optimal_parameter_value(self, param_name: str) -> float:
        """
        VDummy implementation for obtaining an optimal parameter value.
        todo: Replace or extend with real logic from  knowledge graph or data.
        """
        try:
            defaults = {
                "alpha": 0.2,
                "gamma": 0.9,
                "temp": 1.0,
                "bias": 0.0
            }
            return defaults.get(param_name, 0.1)  
        except Exception as e:
            logger.error(f"Error in _get_optimal_parameter_value: {str(e)}")
            return 0.1
