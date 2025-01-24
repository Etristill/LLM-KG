# pipeline.py

import asyncio
from src.core import ModelState, generate_test_data
from src.llm_client import UnifiedLLMClient  
from src.mcts_kg import EnhancedMCTS, MCTSNode
from src.graph_workflow import ModelDiscoveryGraph, AgentState
from src.evaluation import SimpleEvaluator
from src.llm import EnhancedLLMInterface
from src.enhanced_knowledge_graph import EnhancedKnowledgeGraph, MechanismInfo
from src.transformations import ThoughtTransformations
from src.config import Config
import numpy as np
from typing import Optional, Tuple, List, Dict
import json
from datetime import datetime
import logging
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('discovery.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedModelDiscoveryPipeline:
    def __init__(self, 
                model_name: str = "claude-3-opus-20240229",
                use_mock_llm: bool = False,  #
                n_iterations: int = 50,
                exploration_constant: float = 1.414,
                use_csv: bool = True,
                knowledge_path: str = None):
        """Initialize the enhanced pipeline with both MCTS and workflow components"""
        logger.info("Initializing Enhanced Model Discovery Pipeline...")
    
        # Validate model configuration
        try:
            self.config = Config(model_name)
        except ValueError as e:
            logger.error(f"Model configuration error: {e}")
            raise
        
        # Initialize core components with enhanced knowledge graph
        self.test_data = generate_test_data(n_trials=100)
        
        # Initialize knowledge graph based on CSV availability
        if use_csv and knowledge_path and Path(knowledge_path).exists():
            logger.info(f"Initializing knowledge graph from {knowledge_path}")
            self.kg = EnhancedKnowledgeGraph(load_initial=True)
            self.kg.load_from_csv(knowledge_path)
        else:
            logger.info("Using mock knowledge graph")
            self.kg = EnhancedKnowledgeGraph(load_initial=True)
            self._initialize_mock_knowledge()
        
        # Initialize additional mechanisms
        self._initialize_custom_mechanisms()
        
        self.llm = UnifiedLLMClient(model_name=model_name)  
        self.mcts = EnhancedMCTS(self.kg, exploration_constant)
        self.evaluator = SimpleEvaluator(self.test_data)
        self.transformations = ThoughtTransformations(self.llm)  
        self.graph_workflow = ModelDiscoveryGraph(self.kg, test_data=self.test_data)
        
        self.n_iterations = n_iterations
        
        # Thought pool management
        self.thought_pool: List[ModelState] = []
        self.max_thought_pool_size = 10
        
        # Best model tracking
        self.best_score = float('-inf')
        self.best_model: Optional[ModelState] = None
        
        # Comprehensive metrics tracking
        self.metrics = {
            'scores': [],
            'model_complexity': [],
            'iterations': [],
            'exploration_paths': [],
            'thought_volumes': [],
            'thought_latencies': [],
            'aggregation_counts': [],
            'refinement_counts': [],
            'kg_updates': [],
            'evaluation_times': [],
            'successful_mechanisms': set()
        }
        
        # State tracking
        self.is_running = True
        
        logger.info(f"Pipeline initialized with model: {model_name}")

    def _initialize_mock_knowledge(self):
        """Initialize mock knowledge graph with basic mechanisms"""
        basic_mechanisms = [
            MechanismInfo(
                name="basic_rl",
                description="Basic reinforcement learning",
                base_equations=["Q(t+1) = Q(t) + alpha * (R(t) - Q(t))"],
                parameters={"alpha": {"typical_range": [0.1, 0.5]}},
                requirements=[],
                related_mechanisms=[],
                constraints=[]
            ),
            MechanismInfo(
                name="simple_memory",
                description="Simple memory mechanism",
                base_equations=["Q(t+1) = gamma * Q(t) + (1-gamma) * R(t)"],
                parameters={"gamma": {"typical_range": [0.1, 0.9]}},
                requirements=[],
                related_mechanisms=[],
                constraints=[]
            )
        ]
        
        for mech in basic_mechanisms:
            self.kg.add_mechanism(mech)
            logger.info(f"Added mock mechanism: {mech.name}")

    def _initialize_custom_mechanisms(self):
        """Initialize custom cognitive mechanisms in the knowledge graph"""
        mechanisms = [
            MechanismInfo(
                name="td_learning",
                description="Temporal difference learning mechanism",
                base_equations=["Q(t+1) = Q(t) + α(R(t) + γQ(t+1) - Q(t))"],
                parameters={
                    "alpha": {"typical_range": [0.1, 0.5]},
                    "gamma": {"typical_range": [0.8, 0.99]}
                },
                requirements=["needs_learning_rate", "needs_discount"],
                related_mechanisms=["reinforcement_learning", "prediction_error"],
                constraints=["learning_rate_bounded", "discount_bounded"]
            ),
            MechanismInfo(
                name="exploration_bonus",
                description="Exploration bonus mechanism",
                base_equations=["Q(t+1) = Q(t) + α(R(t) - Q(t)) + β/sqrt(N(t))"],
                parameters={
                    "beta": {"typical_range": [0.1, 2.0]}
                },
                requirements=["needs_visit_counts"],
                related_mechanisms=["reinforcement_learning"],
                constraints=["positive_beta"]
            ),
            MechanismInfo(
                name="adaptive_learning",
                description="Adaptive learning rate mechanism",
                base_equations=["α(t) = α₀/sqrt(t)", "Q(t+1) = Q(t) + α(t)(R(t) - Q(t))"],
                parameters={
                    "alpha_0": {"typical_range": [0.5, 2.0]}
                },
                requirements=["needs_time_tracking"],
                related_mechanisms=["reinforcement_learning"],
                constraints=["positive_alpha"]
            )
        ]
        
        for mech in mechanisms:
            self.kg.add_mechanism(mech)
            logger.info(f"Added mechanism: {mech.name}")

    def setup_signal_handlers(self):
        """Set up handlers for graceful shutdown"""
        def handle_interrupt(signum, frame):
            logger.info("\nReceived interrupt signal. Initiating graceful shutdown...")
            self.is_running = False
            self.save_results()
            
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    async def run(self):
        """Run the discovery pipeline with enhanced error handling and metrics"""
        try:
            # Initialize with basic model
            initial_state = ModelState(
                equations=["Q(t+1) = Q(t) + (R(t) - Q(t))"],
                parameters={}
            )
            
            # Create initial MCTS node
            root = MCTSNode(initial_state)
            self._manage_thought_pool(initial_state)
            
            # Initialize workflow state
            workflow_state = AgentState(
                current_model=initial_state,
                knowledge={},
                messages=[],
                next_step="start",
                metrics=self.metrics.copy(),
                thought_history=[],
                active_thoughts=[]
            )
            
            logger.info("Starting model discovery process...")
            
            for i in range(self.n_iterations):
                if not self.is_running:
                    logger.info("Stopping discovery process due to interrupt...")
                    break
                    
                try:
                    # MCTS Selection with knowledge guidance
                    node = self.mcts.select_node(root)
                    
                    # MCTS Expansion with error handling
                    if node.untried_actions:
                        child = self.mcts.expand(node)
                        if child:
                            node = child
                            self.metrics['exploration_paths'].append(child.state.equations[0])
                    
                    # Update workflow state
                    workflow_state["current_model"] = node.state
                    
                    # Run workflow
                    workflow_state = await self.graph_workflow.run_workflow(workflow_state)
                    simulation_state = workflow_state["current_model"]
                    
                    # Ensure the state is properly copied
                    simulation_state = simulation_state.copy()
                    self._manage_thought_pool(simulation_state)
                    
                    # Evaluate model
                    score = self.evaluator.evaluate_model(simulation_state)
                    
                    # Important: Assign score to the state
                    simulation_state.score = score
                    
                    # Update metrics and knowledge
                    self._update_metrics(simulation_state, score, i)
                    await self._update_knowledge_graph(simulation_state)
                    
                    # Update best model if needed
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = simulation_state.copy()
                        logger.info(f"\nNew best model found (iteration {i+1}):")
                        self._print_model_details(simulation_state, score)
                    
                    # Get and update thought metrics
                    got_metrics = self.graph_workflow.compute_thought_metrics(simulation_state)
                    self.metrics['thought_volumes'].append(got_metrics['volume'])
                    self.metrics['thought_latencies'].append(got_metrics['latency'])
                    
                    # MCTS backpropagation with enhanced scoring
                    enhanced_score = self._compute_enhanced_score(score, got_metrics)
                    self._backpropagate(node, enhanced_score)
                    
                    # Check workflow completion
                    if workflow_state["next_step"] == "complete":
                        logger.info("Workflow signaled completion")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in iteration {i}: {str(e)}")
                    continue
                
                # Progress update
                if (i + 1) % 10 == 0:
                    self._print_progress_update(i + 1)
                    
        except Exception as e:
            logger.error(f"Critical error in pipeline: {str(e)}")
            raise
        
        finally:
            logger.info("\nDiscovery process complete!")
            if self.best_model:
                logger.info("\nBest model found:")
                self._print_model_details(self.best_model, self.best_score)
                self.save_results()
            self._save_metrics()

    def _manage_thought_pool(self, new_thought: ModelState):
        """Manage the thought pool with enhanced tracking"""
        try:
            self.thought_pool.append(new_thought)
            
            if len(self.thought_pool) > self.max_thought_pool_size:
                # Keep best thoughts based on both score and diversity
                self.thought_pool.sort(
                    key=lambda x: (
                        x.score if x.score is not None else float('-inf')
                    ) + self._compute_diversity_bonus(x),
                    reverse=True
                )
                self.thought_pool = self.thought_pool[:self.max_thought_pool_size]
                
        except Exception as e:
            logger.error(f"Error in thought pool management: {str(e)}")

    def _compute_diversity_bonus(self, thought: ModelState) -> float:
        """Compute diversity bonus for thought selection"""
        try:
            if not self.thought_pool:
                return 0.0
                
            # Calculate average similarity to other thoughts
            similarities = []
            for other in self.thought_pool:
                if other != thought:
                    sim = self._compute_equation_similarity(
                        thought.equations[0],
                        other.equations[0]
                    )
                    similarities.append(sim)
            
            # Reward diversity (lower similarity)
            avg_similarity = np.mean(similarities) if similarities else 1.0
            return 0.2 * (1.0 - avg_similarity)
            
        except Exception as e:
            logger.error(f"Error computing diversity bonus: {str(e)}")
            return 0.0

    def _compute_equation_similarity(self, eq1: str, eq2: str) -> float:
        """Compute similarity between equations"""
        try:
            terms1 = set(eq1.split())
            terms2 = set(eq2.split())
            return len(terms1 & terms2) / len(terms1 | terms2)
        except Exception:
            return 0.0

    def _extract_mechanisms(self, state: ModelState) -> List[str]:
        """Extract cognitive mechanisms from model state"""
        mechanisms = []
        equation = state.equations[0].lower()
        
        # Enhanced mechanism detection patterns
        mechanism_patterns = {
            "reinforcement_learning": ["q(t)", "r(t)", "alpha", "learning"],
            "working_memory": ["wm(t)", "memory", "gamma", "decay"],
            "prediction_error": ["pe", "r(t)-q(t)", "error", "delta"],
            "td_learning": ["q(t+1)", "gamma", "discount"],
            "exploration_bonus": ["beta", "sqrt", "n(t)", "count"],
            "adaptive_learning": ["alpha(t)", "alpha_0", "sqrt(t)"],
            "basic_rl": ["q(t)", "alpha", "r(t)"],
            "simple_memory": ["gamma", "memory"]
        }
        
        for mechanism, patterns in mechanism_patterns.items():
            if any(pattern in equation for pattern in patterns):
                mechanisms.append(mechanism)
        
        return mechanisms

    async def _update_knowledge_graph(self, state: ModelState):
        """Update knowledge graph with new model information"""
        try:
            if state.score is not None:
                # Extract mechanisms with enhanced detection
                mechanisms = self._extract_mechanisms(state)
                
                for mechanism in mechanisms:
                    # Add model knowledge to graph
                    self.kg.add_model_knowledge(mechanism, state)
                    
                    # Update metrics
                    self.metrics['kg_updates'].append(mechanism)
                    self.metrics['successful_mechanisms'].add(mechanism)
                    
                    # Log performance statistics
                    perf_stats = self.kg.get_mechanism_performance(mechanism)
                    if perf_stats:
                        logger.debug(f"Mechanism {mechanism} stats: {perf_stats}")
                    
                logger.debug(f"Updated KG with model for mechanisms: {mechanisms}")
                # Export graph data periodically
                if len(self.metrics['kg_updates']) % 50 == 0:
                    self._export_graph_data()
                
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {str(e)}")

    def _compute_enhanced_score(self, base_score: float, got_metrics: dict) -> float:
        """Compute enhanced score incorporating various metrics"""
        try:
            volume_bonus = 0.1 * got_metrics['volume']
            latency_penalty = 0.05 * got_metrics['latency']
            
            # Add diversity bonus if available
            diversity_bonus = 0.0
            if self.thought_pool:
                current_mechanisms = self._extract_mechanisms(self.best_model)
                diversity_bonus = 0.1 * len(current_mechanisms)
            
            return base_score + volume_bonus - latency_penalty + diversity_bonus
            
        except Exception as e:
            logger.error(f"Error computing enhanced score: {str(e)}")
            return base_score

    def _update_metrics(self, state: ModelState, score: float, iteration: int):
        """Update all metrics including GoT-specific ones"""
        try:
            self.metrics['scores'].append(score)
            self.metrics['model_complexity'].append(len(state.equations[0].split()))
            self.metrics['iterations'].append(iteration)
            
            # Track successful mechanisms
            mechanisms = self._extract_mechanisms(state)
            if score > self.best_score:
                self.metrics['successful_mechanisms'].update(mechanisms)
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate results through the tree"""
        try:
            while node:
                node.visits += 1
                node.value += score
                node = node.parent
        except Exception as e:
            logger.error(f"Error in backpropagation: {str(e)}")

    def _print_model_details(self, state: ModelState, score: float):
        """Print detailed model information including GoT metrics"""
        print("\nEquation:")
        print(f"  {state.equations[0]}")
        print("\nParameters:")
        for param, value in state.parameters.items():
            print(f"  {param}: {value:.3f}")
        print(f"\nScore: {score:.3f}")
        
        got_metrics = self.graph_workflow.compute_thought_metrics(state)
        print(f"Thought Volume: {got_metrics['volume']}")
        print(f"Thought Latency: {got_metrics['latency']}")
        print(f"Model Complexity: {len(state.equations[0].split())}")

    def _print_progress_update(self, iteration: int):
        """Print detailed progress update with GoT metrics"""
        print(f"\nCompleted iteration {iteration}/{self.n_iterations}")
        if self.metrics['scores']:
            avg_score = np.mean(self.metrics['scores'][-10:])
            print(f"Average score (last 10): {avg_score:.3f}")
            print(f"Best score so far: {self.best_score:.3f}")
            print(f"Average thought volume: {np.mean(self.metrics['thought_volumes'][-10:]):.2f}")
            print(f"Average thought latency: {np.mean(self.metrics['thought_latencies'][-10:]):.2f}")
            print(f"Successful mechanisms: {len(self.metrics['successful_mechanisms'])}")

    def save_results(self):
        """Save comprehensive results to JSON"""
        try:
            if not self.best_model:
                return
                
            got_metrics = self.graph_workflow.compute_thought_metrics(self.best_model)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'best_model': {
                    'equations': self.best_model.equations,
                    'parameters': {k: float(v) for k, v in self.best_model.parameters.items()},
                    'score': float(self.best_score),
                    'thought_volume': got_metrics['volume'],
                    'thought_latency': got_metrics['latency']
                },
                'knowledge_graph_stats': self.kg.get_graph_statistics(),
                'settings': {
                    'n_iterations': self.n_iterations,
                    'exploration_constant': self.mcts.exploration_constant,
                    'model_name': self.config.model_name
                },
                'successful_mechanisms': list(self.metrics['successful_mechanisms'])
            }
            
            filename = f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def _save_metrics(self):
        """Save detailed metrics including GoT metrics"""
        try:
            metrics_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'scores': [float(s) for s in self.metrics['scores']],
                    'model_complexity': self.metrics['model_complexity'],
                    'iterations': self.metrics['iterations'],
                    'exploration_paths': self.metrics['exploration_paths'][-10:],
                    'thought_volumes': self.metrics['thought_volumes'],
                    'thought_latencies': self.metrics['thought_latencies'],
                    'aggregation_counts': self.metrics['aggregation_counts'],
                    'refinement_counts': self.metrics['refinement_counts'],
                    'kg_updates': self.metrics['kg_updates'][-50:],
                    'successful_mechanisms': list(self.metrics['successful_mechanisms']),
                    'evaluation_times': self.metrics['evaluation_times'],
                    'summary_stats': {
                        'avg_score': float(np.mean(self.metrics['scores'])) if self.metrics['scores'] else 0.0,
                        'max_score': float(np.max(self.metrics['scores'])) if self.metrics['scores'] else 0.0,
                        'avg_thought_volume': float(np.mean(self.metrics['thought_volumes'])) if self.metrics['thought_volumes'] else 0.0,
                        'avg_thought_latency': float(np.mean(self.metrics['thought_latencies'])) if self.metrics['thought_latencies'] else 0.0,
                        'total_kg_updates': len(self.metrics['kg_updates']),
                        'unique_mechanisms': len(self.metrics['successful_mechanisms'])
                    }
                }, f, indent=2)
            logger.info(f"Detailed metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def _export_graph_data(self):
        """Export graph data for visualization"""
        try:
            # Export Neo4j compatible data
            neo4j_data = self.kg.export_to_neo4j()
            if neo4j_data:
                filename = f"graph_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    f.write(neo4j_data)
                logger.info(f"Exported graph data to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting graph data: {str(e)}")

async def main():
    """Main entry point with enhanced error handling"""
    try:
        # Check for knowledge CSV file
        default_knowledge_path = "knowledge.csv"
        use_csv = Path(default_knowledge_path).exists()
        
        # Init pipeline with configuration
        pipeline = EnhancedModelDiscoveryPipeline(
            model_name="claude-3-opus-20240229",  # Specify the model explicitly
            use_mock_llm=False,  # Changed to False to use real LLM
            n_iterations=50,
            exploration_constant=1.414,
            use_csv=use_csv,
            knowledge_path=default_knowledge_path if use_csv else None
        )
        
        pipeline.setup_signal_handlers()
        
        await pipeline.run()
        
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        if hasattr(pipeline, 'save_results'):
            pipeline.save_results()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        try:
            if hasattr(pipeline, '_save_metrics'):
                pipeline._save_metrics()
        except Exception as e:
            logger.error(f"Error in final cleanup: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())