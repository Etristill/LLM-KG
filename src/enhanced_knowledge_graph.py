# src/enhanced_knowledge_graph.py, new code that works with our setup

from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
from dataclasses import dataclass
import pandas as pd
import json
import logging
from core import ModelState
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MechanismInfo:
    """Structured information about cognitive mechanisms"""
    name: str
    description: str
    base_equations: List[str]
    parameters: Dict[str, Dict[str, List[float]]]
    requirements: List[str]
    related_mechanisms: List[str]
    constraints: List[str]

class EnhancedKnowledgeGraph:
    """Enhanced knowledge graph for cognitive modeling"""
    
    def __init__(self, load_initial: bool = True):
        self.mechanism_graph = nx.DiGraph()  # For mechanism relationships
        self.model_graph = nx.DiGraph()      # For model relationships
        self.experiment_graph = nx.DiGraph()  # For experiment data
        self.init_experiment_graph()
        
        # Performance
        self.mechanism_performance: Dict[str, Dict] = {}
        self.model_cache: Dict[str, ModelState] = {}
        
        # Initialize with basic knowledge!!!!!
        if load_initial:
            self._initialize_base_knowledge()
    
    def add_mechanism(self, info: MechanismInfo) -> bool:
        """Add or update a mechanism in the graph"""
        try:
            self.mechanism_graph.add_node(
                info.name,
                description=info.description,
                base_equations=info.base_equations,
                parameters=info.parameters,
                requirements=info.requirements,
                constraints=info.constraints
            )
            
            #Add relationships to other mechanisms
            for related in info.related_mechanisms:
                if related in self.mechanism_graph:
                    self.mechanism_graph.add_edge(
                        info.name,
                        related,
                        relationship_type="related"
                    )
            
            logger.info(f"Added mechanism: {info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding mechanism {info.name}: {e}")
            return False


    def init_experiment_graph(self) -> bool:
        """Initialize the experiment graph"""
        self.experiment_graph = nx.DiGraph()

        # add experimental factors
        self.experiment_graph.add_node('n_trials', type = "experiment factor")
        self.experiment_graph.add_node('p_init', type = "experiment factor")
        self.experiment_graph.add_node('sigma', type = "experiment factor")
        self.experiment_graph.add_node('hazard_rate', type = "experiment factor")

        return


    def add_experiment_from_csv(self, file_location: str) -> bool:
        """Add experiment and participant information from experiment data to the graph"""

        # read csv file into pandas df
        df = pd.read_csv(file_location)

        exp_participant_pairs = df.drop_duplicates(subset=['experiment_id', 'session'])

        for index, row in exp_participant_pairs.iterrows():

            # add experiment
            try:
                exp_node_name = "Experiment " + str(row['experiment_id'])
                # check if experiment node exists
                if not (self.experiment_graph.has_node(exp_node_name) and
                                                      self.experiment_graph.nodes[exp_node_name].get('type') == 'experiment'):
                    self.experiment_graph.add_node(
                        exp_node_name,
                        type = "experiment")

                    # add experimental factors
                    self.experiment_graph.add_edge('n_trials', exp_node_name, relationship_type="has " + str(row['n_trials']))
                    self.experiment_graph.add_edge('p_init', exp_node_name, relationship_type="has " + str(row['p_init']))
                    self.experiment_graph.add_edge('sigma', exp_node_name, relationship_type="has " + str(row['sigma']))
                    self.experiment_graph.add_edge('hazard_rate', exp_node_name, relationship_type="has " + str(row['hazard_rate']))

            except Exception as e:
                logger.error(f"Error adding node {exp_node_name}: {e}")
                return False

            # add participant
            try:
                participant_node_name = "Participant " + str(row['session'])
                # check if participant node exists
                if not (self.experiment_graph.has_node(participant_node_name) and
                                                       self.experiment_graph.nodes[participant_node_name].get(
                                                           'type') == 'participant'):

                    self.experiment_graph.add_node(
                        participant_node_name,
                        age = row['age'],
                        gender = row['gender'],
                        type="participant")

                    # add relationships to experiment
                    self.experiment_graph.add_edge(participant_node_name, exp_node_name, relationship_type='participated in')

            except Exception as e:
                logger.error(f"Error adding node {participant_node_name}: {e}")
                return False

    
    def add_model_knowledge(self, mechanism: str, model: ModelState) -> bool:
        """Add model knowledge to the graph"""
        try:
            if not model.id:
                logger.error("Model must have an ID")
                return False
            
            # Add model node if not exists
            if not self.model_graph.has_node(model.id):
                self.model_graph.add_node(
                    model.id,
                    equations=model.equations,
                    parameters=model.parameters,
                    score=model.score,
                    mechanisms=[mechanism]
                )
            
            # Link to mechanism
            if mechanism in self.mechanism_graph:
                self.model_graph.add_edge(
                    mechanism,
                    model.id,
                    relationship_type="implements"
                )
                
                # Update performance tracking
                if model.score is not None:
                    if mechanism not in self.mechanism_performance:
                        self.mechanism_performance[mechanism] = {
                            'scores': [],
                            'best_score': float('-inf'),
                            'best_model': None
                        }
                    
                    perf = self.mechanism_performance[mechanism]
                    perf['scores'].append(model.score)
                    
                    if model.score > perf['best_score']:
                        perf['best_score'] = model.score
                        perf['best_model'] = model.id
                
                self.model_cache[model.id] = model
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding model knowledge: {e}")
            return False
    
    def query_mechanism(self, mechanism: str) -> Dict:
        """Get comprehensive information about a mechanism"""
        try:
            if mechanism not in self.mechanism_graph:
                return {}
            
            # Get base mechanism data
            info = dict(self.mechanism_graph.nodes[mechanism])
            
            # Add performance data
            if mechanism in self.mechanism_performance:
                perf = self.mechanism_performance[mechanism]
                info['performance'] = {
                    'best_score': perf['best_score'],
                    'avg_score': np.mean(perf['scores']) if perf['scores'] else 0.0,
                    'n_models': len(perf['scores'])
                }
            
            # Add related mechanisms
            info['related_mechanisms'] = list(self.mechanism_graph.neighbors(mechanism))
            
            # Add best model if exists
            if mechanism in self.mechanism_performance:
                best_model_id = self.mechanism_performance[mechanism]['best_model']
                if best_model_id and best_model_id in self.model_cache:
                    info['best_model'] = self.model_cache[best_model_id]
            
            return info
            
        except Exception as e:
            logger.error(f"Error querying mechanism {mechanism}: {e}")
            return {}
    
    def get_related_mechanisms(self, mechanism: str) -> List[Dict]:
        """Get mechanisms related to the given one"""
        try:
            if mechanism not in self.mechanism_graph:
                return []
            
            related = []
            for neighbor in self.mechanism_graph.neighbors(mechanism):
                edge_data = self.mechanism_graph.edges[mechanism, neighbor]
                related.append({
                    'mechanism': neighbor,
                    'relationship': edge_data.get('relationship_type', 'related'),
                    'weight': edge_data.get('weight', 1.0)
                })
            
            return related
            
        except Exception as e:
            logger.error(f"Error getting related mechanisms: {e}")
            return []
    
    def get_mechanism_performance(self, mechanism: str) -> Dict:
        """Get performance statistics for a mechanism"""
        try:
            if mechanism not in self.mechanism_performance:
                return {}
            
            perf = self.mechanism_performance[mechanism]
            scores = perf['scores']
            
            return {
                'best_score': perf['best_score'],
                'avg_score': np.mean(scores) if scores else 0.0,
                'std_score': np.std(scores) if scores else 0.0,
                'n_models': len(scores),
                'score_history': scores[-10:],  # Last 10 scores
                'best_model_id': perf['best_model']
            }
            
        except Exception as e:
            logger.error(f"Error getting mechanism performance: {e}")
            return {}
    
    def load_from_csv(self, path: str) -> bool:
        """Load knowledge from CSV file - here would be nice to have smething smart"""
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                # Add mechanism if not exists
                if not self.mechanism_graph.has_node(row['source_node']):
                    self.add_mechanism(MechanismInfo(
                        name=row['source_node'],
                        description="Loaded from CSV",
                        base_equations=[],
                        parameters={},
                        requirements=[],
                        related_mechanisms=[],
                        constraints=[]
                    ))
                
                # Add target if not exists
                if not self.mechanism_graph.has_node(row['target_node']):
                    self.add_mechanism(MechanismInfo(
                        name=row['target_node'],
                        description="Loaded from CSV",
                        base_equations=[],
                        parameters={},
                        requirements=[],
                        related_mechanisms=[],
                        constraints=[]
                    ))
                
                # Add relationship
                self.mechanism_graph.add_edge(
                    row['source_node'],
                    row['target_node'],
                    relationship_type=row['edge_relation']
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}")
            return False
    
    def export_to_neo4j(self) -> str:
        """Export graph data in Neo4j format"""
        try:
            combined_graph = nx.compose(self.mechanism_graph, self.model_graph)
            
            # Convert to Neo4j format
            neo4j_data = []
            for node, data in combined_graph.nodes(data=True):
                neo4j_data.append({
                    "node": str(node),
                    "properties": json.dumps(data)
                })
            
            for source, target, data in combined_graph.edges(data=True):
                neo4j_data.append({
                    "source": str(source),
                    "target": str(target),
                    "type": data.get('relationship_type', 'RELATED_TO'),
                    "properties": json.dumps(data)
                })
            
            return json.dumps(neo4j_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting to Neo4j: {e}")
            return ""
    
    def _initialize_base_knowledge(self):
        """Initialize with basic cognitive mechanisms"""
        base_mechanisms = [
            MechanismInfo(
                name="reinforcement_learning",
                description="Learning from rewards and punishments",
                base_equations=["Q(t+1) = Q(t) + α(R(t) - Q(t))"],
                parameters={
                    "learning_rate": {"typical_range": [0.1, 0.5]},
                    "temperature": {"typical_range": [0.5, 2.0]}
                },
                requirements=["needs_learning_rate"],
                related_mechanisms=["prediction_error"],
                constraints=["learning_rate_bounded", "positive_temperature"]
            ),
            MechanismInfo(
                name="working_memory",
                description="Temporary storage and manipulation of information",
                base_equations=["WM(t+1) = γWM(t) + (1-γ)Input(t)"],
                parameters={
                    "decay_rate": {"typical_range": [0.1, 0.9]},
                    "capacity": {"typical_range": [4, 7]}
                },
                requirements=["needs_decay_rate"],
                related_mechanisms=["prediction_error"],
                constraints=["decay_rate_bounded"]
            ),
            MechanismInfo(
                name="prediction_error",
                description="Difference between expected and actual outcomes",
                base_equations=["PE(t) = R(t) - Q(t)"],
                parameters={},
                requirements=[],
                related_mechanisms=["reinforcement_learning", "working_memory"],
                constraints=[]
            )
        ]
        
        for mech in base_mechanisms:
            self.add_mechanism(mech)
            
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph"""
        try:
            n_mechanisms = self.mechanism_graph.number_of_nodes()
            n_models = self.model_graph.number_of_nodes()
            n_relationships = (self.mechanism_graph.number_of_edges() + 
                            self.model_graph.number_of_edges())
            
            stats = {
                'n_mechanisms': n_mechanisms,
                'n_models': n_models,
                'n_relationships': n_relationships,
                'avg_model_score': None
            }
            
            # Calculate average model score if we have models
            if n_models > 0:
                scores = [data.get('score', 0) 
                         for _, data in self.model_graph.nodes(data=True)
                         if data.get('score') is not None]
                if scores:
                    stats['avg_model_score'] = float(np.mean(scores))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                'n_mechanisms': 0,
                'n_models': 0,
                'n_relationships': 0,
                'avg_model_score': None
            }