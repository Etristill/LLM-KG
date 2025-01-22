# src/knowledge_graph.py

# - Stores known information about cognitive mechanisms
# - Helps guide the search toward promising directions. 
# this is very basic ! but works

import networkx as nx
from typing import Dict, List, Optional, Any
from .core import ModelState
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CognitiveKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_knowledge()
        
        # Cache for performance
        self.mechanism_cache = {}
        self.model_cache = {}
        
    def query_mechanism(self, mechanism: str) -> Dict:
        """Get information about a specific mechanism including best models"""
        try:
            # Check cache first
            if mechanism in self.mechanism_cache:
                return self.mechanism_cache[mechanism]
            
            if mechanism in self.graph:
                # Get base mechanism data
                node_data = dict(self.graph.nodes[mechanism])
                
                # Get associated models
                models = self._get_mechanism_models(mechanism)
                node_data['models'] = models
                
                # Find best model
                if models:
                    best_model = max(models, key=lambda x: x.get('score', float('-inf')))
                    node_data['best_model'] = best_model
                    node_data['best_score'] = best_model.get('score', 0.0)
                
                # Cache and return
                self.mechanism_cache[mechanism] = node_data
                return node_data
                
            return {}
            
        except Exception as e:
            logger.error(f"Error querying mechanism {mechanism}: {e}")
            return {}

    def _get_mechanism_models(self, mechanism: str) -> List[Dict]:
        """Get all models associated with a mechanism"""
        try:
            models = []
            for node in self.graph.nodes():
                if node.startswith('model_') and self.graph.has_edge(mechanism, node):
                    model_data = dict(self.graph.nodes[node])
                    models.append({
                        'id': node,
                        'equations': model_data.get('equations', []),
                        'parameters': model_data.get('parameters', {}),
                        'score': model_data.get('score', 0.0)
                    })
            return models
            
        except Exception as e:
            logger.error(f"Error getting mechanism models: {e}")
            return []

    def get_related_mechanisms(self, mechanism: str) -> List[Dict]:
        """Get mechanisms related to the given one"""
        try:
            if mechanism not in self.graph:
                return []
                
            related = []
            for neighbor in self.graph.neighbors(mechanism):
                if not neighbor.startswith('model_'):
                    edge_data = self.graph.edges[mechanism, neighbor]
                    related.append({
                        'mechanism': neighbor,
                        'relationship': edge_data.get('type', ''),
                        'weight': edge_data.get('weight', 0.0)
                    })
            return related
            
        except Exception as e:
            logger.error(f"Error getting related mechanisms: {e}")
            return []

    def add_model_knowledge(self, mechanism: str, model: ModelState):
        """Add new model knowledge to the graph"""
        try:
            # Generate unique model identifier
            model_id = f"model_{len([n for n in self.graph.nodes() if n.startswith('model_')])}"
            
            # Add model node with comprehensive data
            self.graph.add_node(
                model_id,
                type='model',
                equations=model.equations,
                parameters=model.parameters,
                score=model.score if model.score is not None else 0.0,
                mechanism_type=mechanism
            )
            
            # Create edge between mechanism and model
            if mechanism in self.graph:
                self.graph.add_edge(
                    mechanism,
                    model_id,
                    relationship='implements',
                    weight=model.score if model.score is not None else 0.0
                )
                
                # Update mechanism's best model if applicable
                current_best = self.graph.nodes[mechanism].get('best_score', float('-inf'))
                if model.score and model.score > current_best:
                    self.graph.nodes[mechanism]['best_score'] = model.score
                    self.graph.nodes[mechanism]['best_model'] = model_id
                    logger.info(f"Updated best model for mechanism {mechanism}")
                
                # Clear cache
                self.mechanism_cache.pop(mechanism, None)
            
            # Add component relationships
            self._add_component_relationships(model_id, model)
            
            logger.info(f"Successfully added model {model_id} to knowledge graph")
            
        except Exception as e:
            logger.error(f"Error adding model to knowledge graph: {e}")

    def _add_component_relationships(self, model_id: str, model: ModelState):
        """Add relationships between model and its components"""
        try:
            equation = model.equations[0].lower()
            
            # Component detection patterns
            components = {
                'prediction_error': ['r(t)', 'q(t)', 'pe', 'error'],
                'working_memory': ['wm', 'memory', 'gamma', 'decay'],
                'exploration': ['temp', 'temperature', 'beta']
            }
            
            for component, keywords in components.items():
                if any(keyword in equation for keyword in keywords):
                    if component in self.graph:
                        self.graph.add_edge(
                            model_id,
                            component,
                            relationship='uses',
                            weight=0.5
                        )
                        
        except Exception as e:
            logger.error(f"Error adding component relationships: {e}")

    def get_mechanism_performance(self, mechanism: str) -> Dict:
        """Get performance statistics for a mechanism"""
        try:
            if mechanism not in self.graph:
                return {}
                
            models = self._get_mechanism_models(mechanism)
            scores = [m.get('score', 0.0) for m in models]
            
            if not scores:
                return {}
                
            return {
                'n_models': len(models),
                'best_score': max(scores),
                'avg_score': sum(scores) / len(scores),
                'score_distribution': scores
            }
            
        except Exception as e:
            logger.error(f"Error getting mechanism performance: {e}")
            return {}

    def get_graph_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph"""
        try:
            mechanisms = [n for n in self.graph.nodes() if not n.startswith('model_')]
            models = [n for n in self.graph.nodes() if n.startswith('model_')]
            
            return {
                'n_mechanisms': len(mechanisms),
                'n_models': len(models),
                'n_relationships': self.graph.number_of_edges(),
                'mechanisms': mechanisms,
                'model_distribution': {
                    mech: len([m for m in models if self.graph.has_edge(mech, m)])
                    for mech in mechanisms
                },
                'avg_model_score': np.mean([
                    self.graph.nodes[m].get('score', 0.0) for m in models
                ]) if models else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}

    def _initialize_knowledge(self):
        """Initialize with basic cognitive mechanisms"""
        # Basic mechanism definitions
        mechanisms = {
            'reinforcement_learning': {
                'description': 'Learning from rewards and punishments',
                'base_equations': ['Q(t+1) = Q(t) + α(R(t) - Q(t))'],
                'parameters': {
                    'learning_rate': {'typical_range': [0.1, 0.5]},
                    'temperature': {'typical_range': [0.5, 2.0]}
                },
                'components': ['prediction_error', 'value_update'],
                'requirements': ['needs_learning_rate']
            },
            'working_memory': {
                'description': 'Temporary storage and manipulation of information',
                'base_equations': ['WM(t+1) = γWM(t) + (1-γ)Input(t)'],
                'parameters': {
                    'decay_rate': {'typical_range': [0.1, 0.9]},
                    'capacity': {'typical_range': [4, 7]}
                },
                'requirements': ['needs_decay_rate']
            },
            'prediction_error': {
                'description': 'Difference between expected and actual outcomes',
                'base_equations': ['PE(t) = R(t) - Q(t)'],
                'requirements': []
            }
        }
        
        # Add mechanism nodes
        for mechanism, data in mechanisms.items():
            self.graph.add_node(mechanism, **data)
            logger.debug(f"Added mechanism: {mechanism}")
        
        # Add relationships
        relationships = [
            ('reinforcement_learning', 'prediction_error', {'type': 'contains', 'weight': 1.0}),
            ('working_memory', 'prediction_error', {'type': 'can_use', 'weight': 0.5})
        ]
        
        for source, target, attrs in relationships:
            self.graph.add_edge(source, target, **attrs)
            logger.debug(f"Added relationship: {source} -> {target}")