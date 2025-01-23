# src/core.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import uuid
from experiment import generate_bandit_trials, generate_experiment
from participant import execute_experiment_simple

@dataclass
class ModelState:
    """Represents a state in our cognitive model search"""
    equations: List[str]  # The mathematical equations of the model
    parameters: Dict[str, float]  # Model parameters (e.g., learning rate)
    score: Optional[float] = None  # How well the model performs
    visits: int = 0  # For MCTS tracking
    value: float = 0.0  # For MCTS tracking
    id: str = ''  # Unique identifier for graph operations
    
    def __post_init__(self):
        """Initialize unique ID if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def __hash__(self):
        """Make ModelState hashable using ID"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality comparison using ID"""
        if not isinstance(other, ModelState):
            return False
        return self.id == other.id
    
    def copy(self):
        """Create a deep copy of the state"""
        return ModelState(
            equations=self.equations.copy(),
            parameters=self.parameters.copy(),
            score=self.score,
            visits=self.visits,
            value=self.value,
            id=str(uuid.uuid4())
        )

def generate_test_data(n_trials: int = 100) -> Dict:
    """Generate synthetic two-armed bandit data"""
    
    reward_probabilities = (0.7, 0.3)  # Arm 0 has 70% reward probability, Arm 1 has 30%
    trial_sequence = generate_bandit_trials(n_trials, reward_probabilities)

    # Generate an experiment with the trial sequence
    experiment = generate_experiment(trial_sequence)

    # Run the experiment on the synthetic participant
    rewards, choices = execute_experiment_simple(trial_sequence, "18", "male")
    
    data = {
        'timestamps': np.arange(n_trials),
        'actions': choices,
        'rewards': rewards
    }
    
    return data