# src/core.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import uuid
from .experiment import ExperimentInfo, generate_bandit_trials, generate_experiment
from .participant import ParticipantInfo, run_experiment

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

async def generate_test_data(n_trials: int = 100) -> Dict:
    """
    Generate synthetic two-armed bandit data for testing purposes.
    
    Args:
        n_trials (int): Number of trials to generate
        
    Returns:
        Dict containing:
            - timestamps: Array of trial numbers
            - actions: Array of choices made (0 or 1)
            - rewards: Array of rewards received
    """
    # Create experiment info object
    experiment_info = ExperimentInfo()
    experiment_info.id = "test"
    experiment_info.n_trials = n_trials
    experiment_info.p_init = (0.7, 0.3)  # Arm 0 has 70% reward probability, Arm 1 has 30%
    experiment_info.sigma = (0.02, 0.02)
    experiment_info.hazard_rate = 0.05

    # Generate trial sequence
    trial_sequence = generate_bandit_trials(experiment_info=experiment_info)

    # Create participant info
    participant_info = ParticipantInfo()
    participant_info.id = "test"
    participant_info.age = 18
    participant_info.gender = "male"

    # Run the experiment and get results
    df = await run_experiment(trial_sequence, participant_info, experiment_info)
    
    # Transform the results into the expected format
    data = {
        'timestamps': np.arange(n_trials),
        'actions': df['choice'].values,  # Convert from 1-based to 0-based indexing if needed
        'rewards': df['reward'].values
    }
    
    return data

class SearchNode:
    """Represents a node in the Monte Carlo Tree Search"""
    def __init__(self, state: ModelState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        
    def add_child(self, child_state: ModelState) -> 'SearchNode':
        """Add a child node with the given state"""
        child = SearchNode(child_state, self)
        self.children.append(child)
        return child
        
    def update(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.value += (result - self.value) / self.visits

    def get_ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate the UCB score for this node"""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.value
            
        exploitation = self.value
        exploration = exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

def backpropagate(node: SearchNode, value: float):
    """Backpropagate the result through the tree"""
    current = node
    while current is not None:
        current.update(value)
        current = current.parent