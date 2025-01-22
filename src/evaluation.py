# src/evaluation.py:

# - Tests how well models perform
# - Computes scores for each model

from typing import Dict, List, Tuple
import numpy as np
from .core import ModelState

class SimpleEvaluator:
    """Evaluates cognitive models on test data"""
    def __init__(self, test_data: Dict):
        self.data = test_data
        self.cached_scores = {}  

    def evaluate_model(self, state: ModelState) -> float:
        """
        Evaluate a model based on:
        1. Prediction negative log likelihood (NLL)
        2. Model complexity (penalty)
        3. Parameter reasonableness
        """
        equation_key = ''.join(state.equations)
        if equation_key in self.cached_scores:
            return self.cached_scores[equation_key]

        # Calculate different score components
        nll_score = self._compute_negative_log_likelihood(state)
        complexity_penalty = self._compute_complexity_penalty(state)
        parameter_penalty = self._compute_parameter_penalty(state)

        # Combine scores
        total_score = -nll_score - complexity_penalty - parameter_penalty
        
        # Cache and return
        self.cached_scores[equation_key] = total_score
        return total_score

    def _compute_negative_log_likelihood(self, state: ModelState) -> float:
        """Compute the negative log likelihood of the model's predictions given the data"""
        predictions = self._simulate_model_probabilities(state)
        actual_actions = self.data['actions']

        # Prevent log(0) by adding a small epsilon
        epsilon = 1e-10
        log_likelihood = np.sum(np.log(predictions[np.arange(len(actual_actions)), actual_actions] + epsilon))

        return -log_likelihood

    def _compute_complexity_penalty(self, state: ModelState) -> float:
        """Penalize complex models"""
        # Count operations in equation
        equation = state.equations[0]
        operations = equation.count('+') + equation.count('*') + equation.count('/')
        
        # Penalize based on number of operations
        return 0.1 * operations

    def _compute_parameter_penalty(self, state: ModelState) -> float:
        """Penalize unreasonable parameter values"""
        penalty = 0.0
        
        # Check each parameter is in reasonable range
        for param, value in state.parameters.items():
            if param == 'alpha' and (value < 0 or value > 1):
                penalty += 0.2
            elif param == 'temp' and (value <= 0 or value > 10):
                penalty += 0.2
            elif param == 'gamma' and (value < 0 or value > 1):
                penalty += 0.2
                
        return penalty

    def _simulate_model_probabilities(self, state: ModelState) -> np.ndarray:
        """
        Simulate model probabilities for actions
        For MVP: Simplified simulation using basic Q-learning
        """
        n_trials = len(self.data['timestamps'])
        q_values = np.zeros(2)  # Q-values for two actions
        probabilities = np.zeros((n_trials, 2))
        alpha = state.parameters.get('alpha', 0.1)
        temp = state.parameters.get('temp', 1.0)

        for t in range(n_trials):
            # Softmax probabilities
            exp_q = np.exp(q_values / temp)
            probabilities[t] = exp_q / np.sum(exp_q)

            # Upd Q-values based on reward
            reward = self.data['rewards'][t]
            chosen_action = self.data['actions'][t]
            q_values[chosen_action] += alpha * (reward - q_values[chosen_action])

        return probabilities

class ModelSimulator:
    """Simulates cognitive models to generate predictions"""
    @staticmethod
    def simulate_choices(state: ModelState, n_trials: int) -> Tuple[List[int], List[float]]:
        """
        Simulate choices and rewards for a given model
        Returns: (choices, rewards)
        """
        choices = []
        rewards = []
        q_values = np.zeros(2)  # Two-armed bandit
        
        for _ in range(n_trials):
            # Make choice using softmax
            temp = state.parameters.get('temp', 1.0)
            exp_q = np.exp(q_values / temp)
            probabilities = exp_q / np.sum(exp_q)
            choice = np.random.choice([0, 1], p=probabilities)
            
            # Generate reward
            reward_prob = 0.7 if choice == 1 else 0.3
            reward = float(np.random.random() < reward_prob)
            
            # Update Q-values
            alpha = state.parameters.get('alpha', 0.1)
            q_values[choice] += alpha * (reward - q_values[choice])
            
            choices.append(choice)
            rewards.append(reward)
            
        return choices, rewards
