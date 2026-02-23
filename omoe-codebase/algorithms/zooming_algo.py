import numpy as np
from scipy.spatial.distance import euclidean
from algorithms.maj_voting import empirical_is_majority_win
from algorithms.moe_base import PseudoExpert
from config.variables import *
from scipy.spatial import KDTree
from scipy.stats.qmc import LatinHypercube
from algorithms.maj_voting import *
from utils.helper_functions import expert_printout
from algorithms.lp_solver import solve_lp_weights
from utils.helper_functions import hoeffding_bounds
from llm.llm_moe_base import LLMExpert
from random import random
class ZoomingBandit:
    def __init__(self, expert_p, rho=15.0, phase_length_multiplier=2, max_value=None, 
                 max_strategies=10, prune_frequency=60, llm_expert_set=[], side_ordinal_info=True):
        self.expert_p = expert_p
        self.D = len(expert_p)
        self.rho = rho
        self.phase_length_multiplier = phase_length_multiplier
        self.max_value = self.D if max_value is None else max_value
        self.max_strategies = max_strategies  # Maximum number of strategies to keep
        self.prune_frequency = prune_frequency  # How often to check for pruning
        self.llm_expert_set = llm_expert_set
        self.side_ordinal_info = side_ordinal_info
        self.reset()

    def reset(self):
        self.active_strategies = {}
        self.phase = 1
        self.rounds_in_phase = 0
        self.total_rounds = 0
        self.prune_counter = 0  # Counter for pruning frequency

    def get_confidence_radius(self, n, phase):
        return np.sqrt(8 * phase / (2 + n))

    def is_covered(self, theta, active_strategies):
        if not active_strategies:
            return False
        thetas = np.array(list(active_strategies.keys()))
        radii = np.array([data['r'] for data in active_strategies.values()])
        tree = KDTree(thetas)
        dist, idx = tree.query(np.array(theta), k=1)
        return dist <= radii[idx]

    def select_strategy(self):
        best_theta = max(
            self.active_strategies.items(),
            key=lambda item: item[1]['mu'] + 2 * item[1]['r']
        )[0]
        return best_theta

    def generate_bounded_theta(self):
        sampler = LatinHypercube(d=self.D)
        sample = sampler.random(n=1)[0] * self.max_value
        return tuple(sample)
    
    def reward_func(self, theta, empirical_p=None): 
        if empirical_p is None:
            empirical_p = self.expert_p
        return empirical_is_majority_win(empirical_p, weights=theta)
    
    def update(self, theta, reward):
        data = self.active_strategies[theta]
        data['n'] += 1
        data['mu'] = (data['mu'] * (data['n'] - 1) + reward) / data['n']
        data['r'] = self.get_confidence_radius(data['n'], self.phase)

    def prune_strategies(self):
        """Remove poorly performing strategies when we exceed the maximum allowed"""
        if len(self.active_strategies) <= self.max_strategies:
            return
        
        # Sort strategies by their upper confidence bound (mu + 2r)
        sorted_strategies = sorted(
            self.active_strategies.items(),
            key=lambda item: -(item[1]['mu'] + 2 * item[1]['r'])  # Descending order
        )
        
        # Keep only the top-performing strategies
        self.active_strategies = dict(sorted_strategies[:self.max_strategies])
        
        # Rebuild the KDTree if needed (called automatically in is_covered)
        print(f"Pruned strategies. Current count: {len(self.active_strategies)}")

    def step(self):
        # Check if we should prune before proceeding
        self.prune_counter += 1
        if self.prune_counter >= self.prune_frequency:
            self.prune_strategies()
            self.prune_counter = 0

        if self.rounds_in_phase >= self.phase_length_multiplier ** self.phase:
            self.phase += 1
            self.rounds_in_phase = 0
            for theta in self.active_strategies:
                self.active_strategies[theta]['r'] = self.get_confidence_radius(
                    self.active_strategies[theta]['n'], self.phase
                )

        theta = self.generate_bounded_theta()
        attempts = 0
        max_attempts = 100  # Prevent infinite loops
        while self.is_covered(theta, self.active_strategies) and attempts < max_attempts:
            theta = self.generate_bounded_theta()
            attempts += 1

        if attempts == max_attempts:
            # If we can't find an uncovered point, select the best existing strategy
            theta = self.select_strategy()
        else:
            self.active_strategies[tuple(theta)] = {
                'n': 0, 
                'mu': 0, 
                'r': self.get_confidence_radius(0, self.phase)
            }

        theta = self.select_strategy()
        
        if self.side_ordinal_info:
            theta_prox = tuple(sorted(list(theta)))
        else:
            theta_prox = theta

        if len(self.llm_expert_set) > 0 and type(self.llm_expert_set[0]) == LLMExpert:
            expert_p = [e.play() for e in self.llm_expert_set]
            reward = self.reward_func(theta_prox, expert_p)
        else:
            reward = self.reward_func(theta)

        if reward:
            pass
        
        self.update(theta, reward)

        self.rounds_in_phase += 1
        self.total_rounds += 1
        return theta, reward

def zooming_bandit(experts, 
                   time_steps=T, 
                   t_burn_in=T_BURN_IN, 
                   opt_solution=None, 
                   llm_expert_set=[],
                   side_ordinal_info=False):
    
    current_experts = experts.copy()

    expert_p = [expert.p for expert in experts]
    if opt_solution is None:
        theta_weights = solve_lp_weights(expert_p)
        opt_solution = p_maj_vote(expert_p, weights=theta_weights)
        _, opt_solution_see = find_opt_egal_voting([expert.p for expert in experts])
    
    success_count = 0
    empirical_reward_tracker = []
    regret_tracker = []
    ucb_tracker = []
    lcb_tracker = []

    bandit = ZoomingBandit(expert_p, llm_expert_set=llm_expert_set, side_ordinal_info=side_ordinal_info)

    # Continuously check for elimination opportunities
    theta_weights_est = list(np.ones(len(expert_p)))

    for t in range(1, time_steps+1):
        
        if t <= t_burn_in:
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
        
        if t > t_burn_in:

            # Display current status
            estimates = []
            for ex in current_experts:
                est = ex.get_estimate()
                lcb = ex.get_lcb()
                ucb = ex.get_ucb()
                estimates.append(f"E{ex.id}: {est:.2f} ({lcb:.2f},{ucb:.2f})")
            
            print(f"{t:<6} {str([ex.id for ex in current_experts]):<20} "
                f"E{ex.id:<9} {str(outcome):<8} {' | '.join(estimates)}")
            
            expert_p = [e.get_estimate() for e in current_experts]

            theta_weights_est, is_success = bandit.step()

            if is_success:
                success_count += 1

            p_maj_lcb, p_maj_ucb = hoeffding_bounds(success_count, t)

            ucb_tracker.append(p_maj_ucb)
            lcb_tracker.append(p_maj_lcb)

            empirical_reward = success_count/t
            empirical_regret = opt_solution - empirical_reward
            
            empirical_reward_tracker.append(empirical_reward)
            regret_tracker.append(empirical_regret)
            
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
            
            expert_printout(current_experts)
    
    return_dic = {
        "bandit": bandit,
        "current_experts": current_experts,
        "empirical_reward_tracker": empirical_reward_tracker,
        "regret_tracker": regret_tracker,
        "ucb_tracker": ucb_tracker,
        "lcb_tracker": lcb_tracker,
        "opt_solution": opt_solution,
        "success_count": success_count,
        "original_experts": experts,
        "theta_weights_est": theta_weights_est,
    }

    return return_dic

if __name__ == "__main__":
    # Initialize and run
    current_experts = [PseudoExpert(perf, idx) for idx, perf in enumerate(EXPERT_P)]
    zooming_dic = zooming_bandit(current_experts)

    print("theta_weights_est:", zooming_dic['theta_weights_est'])


