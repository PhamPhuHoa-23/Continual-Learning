from algorithms.maj_voting import find_opt_egal_voting
import numpy as np
import math
from config.variables import ALPHA_DISPLAY
from llm.llm_moe_base import LLMExpert

def expert_printout(experts):
    print(f"{'Expert':<20} {'True p':<10} {'Trials':<10} {'Successes':<10} {'Estimate':<10} {'LCB':<10} {'UCB':<10}")
    for expert in experts:
        est = expert.get_estimate()
        lcb = expert.get_lcb()
        ucb = expert.get_ucb()
        if type(expert) == LLMExpert:
            print(f"{expert.llm_name:<20} {expert.p:<10.2f} "
                f"{expert.trials:<10} {expert.successes:<10} "
                f"{est:<10.2f} {lcb:<10.2f} {ucb:<10.2f}")
        else:
            print(f"Expert {expert.id:<9} {expert.p:<10.2f} "
                f"{expert.trials:<10} {expert.successes:<10} "
                f"{est:<10.2f} {lcb:<10.2f} {ucb:<10.2f}")

def get_config_for_summary(see_dic,):
    original_experts = see_dic['original_experts']
    expert_p = [expert.p for expert in original_experts]
    best_sublist, max_prob = find_opt_egal_voting(expert_p)
    n_experts = len(expert_p)
    n_best = len(best_sublist)
    opt_reward = max_prob
    min_gap = np.min(np.diff(expert_p))

    time_steps = len(see_dic['empirical_reward_tracker'])

    return {
        'n_experts': n_experts,
        'n_best': n_best,
        'opt_reward': opt_reward,
        'min_gap': min_gap,
        'time_steps': time_steps,
        'final_cum_regret': see_dic['regret_tracker'][-1],
        'final_empirical_reward': see_dic['empirical_reward_tracker'][-1],
        'final_ucb': see_dic['ucb_tracker'][-1],
        'final_lcb': see_dic['lcb_tracker'][-1]
    }

def hoeffding_bounds(successes, trials, alpha=ALPHA_DISPLAY):
    if trials == 0:
        return (0.0, 1.0)  # Default bounds if no data
    
    p_hat = successes / trials
    margin = math.sqrt(-math.log(alpha / 2) / (2 * trials))
    
    lcb = max(0.0, p_hat - margin)  # Clamped to [0, 1]
    ucb = min(1.0, p_hat + margin)  # Clamped to [0, 1]
    
    return lcb, ucb

def get_regret_data_to_plot(see_dic):
    
    original_experts = see_dic['original_experts']
    current_experts = see_dic['current_experts']
    empirical_reward_tracker = see_dic['empirical_reward_tracker']
    regret_tracker = see_dic['regret_tracker']
    ucb_tracker = see_dic['ucb_tracker']
    lcb_tracker = see_dic['lcb_tracker']

    _, opt_solution = find_opt_egal_voting([expert.p for expert in original_experts])

    simple_regret_series = [opt_solution - x for x in empirical_reward_tracker]
    simple_regret_ucb = [opt_solution - x for x in ucb_tracker]
    simple_regret_lcb = [opt_solution - x for x in lcb_tracker]

    alg_cum_regret = np.cumsum(simple_regret_series)
    alg_cum_regret_ucb = np.cumsum(simple_regret_ucb)
    alg_cum_regret_lcb = np.cumsum(simple_regret_lcb)

    return alg_cum_regret, alg_cum_regret_lcb, alg_cum_regret_ucb