import numpy as np

def monte_carlo_simulation(N, C, C_prime, p_list, beta_list, num_trials=10000):
    """
    Monte Carlo simulation for mixture of N experts with majority voting.

    Parameters:
    - N: Number of experts.
    - C: Total number of categories.
    - C_prime: Number of correct categories.
    - p_list: List of probabilities for each expert to choose the correct answer.
    - beta_list: List of weights for each expert.
    - num_trials: Number of Monte Carlo trials to run.

    Returns:
    - Probability that the committee arrives at the correct prediction.
    """
    correct_predictions = 0

    for _ in range(num_trials):
        # Initialize a list to store the votes for each category
        category_votes = np.zeros(C)

        for n in range(N):
            # Determine if the expert chooses the correct category
            if np.random.rand() < p_list[n]:
                # Expert chooses one of the correct categories
                chosen_category = np.random.randint(0, C_prime)
            else:
                # Expert chooses one of the incorrect categories
                chosen_category = np.random.randint(C_prime, C)

            # Add the expert's weight to the chosen category
            category_votes[chosen_category] += beta_list[n]

        # Determine the committee's prediction by weighted majority voting
        total_weight = sum(beta_list)
        majority_threshold = total_weight / 2

        # Check if any of the correct categories received more than 50% of the total weight
        for c in range(C_prime):
            if category_votes[c] > majority_threshold:
                correct_predictions += 1
                break

    # Estimate the probability of correct prediction
    probability_correct = correct_predictions / num_trials
    return probability_correct
