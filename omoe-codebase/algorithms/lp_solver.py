import gurobipy as gp
from gurobipy import GRB
import itertools
from algorithms.maj_voting import compute_probability_products
from config.variables import *
from algorithms.mc_algos import monte_carlo_simulation


def solve_lp(expert_competencies, quota=None, p=None, max_weights=None, theta_lb=THETA_LB, epsilon=LP_EPSILON, verbose_bool=True):

    N_experts = len(expert_competencies)  # Number of bits in each binary vector
    n = 2**N_experts  # Number of binary vectors (2^N)
    x = list(itertools.product([0, 1], repeat=N_experts))  # Generate all 2^N binary vectors of length N
    x = [list(vec) for vec in x] # Convert x to a list of lists for easier indexing

    if p is None:
        p = compute_probability_products(x, expert_competencies)

    if max_weights is None:
        max_weights = N_experts
    
    if quota is None:
        quota = 0.5 * max_weights
    
    # Create the Gurobi model
    model = gp.Model("Maximize_p_z")
    # model.Params.NumericFocus = 3
    # model.Params.FeasibilityTol = 1e-9
    # model.Params.IntFeasTol = 1e-9

    # Define the decision variables
    theta = model.addVars(max_weights, lb=theta_lb, name="theta")  # theta_i >= 0
    z = model.addVars(n, vtype=GRB.BINARY, name="z")  # z_i is binary

    # Objective function: Maximize sum(p_i * z_i)
    model.setObjective(gp.quicksum(p[i] * z[i] for i in range(n)), GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(theta[j] for j in range(N_experts)) <= max_weights)

    # Constraints
    for i in range(n):
        # Define the condition: x_i . theta >= 0.5 * sum(theta) + epsilon
        # condition = gp.quicksum(x[i][j] * theta[j] for j in range(N)) >= 0.5 * gp.quicksum(theta[j] for j in range(N))
        condition = gp.quicksum(x[i][j] * theta[j] for j in range(N_experts)) - quota >= epsilon
        
        # Indicator constraint: z_i = 1 if and only if condition is true
        model.addGenConstrIndicator(
            z[i], 1, condition
        )

        model.addGenConstrIndicator(
            z[i], 0, gp.quicksum(x[i][j] * theta[j] for j in range(N_experts)) - quota <= epsilon
        )

    # Enforce complement constraint: z_i + z_j <= 1 for complements x_i and x_j
    for i in range(n):
        x_i = x[i]
        x_j = [1 - val for val in x_i]  # Complement of x_i
        j = x.index(x_j)  # Find the index of the complement x_j
        model.addConstr(z[i] + z[j] <= 1, f"complement_{i}_{j}")

    # Optimize the model
    if not verbose_bool:
        model.setParam('OutputFlag', 0)
    
    model.optimize()

    return model, theta, z

def solve_lp_weights(expert_competencies):
    model, theta, _ = solve_lp(expert_competencies, verbose_bool=False) # model is needed
    # Extract the optimal weights
    optimal_weights = [theta[j].X for j in range(len(theta))]
    # optimal_weights.reverse()
    return optimal_weights

def lp_printout(N_experts, model, theta, z):

    n = 2**N_experts  # Number of binary vectors (2^N)
    x = list(itertools.product([0, 1], repeat=N_experts))  # Generate all 2^N binary vectors of length N
    x = [list(vec) for vec in x] # Convert x to a list of lists for easier indexing

    z_count = 0
    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Optimal value of the objective function:", model.objVal)

        print("\nOptimal values of theta:")
        for j in range(N_experts):
            print(f"theta_{j} =", theta[j].X)

        print("\nOptimal values of z:")
        for i in range(n):
            print(f"z_{i} (for x_{i} = {x[i]}) =", int(z[i].X), ", sum(x_i * theta) =", sum(x[i][j] * theta[j].X for j in range(N_experts)))
            z_count += z[i].X
    else:
        print("No optimal solution found. Model status:", model.status)

    print("Total number of selected scenarios:", z_count)
    print("Total number of scenarios:", n)

    return True

if __name__ == "__main__":
    expert_competencies = EXPERT_P
    N_experts = len(expert_competencies)  # Number of bits in each binary vector
    
    model, theta, z = solve_lp(expert_competencies)
    lp_printout(N_experts, model, theta, z)

    theta_weights = solve_lp_weights(expert_competencies)


    #########
    N = len(expert_competencies)  # Number of experts
    C = 5  # Total number of categories
    C_prime = 1  # Number of correct categories
    p_list = expert_competencies  # Probabilities of each expert being correct
    beta_list = theta_weights  # Weights of each expert

    # Run the Monte Carlo simulation
    probability = monte_carlo_simulation(N, C, C_prime, p_list, beta_list, num_trials=100000)
    print(f"Estimated MC probability of correct prediction: {probability:.4f}")