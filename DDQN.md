## Important notes based on the DDQN algorithm in our paper:

### Line 5:
**"Select an action *a* randomly with probability *ε* or *a = arg max_{a∈A} Q(s_k, a; θ)* with probability *(1 − ε)*."**

This line describes a decision-making process commonly used in reinforcement learning, specifically in the context of an **ε-greedy strategy**. Here’s what it means step-by-step:

1. **Purpose**: The algorithm is deciding which action (*a*) to take in the current state (*s_k*) during iteration *k*. It balances **exploration** (trying new actions randomly) and **exploitation** (choosing the best-known action based on current knowledge).

2. **Two Options**:
   - **Random Action**: With probability *ε* (a small value between 0 and 1), the action *a* is chosen randomly. This encourages exploration of the environment.
   - **Greedy Action**: With probability *(1 − ε)*, the action *a* is chosen as the one that maximizes the action-value function *Q(s_k, a; θ)*. This is the exploitation step, where the algorithm picks the action it currently believes is the best.

3. **Key Symbols**:
   - *a*: The action to be selected.
   - *ε* (epsilon): A parameter (typically small, e.g., 0.1) that controls the trade-off between exploration and exploitation. It represents the probability of choosing a random action.
   - *arg max_{a∈A}*: This mathematical notation means "the action *a* from the set of possible actions *A* that maximizes the following function." In this case, it’s finding the action that gives the highest value of *Q*.
   - *Q(s_k, a; θ)*: The action-value function, which estimates the expected future reward for taking action *a* in state *s_k*, given the current model parameters *θ*. Here:
     - *s_k*: The state at iteration *k*.
     - *a*: The action being evaluated.
     - *θ* (theta): The weights or parameters of the model (e.g., a neural network) that define the *Q*-function.
   - *(1 − ε)*: The probability of choosing the greedy (optimal) action instead of a random one.

### Explanation of the Symbol *ε* (Epsilon):
- *ε* is a Greek letter commonly used in reinforcement learning to denote the exploration rate.
- It determines how often the algorithm explores randomly versus exploiting its current knowledge.
- For example, if *ε = 0.1*, there’s a 10% chance of picking a random action and a 90% chance of picking the action with the highest *Q*-value.

### What Happens in Line 5:
The algorithm flips a biased coin:
- If the result falls within the *ε* probability (e.g., 10%), it picks a random action *a* from the set of possible actions *A*.
- Otherwise, with probability *(1 − ε)* (e.g., 90%), it computes the *Q*-value for all possible actions in the current state *s_k* using the model parameters *θ*, and selects the action *a* that gives the highest *Q*-value.

### In Simple Terms:
Line 5 is like deciding whether to "try something new" (random action) or "go with what you know works best" (maximize *Q*). The symbol *ε* controls how often you take a chance versus playing it safe.

Mini-batch size means:
Taking samples from the memory to train the model

Batch size means: taking samples from the pytorch to train the model
