---
layout: single
tags: Deep learning
category: Reinforcement_learning
---

http://hunch.net/~rwil
http://aka.ms/personalizer
http://vowpalwabbit.org

# Summary
The k-armed bandit is made up of a few questions:
- What is the Expected Value of all available actions? Aka what is the distribution of rewards for each possible action?
- What choice of distribution will we use to model each actions distribution? We covered sample average and recency weighted exponential decay
- What value maximizing approach to take? We covered greedy, epsilon-greedy, and upper-confidence bound

The choice of whether to explore or whether to take the action with the highest expected value is the crux of the problem. Exploration leads to better information while exploitation leads to higher immediate returns.

## RL Glossary

_sample-average_: Expected value of action A at time t is sum of all previous values from action a over the times A has been chosen. Computationally, Q_t = Q_t-1 + 1/n * (A_t-Q_t-1)

_greedy_: Choosing the action with the highest expected value every time

_incremental update rule_: new_estimate = old_estimate + step_size * (target - old_estimate)

_non-stationary bandit problem_: distribtuion of rewards chagnes with time. If alpha is constant, most recent rewards have more weight than old rewards

_Decaying past rewards_: Q_n = (1-a)^n * Q_0 + alpha * sum[(1-a)^(n-) * R_i] for i in [1,n]

_exploration-exploitation tradeoff_: Exploration allows improving knowledge of each action but lowers immediate returns, eploitation increases immediate rewards (you're grabbing what you think will have the highest returns), but lower long term tradeoffs as you won't fully discover the underlying distributions per bandit

_epsilon greedy_: Each cycle take the highest expected reward. Every epsilon times, randomly choose from the non-highest expected rewards

_optimistic initial values_: encourages early exploration, but poor for non-stationary problems (since distros of other bandits can change but we could potentially not notice)

_UCB Action Selection_: Choose reward with highest upper bound; A = argmax[Q(a) + c * (ln(t)/(N(a)))^.5]
