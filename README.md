# Soft Actor Critic Implementation of Rational Speech Act Model

This repository demonstrates how the Rational Speech Act (RSA) model can be framed as a Reinforcement Learning problem, specifically using the Soft Actor Critic (SAC) algorithm. The implementation shows that RSA's speaker-listener dynamics can be modeled using SAC's actor-critic framework.

## Table of Contents
1. [Introduction](#introduction)
2. [Rational Speech Act](#rational-speech-act)
3. [Links between RL and RSA](#links-between-rl-and-rsa)
4. [Data Generation](#data-generation)
5. [Soft Actor Critic (SAC) Implementation](#soft-actor-critic-sac-implementation)
6. [Environment](#environment)
7. [Training](#training)
8. [Results and Analysis](#results-and-analysis)
9. [Discussion](#discussion)

## Introduction

The Rational Speech Act (RSA) model is a framework for understanding language pragmatics. This project reimagines RSA as a reinforcement learning problem, implementing it using the Soft Actor Critic algorithm. In this analogy:
- The speaker is modeled as the actor
- The listener is modeled as the critic

The rational speech act has initially been introduced by [Frank and Goodman (2012)](https://www.mitpressjournals.org/doi/abs/10.1162/jocn_a_00232) and has been widely used in the field of computational linguistics but has few applications. Soft actor critic is a reinforcement learning algorithm introduced by [Haarnoja et al. (2018)](https://arxiv.org/abs/1801.01290) building on top of the actor-critic but introducing entropy regularization that we found in Rational Speech Act Model.

The entropy term implicitely present in RSA has been first highlighted by [Noga Zaslavsky and Jennifer Hu and Roger P. Levy](https://arxiv.org/abs/2005.06641) but no explication have been given on the relevance of this term.

## Aim of the Project

**This project do not aim at implementing a cleaner or more efficient version of RSA**. 

**This project aims at showing the parallel between Reinforcement Learning and Rational Speech Act Model**.

The provided code only aim at showing the parallel between Reinforcement Learning and Rational Speech Act Model. The code is not compute efficient and does not perfectly reflect classic Soft Actor Critic implementation.


## Rational Speech Act

Rational Speech Act (RSA) theory is a computational model of pragmatic reasoning in language use. It posits that speakers and listeners reason about each other's mental states to communicate effectively. In RSA, speakers choose utterances (=messages) that are informative and relevant, considering what a rational listener would infer. Listeners, in turn, interpret utterances by reasoning about what a rational speaker would say. This recursive reasoning process helps explain how people derive rich meanings from often ambiguous language.    

![image-4](https://github.com/basileplus/sac-rsa/assets/115778954/c947323f-ca5e-4e24-8371-600614753a21)
*Figure 1: Rational Speech Act model*

## Links between RL and RSA

To better understand the SAC-RSA model and its relationship to traditional RL and RSA frameworks, we provide several visualizations:

### Classic RL Framework

![image-1](https://github.com/basileplus/sac-rsa/assets/115778954/70d3f572-a51f-432d-8ddf-2428e4e5e06a)
*Figure 2: Traditional Reinforcement Learning setup with Agent and Environment*

In a classic RL framework, an agent learn a policy $\pi_{\theta_A}(a \mid s)$ to maximize the expected reward $R(s,a)$. The agent interacts with an environment, receiving rewards and updating its policy based on the observed states and actions.

### SAC-RSA Framework
![image-2](https://github.com/basileplus/sac-rsa/assets/115778954/ab673ea5-2f78-44b5-8887-289c94dfb945)
*Figure 3: Detailed view of SAC-RSA model, showing Speaker as Actor and Listener as Critic*

In SAC, the agent (the speaker) learns a policy $\pi_{\theta_A}(a \mid s)$ to minimize the loss function $\mathcal{L}_{\text{actor}}$. The critic (the listener) learns a value function $Q_{\theta_C}(a,s)$ to minimize the loss function $\mathcal{L}_{\text{critic}}$. The actor uses the value learned by the critic to update its policy. 

In SAC-RSA model, we do not really learn from an environment (which could be an unknown listener for instance). We fake a Reinforcement Learning setup to show the parallel between RSA and RL. The environment is actually useless because it is an exact copy of the critic : the environment do not provide the agent any additional information.

However it is relevant to understand what the agent really learns. One can see the environment as a mirror in front of which the speaker would refine its speech. In the other hand the critic is an estimation of how the listener would receive the speech. 

### Implementing a different environment

We could also define an unknown environment for the speaker to learn, in this case Reinforcement Learning would be more relevant. The reward sent by the evnironment would be used by the critic to learn the value function $Q_{\theta_C}(a,s)$.

### Comparison: Traditional RSA vs SAC-RSA

| **Aspect**          | **Traditional RSA** | **SAC-RSA**                          |
|---------------------|---------------------|--------------------------------------|
| **Speaker Model**   | $S(u \mid m) \propto \exp(\alpha \cdot \log(L(m \mid u)))$ | $\pi_{\theta_A}(a \mid s)$ where $a$ is utterance, $s$ is meaning |
| **Listener Model**  | $L(m \mid u) \propto S(u \mid m) \cdot P(m)$ | $Q_{\theta_C}(a, s)$ |



#### Key Differences

1. **Explicit Reward Signal**: SAC-RSA has an explicit reward signal vs implicit optimization in Traditional RSA.
2. **Explicit Entropy Regularization**: SAC-RSA explecitely includes entropy regularization, encouraging exploration. This is hidden in classic RSA.
3. **Explicit estimated listener**: in SAC-RSA it is clear that the speaker (actor) learns from an estimated listener (critic)
4. **Continuous Learning Process**: SAC-RSA involves a continuous learning process via Stochastic Gradient Descent vs a fixed-point solution in Traditional RSA.

## Data Generation

We start by generating synthetic data:
- A truth matrix $T\in M_{U\times M}(\mathbb{R})$ composed of 0 and 1
- A prior knowledge matrix $P ∈ M_M(R)$

Where $U$ is the number of utterances and $M$ is the number of meanings.

```python
# Example truth matrix and prior knowledge
T = tensor([[0., 1., 1., 1.],
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [1., 1., 0., 1.],
            [1., 0., 0., 1.]])

P = tensor([0.2000, 0.2000, 0.2000, 0.2000])
```

## Soft Actor Critic (SAC) Implementation

The SAC algorithm is implemented with two main components:

### Actor (Speaker)

The actor represents the speaker and is responsible for:
- Computing probability distribution $π_{θ_A}(a|s)$
- Sampling actions given a state and policy
- Computing its loss function

```python
class Actor:
    def __init__(self, theta_init, lr=0.01):
        ...
    def update_pol(self):
        ... 
    def sample_action(self, action):
        ...    
    def update_loss(self, Q, Prior):
        entropies = torch.stack([dist.entropy() for dist in self.distris])
        entropy_term = torch.sum(Prior*entropies)
        loss = -torch.sum(Prior * (torch.sum(self.probs * Q, dim=0))) - entropy_term
        self.loss = loss
    def update_theta(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
```
Parameters are learned using stochastic gradient descent on the actor loss function defined as :
$$
\mathcal{L}_{\text{actor}} = -\sum_{s}P(s)\left(\sum_{a}\pi_{\theta_A}(a|s)Q_{\theta_C}(a,s) + H_s(\pi_{\theta_A})\right)
$$

### Critic (Listener)

The critic represents the listener and is responsible for:
- Computing value function $Q_{θC}(a,s)$
- Computing its loss function
- Updating its parameters $θ_C$

```python
class Critic:
    def __init__(self, theta_init, alpha=1, lr=0.01):
        ...
    def update_Q(self):
        ...
    def update_loss(self, probs, P):
        ...
    def update_theta(self,lr=0.01):
        self.optimizer = torch.optim.SGD([self.theta], lr=lr)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    
    # Methods for updating Q-values, computing loss, and updating parameters
    ...
```

Parameters are learned using stochastic gradient descent on the critic loss function defined as :
$$
\mathcal{L}_{\text{critic}} = -\sum_{s}P(s)\left(\sum_{a}\pi_{\theta_A}(a|s)Q_{\theta_C}(a,s) + H_s(\pi_{\theta_A})\right)
$$ 
### Soft Actor Critic

We added a SAC class to handle the update the actor and critic:

```python
class SAC:
    def __init__(self, actor, critic, env, config):
        ...
    
    def update(self, P):
        self.actor.update_pol()
        self.critic.update_pol()
        self.critic.update_Q()
        self.actor.update_loss(self.critic.Q.detach(), P)
        self.critic.update_loss(self.actor.probs.detach(), P)
        self.actor.update_theta()
        self.critic.update_theta()
        self.critic.update_pol()
```
May have notice that $\mathcal{L}_{\text{actor}}$ and $\mathcal{L}_{\text{critic}}$ are the same. We chose to use ``.detach()`` method of pytorch to update actor policy then critic policy. We could also have done a single gradient descent on a parameter $\theta = (\theta_A, \theta_C)$, but the Soft Actor Critic would have been less explicit.
## Environment

An environment class is implemented to simulate the interaction between the speaker and listener. In the present case it is not really useful but is included to understand the parallel which can be drawn between RL and RSA.

The environment would for instance be useful if we wanted to :
- learn from an unknown listener
- add noise to the communication

```python
class Environment:
    def __init__(self, P, T):
        self.P = P
        self.T = T
        self.theta_A, self.theta_C = self.init_thetas(self.T)
    
    # Methods for initializing parameters, getting rewards, and resetting environment
    ...
```

## Training

The training loop runs for a specified number of episodes, updating the actor and critic at each step:

```python
def train(agent, env, config):
    a_losses, c_losses, a_entropies = [], [], []
    for episode in range(config.num_episodes):
        state = env.reset()
        next_state, reward = env.step(listener=agent.critic, action=None, state=None)
        agent.update(env.P)
        # Track losses and entropies
        ...
    return a_losses, c_losses, a_entropies
```

## Results and Analysis

Our implementation of the Rational Speech Act (RSA) model using Soft Actor Critic (SAC) shows comparable performance to the traditional RSA approach. Here's a summary of key findings:

![image](https://github.com/basileplus/sac-rsa/assets/115778954/d28ee542-c358-43b4-8536-f6e455060520)

*Figure 4: Actor Loss, Critic Loss, and Entropy over training episodes*



### Listener Model Comparison

Both SAC and traditional RSA produce similar listener models :
Example (u1 interpretation):
```
RSA: m1 (0%), m2 (10%), m3 (59%), m4 (31%)
SAC: m1 (0%), m2 (27%), m3 (44%), m4 (29  %)
```

### Speaker Model Comparison

The speaker models also show similarities:


Example (m2 utterance probabilities):
```
RSA: u1 (8%), u2 (80%), u3 (0%), u4 (12%), u5 (0%)
SAC: u1 (29%), u2 (40%), u3 (0%), u4 (30%), u5 (0%)
```
Those results could be adjusted by changing the entropy regularization parameter $\alpha$, as we can see in SAC we seem to encourage uniform distribution which have high entropy, so by lowering $\alpha$ we would give less importance to entropy.

### Interpretation

These results demonstrate the viability of using reinforcement learning techniques to model pragmatic language understanding.
These results demonstrate how the SAC framework can be used to model the RSA dynamics, providing insights into the speaker-listener interaction in language pragmatics.



## Discussion

Learning from Gradient Descent is way longer than applying classic RSA iteration.
This project only aims at showing the parallel between Reinforcement Learning and Rational Speech Act Model. It is a way to understand the RSA model in a different way and to higlight some important facts :
- The listener can easily be seen as an approximation by the speaker of the listener with SAC-RSA. From this point of view the listener is nothing more than a normalization of the speaker probability distribution.
- We can easily model any environment we want with SAC-RSA adding noise, communication limitations, multiple listeners, etc.
- The entropy regularization is explicit in SAC-RSA and is used to encourage exploration, whereas in classis RSA it can be challenging to explain entropy maximization relevance
- Expansion of the model to more complex environments and tasks is easier. We can learn continuous policies, transmit multiple utterances messages, use different algorithms to learn parameters etc.

The provided code only aim at showing the parallel between Reinforcement Learning and Rational Speech Act Model. The code is not compute efficient and does not perfectly reflect classic Soft Actor Critic implementation. It must be seen as a proof of concept.

## References

1. [Frank and Goodman (2012)](https://www.mitpressjournals.org/doi/abs/10.1162/jocn_a_00232)
2. [Haarnoja et al. (2018)](https://arxiv.org/abs/1801.01290)
3. [Noga Zaslavsky and Jennifer Hu and Roger P. Levy](https://arxiv.org/abs/2005.06641)
