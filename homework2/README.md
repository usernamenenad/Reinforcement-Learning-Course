# Homework Task 2: Maze

A task consists of an agent whose task is to find an optimal path to a *terminal state* in maze. A maze can be constructed as:

- *board*, which looks like an average maze from a bird's eye
- *graph*, which is your usual directed graph

Also, the *environment* (basically, *Markov decision process*) can be either *deterministic* or *stochastic*. *Environment* class
is implemented as a *callable* object which acts as a *MDP*, given state and action. 

It should be mentioned that, when considering *stochastic MDP*, *actions* do not necessarly mean *directions of acting*. 
For example, *Action.A1* will not always mean the transition to the right of the standing cell - 
**there is no clear mapping between *actions* and *directions* in *stochastic case***. Of course, some probabilities collapse to 
$1$ (others to $0$) when we're talking about *deterministic MDP* - **then** we're having a clear map between *actions* and *directions*.

*Probabilities* are implemented as *dataclass*, which, given state and action, returns a probability of taking an action and following 
a certain direction. Here, we'll consider that the probabilities of taking any action in any state is always $1$, for every action 
(meaning, $\pi(a | s) = 1 \text{, } \forall a$). 

$Q$ and $V$ values are implemented as *dataclasses*. What's specific about those two is that they return a *v_table*

- In the case of $Q$ dataclass, for every state the $V$ value is determined by $v(s) = \max_{a}\{q(s, a)\}$
- In the case of $V$ dataclass, it is trivial, as it's already contained in the class itself.

An algorithm used for determining optimal $Q$ and $V$ values is *value iteration*, namely a *dynamic programming* type of algorithm.

## Value iteration

We're considering finding the optimal policy using $Q$ values and $V$ values.

### Value iteration on *deterministic MDP* using $Q$ and $V$ values

![](./images/db.png)

![](./images/dg.png)

### Value iteration on *stochastic MDP* using $Q$ and $V$ values

![](./images/sb.png)

![](./images/sg.png)

Also, after executing the algorithms, *log files* will appear in *logs* folder. These contain information about generated probabilities of *MDP*,
as well as optimal $Q$ and $V$ values.
