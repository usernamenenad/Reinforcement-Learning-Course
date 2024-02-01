from cartpole import *


def test_sarsa():
    cp = Cartpole(m=0.1, M=1, L=0.25)  # Model
    T = 0.0000001  # Sample time
    sarsa = SARSA()
    q: Q = sarsa.run(
        model=cp,
        policy=EpsGreedyPolicy(epsilon=0.1),
        actions=[-1.0, 0.0, 1.0],
        gamma=0.9,
        T=T,
    )
