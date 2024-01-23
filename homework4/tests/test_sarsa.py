from cartpole import Cartpole, SARSA, EpsGreedyPolicy, Q


def test_sarsa():
    cp = Cartpole(m=0.1, M=1, L=0.25)
    sarsa = SARSA()
    q: Q = sarsa.run(model=cp, policy=EpsGreedyPolicy(epsilon=0.1), gamma=0.9)
