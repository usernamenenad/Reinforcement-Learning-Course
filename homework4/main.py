from cartpole import *

if __name__ == "__main__":
    cp = Cartpole(m=0.1, M=1.0, L=1)
    sarsa = SARSA()
    q: Q = sarsa.run(cp, EpsGreedyPolicy())

    print("Actions")
    for s in q.states:
        print(f"{s}, {GreedyPolicy().act(q, s)}")
