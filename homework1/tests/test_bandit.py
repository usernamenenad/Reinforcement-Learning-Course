from bandit import *


def change_law():
    return 10 * (random() - 0.5), 5 * random()


def test_bandit():
    NO_BANDITS = 5
    IS_STATIONARY = False
    ITERATIONS = 10000
    CHANGES_AT = [200, 1000, 6000, 9000]

    bandits = [Bandit(10 * (random() - 0.5), 5 * random()) for _ in range(NO_BANDITS)]
    env = Environment(bandits, is_stationary=IS_STATIONARY)
    q_evol, mean_evol, rewards = env.run(
        policy=EpsGreedyPolicy(),
        iterations=ITERATIONS,
        changes_at=CHANGES_AT,
        change_law=change_law,
    )
    Info.plot_convergence(q_evol, mean_evol, ITERATIONS, CHANGES_AT)
    Info.log_q_evol(q_evol)
