from maze import *
import unittest
import os

DEFAULT_SPECS = [
    (10, lambda: RegularCell(-1)),
    (2, lambda: RegularCell(-10)),
    (2, lambda: WallCell(-1)),
    (1, lambda: TerminalCell(-1)),
    (1, lambda: TeleportCell()),
]


def test_deterministic_graph():
    _, axes_dg = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axes_dg = axes_dg.flatten()

    base = MazeGraph(15, specs=DEFAULT_SPECS)

    env = MazeEnvironment(base=base, env_type=EnvType.DETERMINISTIC, gamma=0.9)

    Info.draw_values(env=env, ax=axes_dg[0])
    axes_dg[0].set_title("Starting V values")

    k = env.compute_values()
    Info.draw_values(env=env, ax=axes_dg[1])
    axes_dg[1].set_title(f"Optimal V values after {k} iterations.")

    Info.draw_policy(env=env, policy=GreedyPolicyQ(), ax=axes_dg[2])
    axes_dg[2].set_title("Optimal policy determined by Q values")

    Info.draw_policy(env=env, policy=GreedyPolicyV(), ax=axes_dg[3])
    axes_dg[3].set_title("Optimal policy determined by V values")

    Info.log_probabilities(env=env, nof="dg")
    Info.log_q_values(q=env.q, nof="dg")

    plt.show()
