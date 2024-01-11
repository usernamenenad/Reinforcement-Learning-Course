import matplotlib.pyplot as plt

from maze import *

DEFAULT_SPECS = [
    (10, lambda: RegularCell(-1)),
    (2, lambda: RegularCell(-10)),
    (2, lambda: WallCell(-1)),
    (1, lambda: TerminalCell(-1)),
    (1, lambda: TeleportCell()),
]


def test_stochastic_graph():
    _, axes_sg = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axes_sg = axes_sg.flatten()

    base = MazeGraph(15, specs=DEFAULT_SPECS)

    env = MazeEnvironment(base=base, env_type=EnvType.STOCHASTIC, gamma=0.9)

    Info.draw_values(env=env, ax=axes_sg[0])
    axes_sg[0].set_title("Starting V values")

    k = env.compute_values()
    Info.draw_values(env=env, ax=axes_sg[1])
    axes_sg[1].set_title(f"Optimal V values after {k} iterations.")

    Info.draw_policy(env=env, policy=GreedyPolicyQ(), ax=axes_sg[2])
    axes_sg[2].set_title("Optimal policy determined by Q values")

    Info.draw_policy(env=env, policy=GreedyPolicyV(), ax=axes_sg[3])
    axes_sg[3].set_title("Optimal policy determined by V values")

    Info.log_probabilities(env=env, nof="sg")
    Info.log_q_values(q=env.q, nof="sg")

    plt.show()
