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
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    base = MazeGraph(size=15, specs=DEFAULT_SPECS)
    env = MazeEnvironment(base=base, env_type=EnvType.STOCHASTIC)

    q_iteration = QIteration(env, gamma=0.9)
    v_iteration = VIteration(env, gamma=0.9)

    Info.draw_values(env=env, vf=q_iteration.q, ax=axes[0])
    axes[0].set_title("Starting V values using QVI")
    Info.draw_values(env=env, vf=v_iteration.v, ax=axes[3])
    axes[3].set_title("Startinv V values using VVI")

    k_q = q_iteration.run()
    k_v = v_iteration.run()

    Info.draw_values(env=env, vf=q_iteration.q, ax=axes[1])
    axes[1].set_title(f"Optimal V values after {k_q} iterations using QVI.")
    Info.draw_values(env=env, vf=v_iteration.v, ax=axes[4])
    axes[4].set_title(f"Optimal V values after {k_v} iterations using VVI.")

    Info.draw_policy(
        env=env,
        vf=q_iteration.q,
        policy=GreedyPolicy(),
        gamma=q_iteration.gamma,
        ax=axes[2],
    )
    axes[2].set_title("Optimal policy determined by Q values")

    Info.draw_policy(
        env=env,
        vf=v_iteration.v,
        policy=GreedyPolicy(),
        gamma=q_iteration.gamma,
        ax=axes[5],
    )
    axes[5].set_title("Optimal policy determined by V values")

    Info.log_probabilities(env=env, nof="sg")

    Info.log_values(vf=q_iteration.q, nof="sg")
    Info.log_values(vf=v_iteration.v, nof="sg")
    plt.show()
