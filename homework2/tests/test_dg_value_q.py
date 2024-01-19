import matplotlib.pyplot as plt

from maze import *

DEFAULT_SPECS = [
    (10, lambda: RegularCell(-1)),
    (2, lambda: RegularCell(-10)),
    (2, lambda: WallCell(0)),
    (1, lambda: TerminalCell(-1)),
    (1, lambda: TeleportCell()),
]


def test_deterministic_graph():
    _, axes_db = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    axes_db = axes_db.flatten()

    base = MazeGraph(size=15, specs=DEFAULT_SPECS)

    env = MazeEnvironment(base=base, env_type=EnvType.DETERMINISTIC)
    q_iteration = QIteration(env, gamma=0.9)

    Info.draw_values(env=env, values=q_iteration.q.v_table, ax=axes_db[0])
    axes_db[0].set_title("Starting V values")

    k = q_iteration.run()

    Info.draw_values(env=env, values=q_iteration.q.v_table, ax=axes_db[1])
    axes_db[1].set_title(f"Optimal V values after {k} iterations.")

    Info.draw_policy(env=env, values=q_iteration.q, policy=GreedyPolicyQ(), gamma=q_iteration.gamma, ax=axes_db[2])
    axes_db[2].set_title("Optimal policy determined by Q values")

    Info.log_probabilities(env=env, nof="db")
    Info.log_q_values(q=q_iteration.q, nof="db")

    plt.show()
