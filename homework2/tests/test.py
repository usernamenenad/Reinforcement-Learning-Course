from maze import *

import unittest

DEFAULT_SPECS = [
    (10, lambda: RegularCell(-1)),
    (2, lambda: RegularCell(-10)),
    (2, lambda: WallCell(-1)),
    (1, lambda: TerminalCell(-1)),
    (1, lambda: TeleportCell())
]


class TestMaze(unittest.TestCase):

    def test_deterministic_board(self):
        _, axes_db = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes_db = axes_db.flatten()

        base = MazeBoard(size=(8, 8),
                         specs=DEFAULT_SPECS)

        env = MazeEnvironment(base=base,
                              env_type=EnvType.DETERMINISTIC,
                              gamma=0.9)

        Info.draw_values(env=env,
                         ax=axes_db[0])
        axes_db[0].set_title("Starting V values")

        k = env.compute_values()
        Info.draw_values(env=env,
                         ax=axes_db[1])
        axes_db[1].set_title(f"Optimal V values after {k} iterations.")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyQ(),
                         ax=axes_db[2])
        axes_db[2].set_title("Optimal policy determined by Q values")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyV(),
                         ax=axes_db[3])
        axes_db[3].set_title("Optimal policy determined by V values")

        plt.show()

    def test_stochastic_board(self):
        _, axes_sb = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes_sb = axes_sb.flatten()

        base = MazeBoard(size=(8, 8),
                         specs=DEFAULT_SPECS)

        env = MazeEnvironment(base=base,
                              env_type=EnvType.STOCHASTIC,
                              gamma=0.9)

        Info.draw_values(env=env,
                         ax=axes_sb[0])
        axes_sb[0].set_title("Starting V values")

        k = env.compute_values()
        Info.draw_values(env=env,
                         ax=axes_sb[1])
        axes_sb[1].set_title(f"Optimal V values after {k} iterations.")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyQ(),
                         ax=axes_sb[2])
        axes_sb[2].set_title("Optimal policy determined by Q values")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyV(),
                         ax=axes_sb[3])
        axes_sb[3].set_title("Optimal policy determined by V values")

        plt.show()

    def test_deterministic_graph(self):
        _, axes_dg = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes_dg = axes_dg.flatten()

        base = MazeGraph(15,
                         specs=DEFAULT_SPECS)

        env = MazeEnvironment(base=base,
                              env_type=EnvType.DETERMINISTIC,
                              gamma=0.9)

        Info.draw_values(env=env,
                         ax=axes_dg[0])
        axes_dg[0].set_title("Starting V values")

        k = env.compute_values()
        Info.draw_values(env=env,
                         ax=axes_dg[1])
        axes_dg[1].set_title(f"Optimal V values after {k} iterations.")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyQ(),
                         ax=axes_dg[2])
        axes_dg[2].set_title("Optimal policy determined by Q values")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyV(),
                         ax=axes_dg[3])
        axes_dg[3].set_title("Optimal policy determined by V values")

        plt.show()

    def test_stochastic_graph(self):
        _, axes_sg = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axes_sg = axes_sg.flatten()
        
        base = MazeGraph(15,
                         specs=DEFAULT_SPECS)

        env = MazeEnvironment(base=base,
                              env_type=EnvType.STOCHASTIC,
                              gamma=0.9)

        Info.draw_values(env=env,
                         ax=axes_sg[0])
        axes_sg[0].set_title("Starting V values")

        k = env.compute_values()
        Info.draw_values(env=env,
                         ax=axes_sg[1])
        axes_sg[1].set_title(f"Optimal V values after {k} iterations.")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyQ(),
                         ax=axes_sg[2])
        axes_sg[2].set_title("Optimal policy determined by Q values")

        Info.draw_policy(env=env,
                         policy=GreedyPolicyV(),
                         ax=axes_sg[3])
        axes_sg[3].set_title("Optimal policy determined by V values")

        plt.show()


def main():
    unittest.main()
