import unittest

from maze import *


class TestMaze(unittest.TestCase):
    """
    A class for testing maze.
    """

    def test_maze(self) -> None:
        # We will make the default board specs and
        # the board itself
        DEFAULT_SPECS = [
            (10, lambda: RegularCell(-1)),
            (2, lambda: RegularCell(-10)),
            (2, lambda: WallCell()),
            (1, lambda: TerminalCell(-1)),
            (0.5, lambda: TeleportCell()),
        ]

        board = MazeBoard(size=(8, 8), specs=DEFAULT_SPECS)

        # Constructing an environment out of board
        env = MazeEnvironment(board=board, gamma=0.9)

        # Constructing the agent which lives in environment
        agent = Agent(env, actions=env.get_actions())

        Info.print_probabilities(agent)

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
        axes = axes.flatten()

        Info.draw_values(agent=agent, ax=axes[0])
        axes[0].set_title("Initial V values")

        k = env.compute_values()

        Info.draw_values(agent=agent, ax=axes[1])
        axes[1].set_title(f"V values computed using Q values after {k} iterations.")

        Info.draw_policy(agent, "greedy_v", ax=axes[2])
        axes[2].set_title("Optimal policy using V values")

        Info.draw_policy(agent, "greedy_q", ax=axes[3])
        axes[3].set_title("Optimal policy using Q values")

        plt.show()


def main() -> None:
    unittest.main()
