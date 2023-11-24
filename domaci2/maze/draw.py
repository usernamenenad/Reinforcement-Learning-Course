import matplotlib.pyplot as plt
from tabulate import tabulate

from agent import *


class Draw:
    @staticmethod
    def draw_board(board: MazeBoard, pos: tuple[int, int] = None, ax=None):
        ax = ax if ax else plt
        board_img = np.ones(
            shape=(board.rows_no, board.cols_no, 3), dtype=np.uint8)

        for i in range(board.rows_no):
            for j in range(board.cols_no):
                board_img[i, j, :] = board[i, j].color
                if isinstance(board[i, j], TeleportCell):
                    ax.text(
                        j - 0.4,
                        i + 0.1,
                        f"({board[i, j].to_teleport_to.position[0]},"
                        f"{board[i, j].to_teleport_to.position[1]})",
                    )
        if pos:
            row, col = pos
            ax.text(col - 0.4, row + 0.1, "X", fontweight="bold")

        ax.imshow(board_img)

    @staticmethod
    def draw_values(agent: Agent, ax=None):
        ax = ax if ax else plt
        Draw.draw_board(agent.env.board, ax=ax)
        for s in agent.env.states:
            ax.text(s[1] - 0.4, s[0] + 0.1, f"{agent.env.determine_v(s):.1f}")

    @staticmethod
    def draw_policy(agent: Agent, policy: str, ax=None):
        ax = ax if ax else plt
        Draw.draw_board(agent.env.board, ax=ax)
        for s in agent.env.states:
            if not agent.env.is_terminal(s):
                # if a == Action.ACTION_R:
                #     ax.text(s[1] - 0.25, s[0] + 0.1, "→")
                # elif a == Action.ACTION_L:
                #     ax.text(s[1] - 0.25, s[0] + 0.1, "←")
                # elif a == Action.ACTION_U:
                #     ax.text(s[1] - 0.25, s[0] + 0.1, "↑")
                # else:
                #     ax.text(s[1] - 0.25, s[0] + 0.1, "↓")
                a = agent.take_policy(s, policy)
                if a == Action.ACTION_R:
                    ax.text(s[1] - 0.25, s[0] + 0.1, 'R')
                elif a == Action.ACTION_L:
                    ax.text(s[1] - 0.25, s[0] + 0.1, 'L')
                elif a == Action.ACTION_U:
                    ax.text(s[1] - 0.25, s[0] + 0.1, 'U')
                else:
                    ax.text(s[1] - 0.25, s[0] + 0.1, 'D')

    @staticmethod
    def print_probabilities(agent: Agent):
        to_print = []
        for s, a in agent.env.probabilities:
            news = agent.take_action(a, s)
            for new in news:
                to_print.append(
                    {
                        "State": s,
                        "Action": a.name,
                        "Direction": new['Direction'].name,
                        "Next state": new['New state'],
                        "Reward": new['Reward'],
                        "Probability(s+, r | s, a)": agent.env.probabilities[(s, a)][new['Direction']],
                    }
                )

        print(tabulate(to_print, "keys", "rst"))


if __name__ == "__main__":
    print("Hi! Here you can find implementation of Draw class, used for plotting.")
