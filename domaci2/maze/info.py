import matplotlib.pyplot as plt

from tabulate import tabulate

from .agent import *


class Info:
    @staticmethod
    def draw_board(board: MazeBoard, state: tuple[int, int] = None, ax=None):
        ax = ax if ax else plt
        board_img = np.ones(shape=(board.rows_no, board.cols_no, 3), dtype=np.uint8)

        for i in range(board.rows_no):
            for j in range(board.cols_no):
                board_img[i, j, :] = board[i, j].color
                if isinstance(board[i, j], TeleportCell):
                    ax.text(
                        j - 0.4,
                        i + 0.1,
                        f"({board[i, j].to_teleport_to.state[0]},"
                        f"{board[i, j].to_teleport_to.state[1]})",
                    )
        if state:
            row, col = state
            ax.text(col - 0.4, row + 0.1, "X", fontweight="bold")

        ax.imshow(board_img)

    @staticmethod
    def draw_values(agent: Agent, ax=None):
        ax = ax if ax else plt
        Info.draw_board(agent.env.board, ax=ax)
        for s in agent.env.states:
            ax.text(s[1] - 0.4, s[0] + 0.1, f"{agent.env.v_values[s]:.1f}")

    @staticmethod
    def draw_policy(agent: Agent, policy: str, ax=None):
        ax = ax if ax else plt
        policy = GreedyPolicyV() if policy == "greedy_v" else GreedyPolicyQ()
        Info.draw_board(agent.env.board, ax=ax)
        sa = agent.determine_optimal_actions(policy)
        for s in sa:
            if sa[s] == Action.ACTION_R:
                ax.text(s[1] - 0.25, s[0] + 0.1, "R")
            elif sa[s] == Action.ACTION_L:
                ax.text(s[1] - 0.25, s[0] + 0.1, "L")
            elif sa[s] == Action.ACTION_U:
                ax.text(s[1] - 0.25, s[0] + 0.1, "U")
            else:
                ax.text(s[1] - 0.25, s[0] + 0.1, "D")

            # if a == Action.ACTION_R:
            #     ax.text(s[1] - 0.25, s[0] + 0.1, "→")
            # elif a == Action.ACTION_L:
            #     ax.text(s[1] - 0.25, s[0] + 0.1, "←")
            # elif a == Action.ACTION_U:
            #     ax.text(s[1] - 0.25, s[0] + 0.1, "↑")
            # else:
            #     ax.text(s[1] - 0.25, s[0] + 0.1, "↓")
            # a = agent.take_policy(s, policy)

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
                        "Direction": new["Direction"].name,
                        "Next state": new["New state"],
                        "Reward": new["Reward"],
                        "Probability(s+, r | s, a)": agent.env.probabilities[(s, a)][
                            new["Direction"]
                        ],
                    }
                )

        print(tabulate(to_print, "keys", "rst"))
