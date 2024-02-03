import os

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from cartpole.utils import *
from cartpole.policy import GreedyPolicy


class Info:
    @staticmethod
    def log_q_values(q: Q, nof: str) -> None:
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/q_{nof}.txt", "w") as logger:
            to_log = []
            for s, a in q:
                to_log.append(
                    {
                        "Position": s.x,
                        "Velocity": s.x_dot,
                        "Angle": s.o,
                        "Angular velocity": s.o_dot,
                        "Action": a,
                        "Q value": q[s, a],
                    }
                )
            logger.write(tabulate(to_log, headers="keys", tablefmt="rst"))

    @staticmethod
    def log_optimal_policy(q: Q, nof: str) -> None:
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/optimal_policy_{nof}.txt", "w") as logger:
            to_log = []
            for s in q.states:
                to_log.append(
                    {
                        "Position": s.x,
                        "Velocity": s.x_dot,
                        "Angle": s.o,
                        "Angular velocity": s.o_dot,
                        "Optimal action": GreedyPolicy().act(q, s),
                    }
                )
            logger.write(tabulate(to_log, headers="keys", tablefmt="rst"))

    @staticmethod
    def __plot_results(results: dict[int, bool], nop: str) -> None:
        sns.set_theme(style="darkgrid")
        colors = sns.color_palette("husl", n_colors=2)
        x = list(results.keys())
        y = [1 if results[i] else -1 for i in results]
        c = ["blue" if results[i] else "red" for i in results]
        plt.title(
            f"Successful (blue) and failed (red) actions with respect to iteration number using {nop} algorithm"
        )
        plt.scatter(x, y, marker=".", s=1, c=c)
        plt.show()

    @staticmethod
    def __log_text_results(results: dict[int, bool], nop: str) -> None:
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open(f"./logs/results_{nop}.txt", "w") as logger:
            logger.write(
                f"Percentage of successful and unsuccessful actions using {nop} algorithm\n"
            )
            res = list(results.values())
            len_res = len(res)
            successful = res.count(True)
            logger.write(
                tabulate(
                    tabular_data=[
                        {
                            "Successful": successful / len_res,
                            "Failed": (len_res - successful) / len_res,
                        }
                    ],
                    headers="keys",
                    tablefmt="rst",
                )
            )

    @staticmethod
    def log_results(results: dict[int, bool], nop: str) -> None:
        Info.__plot_results(results, nop)
        Info.__log_text_results(results, nop)
