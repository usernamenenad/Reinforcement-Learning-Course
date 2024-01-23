import os

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from cartpole.utils import *
from cartpole.policy import GreedyPolicy


class Info:
    @staticmethod
    def log_q_values(q: Q, nof: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/q_{nof}.txt", "w") as logger:
            to_log = []
            for s, a in q:
                to_log.append({"State": s, "Action": a, "Q value": q[s, a]})
            logger.write(tabulate(to_log, headers="keys", tablefmt="rst"))

    @staticmethod
    def log_optimal_policy(q: Q, nof: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/optimal_policy_{nof}.txt", "w") as logger:
            to_log = []
            for s, a in q:
                to_log.append({"State": s, "Optimal action": GreedyPolicy().act(q, s)})
            logger.write(tabulate(to_log, headers="keys", tablefmt="rst"))

    @staticmethod
    def plot_results(results: dict[int, bool], nop: str):
        sns.set_theme(style="darkgrid")
        colors = sns.color_palette("husl", n_colors=2)
        x = list(results.keys())
        y = [1 if results[i] else -1 for i in results]
        c = ["blue" if results[i] else "red" for i in results]
        plt.scatter(x, y, marker=".", s=1, c=c)
        plt.show()
