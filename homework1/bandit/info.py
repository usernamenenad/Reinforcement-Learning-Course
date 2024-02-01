import os
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from bandit.bandit import Bandit


class Info:
    @staticmethod
    def plot_convergence(
        q_evol: dict[Bandit, dict[int, float]],
        mean_evol: dict[Bandit, list[float]],
        iterations: int,
        changes_at: list[int] | None = None,
    ) -> None:
        sns.set_theme(style="darkgrid")
        _, axes = plt.subplots(
            nrows=len(q_evol),
            ncols=1,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = axes.flatten()
        colors = sns.color_palette("husl", n_colors=len(q_evol))

        if not changes_at:
            changes_at = [-1]

        for i, (bandit, ax) in enumerate(zip(q_evol.keys(), axes)):
            copy_changes_at = deepcopy(changes_at)
            change_at = copy_changes_at.pop(0)
            popped = 1
            means: list[float] = []
            q_evols: list[float | None] = []
            for j in range(iterations + 1):
                means.append(mean_evol[bandit][popped - 1])
                q_evols.append(q_evol[bandit][j] if j in q_evol[bandit] else None)
                if j == change_at:
                    change_at = copy_changes_at.pop(0) if copy_changes_at else -1
                    popped += 1
            ax.plot(range(iterations + 1), means, linestyle="--", color=colors[i])
            ax.plot(range(iterations + 1), q_evols, marker=".", ms=2, color=colors[i])
            ax.set_title(f"Q over time for {bandit}")

        # plt.subplot_tool()
        plt.show()

    @staticmethod
    def log_q_evol(q_evol: dict[Bandit, dict[int, float]]):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        for bandit in q_evol:
            to_log = []
            for step in q_evol[bandit]:
                to_log.append({"Moment": step, "Q-Value": q_evol[bandit][step]})

            with open(f"./logs/{bandit}.txt", "w") as blog:
                blog.write(tabulate(to_log, headers="keys", tablefmt="rst"))
