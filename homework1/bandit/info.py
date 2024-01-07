import os
from .environment import *
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


class Info:
    @staticmethod
    def plot_convergence(
        q_evol: dict[Bandit, list[float]],
        mean_evol: dict[Bandit, list[float]],
        iterations: int,
        changes_at: list[int] = None,
    ) -> None:
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(
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
        change_at = 0

        for i, (bandit, ax) in enumerate(zip(q_evol.keys(), axes)):
            means: list[float] = []
            q_evols: list[float] = []
            for j in range(iterations + 1):
                means.append(mean_evol[bandit][change_at])
                q_evols.append(q_evol[bandit][j] if j in q_evol[bandit] else None)
                if j == changes_at[change_at]:
                    change_at = change_at + 1 if change_at + 1 < len(changes_at) else 0
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
                to_log.append({"Timestep": step, "Q-Value": q_evol[bandit][step]})

            with open(f"./logs/{bandit}.txt", "w") as blog:
                blog.write(tabulate(to_log, headers="keys", tablefmt="rst"))
