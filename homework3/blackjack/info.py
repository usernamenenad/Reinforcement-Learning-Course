import os
from copy import copy

import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

from blackjack.agents import Agent
from blackjack.game import Game
from blackjack.policy import GreedyPolicy
from blackjack.utils import Q


class Info:
    @staticmethod
    def draw_experience(game: Game, rnd: int) -> None:
        players = copy(game.players)

        _, axes = plt.subplots(nrows=len(players), ncols=1, figsize=(20, 20))
        axes = axes.flatten()

        for i, player in enumerate(players):
            node_labels = dict()
            edge_labels = dict()
            node_colors = list()

            g = nx.DiGraph()

            for j, exp in enumerate(player.experiences[rnd].experience):
                node_labels[j] = exp[0].total
                node_colors.append("#0000ff")

            node_labels[len(node_labels)] = "T"
            node_colors.append("#ff0000")

            for j, exp in enumerate(player.experiences[rnd].experience):
                label = exp[1].name
                if exp[3]:
                    label += f", {exp[3]}"
                edge_labels[(j, j + 1)] = label

            g.add_nodes_from(node_labels)
            g.add_edges_from(edge_labels)

            pos = nx.planar_layout(g)

            nx.draw_networkx_nodes(
                g, pos=pos, node_size=500, node_color=node_colors, ax=axes[i]
            )

            nx.draw_networkx_labels(
                g, pos=pos, labels=node_labels, font_color="w", font_size=11, ax=axes[i]
            )

            nx.draw_networkx_edges(g, pos=pos, width=2, style="dashed", ax=axes[i])

            nx.draw_networkx_edge_labels(
                g, pos=pos, edge_labels=edge_labels, font_size=12, ax=axes[i]
            )

            axes[i].set_title(f"{player.name}'s experience")

        plt.show()

    @staticmethod
    def log_experiences(players: list[Agent]) -> str:
        logger: str = ""

        for player in players:
            logger += f"{player.name}'s experience:\r\n"
            for rnd in player.experiences:
                to_log = list()
                for experience in player.experiences[rnd]:
                    to_log.append(
                        {
                            "State": experience[0],
                            "Action": experience[1].name,
                            "Drew card": experience[3].number if experience[3] else "-",
                            "Gain": experience[2],
                        }
                    )
                logger += tabulate(to_log, headers="keys", tablefmt="rst") + "\r\n\r\n"

        return logger

    @staticmethod
    def log_game(game: Game, game_number: int, nof: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        to_log = (
            f"[Game {game_number}]:\r\n\r\n"
            + Info.log_experiences(game.players)
            + "\r\n"
        )

        with open(f"./logs/game_log_{nof}.txt", "a") as gl:
            gl.write(to_log)

    @staticmethod
    def log_q_values(q: Q, policy: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        with open(f"./logs/q_values_{policy}.txt", "w") as qv:
            qv.write(q.__str__())

    @staticmethod
    def log_optimal_policy(q: Q, policy: str):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        to_log = []

        for state in q.states:
            action = GreedyPolicy().act(q, state)
            to_log.append({"State": state, "Action": action})

        with open(f"./logs/optimal_policy_{policy}.txt", "w") as opl:
            opl.write(
                "Optimal policy: \r\n\r\n"
                + tabulate(to_log, headers="keys", tablefmt="rst")
            )
