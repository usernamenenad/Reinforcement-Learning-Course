import matplotlib.pyplot as plt
import networkx as nx

from .blackjack import *


class Info:

    @staticmethod
    def draw_experience(game: Game, round: int) -> None:

        players = copy(game.players)

        _, axes = plt.subplots(nrows=len(players), ncols=1, figsize=(20, 50))
        axes = axes.flatten()

        for i, player in enumerate(players):

            node_labels = dict()
            edge_labels = dict()
            node_colors = list()

            g = nx.DiGraph()

            for j, exp in enumerate(player.experiences[round].experience):
                node_labels[j] = exp[0].total
                node_colors.append("#0000ff")

            node_labels[len(node_labels)] = "T"
            node_colors.append("#ff0000")

            for j, exp in enumerate(player.experiences[round].experience):
                edge_labels[(j, j + 1)] = exp[1].name

            g.add_nodes_from(node_labels)
            g.add_edges_from(edge_labels)

            pos = nx.planar_layout(g)

            nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_size=2000,
                                   node_color=node_colors,
                                   ax=axes[i])

            nx.draw_networkx_labels(g,
                                    pos=pos,
                                    labels=node_labels,
                                    font_size=15,
                                    ax=axes[i])

            nx.draw_networkx_edges(g,
                                   pos=pos,
                                   ax=axes[i])

            nx.draw_networkx_edge_labels(g,
                                         pos=pos,
                                         edge_labels=edge_labels,
                                         font_size=20,
                                         ax=axes[i])

            axes[i].set_title(f"{player.name}'s experience")
            # axes[i].set_axis_off()

        plt.show()

    @staticmethod
    def log_experiences(game: Game):
        for player in game.players:
            print(f"{player.name}'s experience:\r\n")
            for round in player.experiences:
                print(f"Round {round + 1}")
                to_print = list()
                for experience in player.experiences[round].experience:
                    to_print.append(
                        {
                            "State": experience[0],
                            "Action": experience[1].name,
                            "Gain": experience[2]
                        }
                    )
                print(tabulate(to_print, headers="keys", tablefmt="rst") + "\r\n")
