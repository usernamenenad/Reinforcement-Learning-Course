import matplotlib.pyplot as plt
import networkx as nx

from .blackjack import *


class Info:

    @staticmethod
    def draw_experience(game: Game, rnd: int) -> None:

        players = copy(game.players)

        _, axes = plt.subplots(nrows=len(players), ncols=1, figsize=(20, 50))
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
    def log_experiences(players: list[Agent]) -> str:
        logger: str = ""

        for player in players:
            logger += f"{player.name}'s experience:\r\n"
            for rnd in player.experiences:
                to_print = list()
                for experience in player.experiences[rnd]:
                    to_print.append(
                        {
                            "State": experience[0],
                            "Action": experience[1].name,
                            "Drew card": experience[3].number if len(experience) == 4 else "-",
                            "Gain": experience[2]
                        }
                    )
                logger += tabulate(to_print, headers="keys", tablefmt="rst") + "\r\n\r\n"

        return logger

    @staticmethod
    def log_game(game: Game, game_number: int, nof: str):
        to_log = f"[Game {game_number}]:\r\n\r\n" + Info.log_experiences(game.players) + "\r\n"
        with open(f"game_log_{nof}.txt", "a") as gl:
            gl.write(to_log)

    @staticmethod
    def log_optimal_policy(q: Q, policy: str):
        to_log = []

        for state in q.all_states:
            action = GreedyPolicy().act(q, state)
            to_log.append(
                {
                    "State": state,
                    "Action": action
                }
            )

        with open(f"optimal_policy_{policy}.txt", "w") as opl:
            opl.write("Optimal policy: \r\n\r\n" + tabulate(to_log, headers="keys", tablefmt="rst"))
