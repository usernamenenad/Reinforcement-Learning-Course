from copy import copy

from .agents import *


class Game:
    """
    A class representing blackjack game.
    """

    @property
    def players(self) -> list[Agent]:
        return self.__players

    @property
    def deck(self) -> CardDeck:
        return self.__deck

    def __init__(self, players: list[Agent] = None, deck: CardDeck = None):
        self.__players = players if players else [Player() for _ in range(2)]
        self.__deck = deck if deck else CardDeck()

    def __initialize_round(self, players: list[Agent], round: int) -> None:
        """
        A blackjack game initializer. One card is given to
        each player and one card is commonly drawn, meaning that
        all players get the same card.
        """

        common_card = self.__deck.draw()

        for player in self.__players:
            player_card = self.__deck.draw()
            player.update_total(player_card)
            player.update_total(common_card)

            if player.state.total > 21:
                # print(f"{player.name} busted while initialization! It will be excluded from the game.")
                player.log_experience(round, [copy(player.state), Action.HIT, -1.0])
                players.remove(player)

    def __play_round(self, players: list[Agent], round: int) -> list[Agent]:
        """
        A private game method which simulates one round.
        Returns this round's winners.
        """
        max_total = 0

        for player in players:
            while True:
                action = player.policy()
                player.log_experience(round, [copy(player.state), action, 0.0])

                if action == Action.HOLD:
                    # Determine if this is the new max_total.
                    max_total = player.state.total if player.state.total > max_total else max_total
                    break

                card = self.__deck.draw()
                player.update_total(card)

                if player.state.total > 21:
                    # Player busts, and we "reset" its score.
                    # print(f"{player.name} busted!")
                    player.state.total = 0
                    break

        return [player for player in players if player.state.total == max_total]

    def play(self, gamma: float = 1.0):
        """
        A gameplay method that simulates one blackjack game.
        """
        for round in range(len(self.__players)):
            players = copy(self.__players)
            players[round], players[0] = players[0], players[round]
            self.__initialize_round(players, round)

            # Play the round and determine which players won the round.
            winners = self.__play_round(players, round)

            # If there's only one player who won the round,
            # it's the only player that will get a positive reward.
            if len(winners) == 1:
                winners[0].build_gains(round, 1.0, gamma)

            # If there are multiple winners, they all get a neutral
            # reward 0, which is already default.
            # Rest of the players get a negative reward.
            for player in players:
                if player not in winners:
                    player.build_gains(round, -1.0, gamma)

            for player in self.__players:
                player.reset()
