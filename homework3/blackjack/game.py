from copy import copy, deepcopy

from observer import Observable

from blackjack.agents import Agent, Player, Dealer
from blackjack.utils import CardDeck, Action, Q


class Game(Observable):
    """
    A class representing blackjack game.
    """

    @property
    def players(self) -> list[Agent]:
        return self.__players

    @property
    def deck(self) -> CardDeck:
        return self.__deck

    def __init__(
        self,
        players: list[Player],
        dealer: Dealer | None = None,
        deck: CardDeck | None = None,
    ):
        super().__init__()
        self.__players: list[Player] = players
        self.__dealer: Dealer | None = dealer
        self.__deck: CardDeck = deck if deck else CardDeck()

    def __initialize_round(self) -> None:
        """
        A blackjack game initializer. One card is given to
        each player and one card is commonly drawn, meaning that
        all players get the same card.
        """

        common_card = self.__deck.draw()

        for player in self.__players:
            player.update_total(common_card)
            player_card = self.__deck.draw()
            player.update_total(player_card)

    def __play_round(self, players: list[Agent], q: Q, rnd: int) -> list[Agent]:
        """
        A private game method which simulates one round.
        Returns this round's winners.
        """
        max_total = 0
        action: Action | None = None

        for player in players:
            while True:
                action = action if action else player.act(q, player.state)

                if action == Action.HOLD:
                    isinstance(player, Dealer) or player.log_experience(
                        rnd, [deepcopy(player.state), action, 0.0, None]
                    )

                    # Determine if this is the new max_total.
                    max_total = (
                        player.state.total
                        if player.state.total > max_total
                        else max_total
                    )
                    break

                card = self.__deck.draw()
                isinstance(player, Dealer) or player.log_experience(
                    rnd, [deepcopy(player.state), action, 0.0, card]
                )
                old_state = deepcopy(player.state)
                player.update_total(card)

                if player.state.total > 21:
                    # Player busts, and we "reset" its score.
                    # print(f"{player.name} busted!")
                    player.state.total = 0
                    break
                else:
                    new_action = player.act(q, player.state)
                    isinstance(player, Dealer) or self.notify(
                        old_state, action, 0.0, player.state, new_action
                    )
                    action = new_action

        # Return round winners
        return [player for player in players if player.state.total == max_total]

    def play(self, q: Q, gamma: float = 1.0) -> None:
        """
        A gameplay method that simulates one blackjack game.
        """

        for rnd in range(len(self.__players)):
            players = copy(self.__players)

            if self.__dealer is not None:
                players.append(copy(self.__dealer))

            # A blackjack rule - each round, other player starts the game.
            # This will be simulated by putting the player on the list's first index.
            players[rnd], players[0] = players[0], players[rnd]
            self.__initialize_round()

            # Play the round and determine which players won the round.
            winners = self.__play_round(players, q, rnd)

            # If there's only one player who won the round, only he
            # will get a positive reward.
            if len(winners) == 1:
                if not isinstance(winners[0], Dealer):
                    winners[0].build_gains(rnd, 1.0, gamma)
                    state = deepcopy(winners[0].experiences[rnd][-1][0])
                    action = deepcopy(winners[0].experiences[rnd][-1][1])
                    reward = 1.0
                    self.notify(state, action, reward, None, None)
            else:
                for winner in winners:
                    if not isinstance(winner, Dealer):
                        state = deepcopy(winner.experiences[rnd][-1][0])
                        action = deepcopy(winner.experiences[rnd][-1][1])
                        reward = 0.0
                        self.notify(state, action, reward, None, None)

            # If there are multiple winners, they all get a neutral reward 0 for drawing,
            # which is already default.
            # Rest of the players get a negative reward.
            for player in players:
                if not isinstance(player, Dealer):
                    if player not in winners:
                        player.build_gains(rnd, -1.0, gamma)
                        state = deepcopy(player.experiences[rnd][-1][0])
                        action = deepcopy(player.experiences[rnd][-1][1])
                        reward = -1.0
                        self.notify(state, action, reward, None, None)

            for player in self.__players:
                player.reset()
