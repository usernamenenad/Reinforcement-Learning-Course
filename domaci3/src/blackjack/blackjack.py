from abc import ABC, abstractmethod
from copy import copy
from random import random

from .utils import *


class Agent(ABC):
    """
    An interface for a type of Blackjack player,
    that can be a regular player or a dealer.
    """

    @property
    def state(self) -> State:
        return self.__state

    @property
    def experience(self):
        return self.__experience

    @experience.setter
    def experience(self, experience: Experience):
        self.__experience = experience

    def __init__(self, state: State):
        self.__state = state
        self.__experience: Experience = Experience()

    def update_total(self, card: Card):
        match card.number:

            case CardNumber.ACE:
                if self.__state.total + card.number.value <= 21:
                    self.__state.total += 11
                    self.__state.has_ace = True
                else:
                    self.__state.total += 1

            case _:
                self.__state.total += card.number.value
                if self.__state.total > 21:
                    if self.__state.has_ace:
                        self.__state.total -= 10
                        self.__state.has_ace = False

    @abstractmethod
    def policy(self) -> Action:
        pass


class Dealer(Agent):
    """
    A dealer agent.
    """

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: DealerState = None):
        super().__init__(state if state else DealerState())

    def policy(self) -> Action:
        return Action.HIT if self.state.total < 17 else Action.HOLD


class Player(Agent):
    """
    A player agent.
    """

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: PlayerState = None):
        super().__init__(state if state else PlayerState())

    def policy(self) -> Action:
        return Action.HIT if random() < 0.5 else Action.HOLD


class Game:

    @property
    def dealer(self) -> Dealer:
        return self.__dealer

    @property
    def players(self) -> list[Player]:
        return self.__players

    @property
    def deck(self) -> CardDeck:
        return self.__deck

    def __init__(self, dealer: Dealer = None, players: list[Player] = None, deck: CardDeck = None):
        self.__players = players if players else [Player() for _ in range(2)]
        self.__dealer = dealer if dealer else Dealer()
        self.__deck = deck if deck else CardDeck()

    def __initialize_game(self):
        self.__dealer.update_total(self.__deck.draw())
        dealer_known = self.__dealer.state.total
        self.__dealer.update_total(self.__deck.draw())

        for player in self.__players:
            player.state.dealer_total = dealer_known
            for card in [self.__deck.draw() for _ in range(2)]:
                player.update_total(card)

    def play(self, gamma: float = 1.0):
        self.__initialize_game()

        for player in copy(self.__players) + [self.__dealer]:
            while True:
                action = player.policy()
                player.experience += [copy(player.state), action, 0.0]
                if action == Action.HOLD:
                    break
                card = self.__deck.draw()
                player.update_total(card)

        self.__dealer.state.total = 0 if self.__dealer.state.total > 21 else self.__dealer.state.total

        for player in self.__players:
            player.state.total = -1 if player.state.total > 21 else player.state.total
            if player.state.total < self.__dealer.state.total:
                reward = -1
                print(f"Agent {player} lost!")
            elif player.state.total == self.__dealer.state.total:
                reward = 0
                print(f"Agent {player} drew!")
            else:
                reward = 1
                print(f"Agent {player} won!")

            player.experience.compute_experience(reward, gamma)
            print(player.experience.experience)
