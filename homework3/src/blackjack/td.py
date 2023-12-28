import warnings
import os.path

from tqdm import trange
from observer import Observer

from .info import *


class TD(ABC):

    @abstractmethod
    def __init__(self, q: Q, gamma: float, alpha: float):
        self.q = q
        self.gamma = gamma
        self.alpha = alpha

    @abstractmethod
    def run(self, game: Game, iterations: int):
        pass


class QLearning(TD, Observer):
    """
    An off-policy method.
    """

    def update(self, new_state: tuple[State, State, Action, float]):
        state = new_state[0]
        next_state = new_state[1]
        action = new_state[2]
        reward = new_state[3]

        if next_state:
            v_plus = self.q.determine_v(next_state)
        else:
            v_plus = 0.0

        self.q[state, action] = (1 - self.alpha) * self.q[state, action] + self.alpha * (reward + self.gamma * v_plus)

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.1):
        super().__init__(q if q else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int) -> Q:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Q-Learning...")

        if os.path.exists("game_log_ql.txt"):
            os.remove("game_log_ql.txt")

        for i in range(iterations):
            # Play a game
            game.play(EpsilonGreedyPolicy(epsilon=0.3), self.q, self.gamma)

            # Log game information in a text file
            Info.log_game(game, i, "ql")

            for player in game.players:
                for rnd in player.experiences:
                    player.experiences[rnd].clear()

        print("Finished Q-Learning!")
        return self.q


class SARSA(TD):
    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.05):
        super().__init__(q if q else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int):
        pass
