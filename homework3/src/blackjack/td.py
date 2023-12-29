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

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.1):
        super().__init__(q if q else Q(), gamma, alpha)

    def update(self, *new_state):
        s = new_state[0][0]
        new_s = new_state[0][1]
        a = new_state[0][2]
        r = new_state[0][3]

        if new_s:
            v_plus = self.q.determine_v(new_s)
        else:
            v_plus = 0

        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * v_plus)

    def run(self, game: Game, iterations: int) -> Q:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Q-Learning...")

        if os.path.exists("game_log_ql.txt"):
            os.remove("game_log_ql.txt")

        for i in trange(iterations):
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
