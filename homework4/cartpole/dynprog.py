from abc import ABC, abstractmethod
from copy import deepcopy

from alive_progress import alive_bar


from cartpole.policy import GreedyPolicy
from cartpole.utils import *
from cartpole.model import Cartpole

class DynamicProgramming(ABC):
    @abstractmethod
    def run(self, model: Cartpole, gamma: float = 1.0, iterations: int = 1000) -> Q:
        pass


class QIteration(DynamicProgramming):

    def __init__(self) -> None:
        self.__ss: State = self.__initialize_ss()
        self.__q: Q = Q()
        self.__result: dict[int, bool] = {}

    def __initialize_ss(self) -> State:
        return (
            round(uniform(-5.0, 5.0), round_prec),
            0.0,
            round(uniform(ANGLE_M20, ANGLE_20), round_prec),
            0.0,
        )
   
    def __update_values(self):
        
                
    
    def run(self, model: Cartpole, gamma: float = 1.0, iterations: int = 1000) -> Q:
        for a in actions:
            self.__q[self.__ss, a] = random()
            
        with alive_bar(iterations) as bar:
            for _ in range(iterations):
                oq = deepcopy(self.__q)
                self.__update_values()
        
