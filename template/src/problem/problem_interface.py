
from abc import ABC, abstractmethod


# ProblemInterface is an abstract base class (ABC). The classes that
# inherit from ProblemInterface need to implement the abstract methods
# in order to be instantiable.
class ProblemInterface(ABC):
    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def new_individual(self):
        pass

    @abstractmethod
    def mutation(self, individual):
        pass

    @abstractmethod
    def crossover(self, p1, p2):
        pass
