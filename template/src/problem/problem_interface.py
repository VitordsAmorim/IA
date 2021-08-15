
from abc import ABC, abstractmethod


# ProblemInterface is an abstract base class (ABC). The classes that
# inherit from ProblemInterface need to implement the abstract methods
# in order to be instantiable.
class ProblemInterface(ABC):
    @abstractmethod
    def fitness(self, population):
        pass

    @abstractmethod
    def initial_population(self, population_size):
        pass

    @abstractmethod
    def elitism(self, newpopulation, bestindividualold, fitn):
        pass

    @abstractmethod
    def mutation(self, individual, mutation_rate):
        pass

    @abstractmethod
    def crossover(self, p1, p2):
        pass

    @abstractmethod
    def subcrossover(self, start_point, end_point, p1, p2, son1):
        pass

    @abstractmethod
    def selection_process(self, fitn):
        pass

    @abstractmethod
    def plot(self, best_fitness, ngeracoes):
        pass
