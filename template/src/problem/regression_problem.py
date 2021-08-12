
from src.problem.problem_interface import ProblemInterface


class RegressionProblem(ProblemInterface):
    def __init__(self, fname):
        ###################################
        # TODO
        ###################################
        pass

    def fitness(self, individual):
        ###################################
        # TODO
        ###################################
        return 0

    def new_individual(self):
        ###################################
        # TODO
        ###################################
        individual = None
        return individual

    def mutation(self, individual):
        ###################################
        # TODO
        ###################################
        return individual

    def crossover(self, p1, p2):
        ###################################
        # TODO
        ###################################
        return p1, p2

    def plot(self, individual):
        ###################################
        # TODO
        ###################################
        pass
