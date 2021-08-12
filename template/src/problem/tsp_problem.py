import numpy as np
from src.problem.problem_interface import ProblemInterface
import random

class TSPProblem(ProblemInterface):

    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()

        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.float32)
        self.x = lines[:, :-1]
        self.y = lines[:, -1:]
        pass

    def initial_population(self, population_size):
        # Write the list with the for 1 to the size of the population(the number of cities)
        list = []
        for i in range(1, len(self.x)): list.append(i)

        # The initial population has size 200
        population = []
        for i in range(0, population_size):
            population.append(random.sample(list, len(self.x) - 1))
        return population


    def fitness(self, population):
        # Calcular a função fitness para toda população
        fitn = []
        for k in range(0, len(population)):
            ind = 0
            total_distance = 0

            # Calcular a função fitness para um indivíduo
            for i in range(0, len(self.x) - 1):
                n = population[k][i]
                distance = np.sqrt((self.x[ind] - self.x[n]) ** 2 + (self.y[ind] - self.y[n]) ** 2)
                total_distance = float(total_distance) + float(distance)
                ind = n
            distance = np.sqrt((self.x[ind] - self.x[0]) ** 2 + (self.y[ind] - self.y[0]) ** 2)
            total_distance = total_distance + float(distance)
            fitn.append(float(total_distance))

        best_fit = max(fitn)
        best_pos = fitn.index(best_fit)
        path = []
        path = population[best_pos]
        return best_fit, best_pos, path, fitn


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
        print("pai1  ", p1)
        print("\npai2  ", p2)

        son1 = [0] * len(p1)
        son2 = [0] * len(p1)
        # Randomly choose two points to slice a solution
        list = []
        for i in range(1, len(p1)): list.append(i)
        corte1, corte2 = random.sample(list, 2)

        if corte1 > corte2:
            start_point = corte2
            end_point   = corte1
        else:
            start_point = corte1
            end_point   = corte2

        for i in range(start_point, end_point):
            son1[i] = p2[i]
        print("filho1", son1)

        j = end_point
        end_list = len(p1)
        for i in range(end_point, end_list):

            if i == (len(p1) - 1):
                end_point = 0
                end_list  = start_point
                i = 0
            elif p1[i] not in son1:
                son1[j] = p1[i]
                j = j + 1

            if j == len(p1):
                j = 0
            print("filho1", son1)

        print("filho1", son1)

        """for i in range(0, start_point):
            son1[i] = p1[i]
        print("filho1", son1)"""

        return son1, son2

    def selection_process(self, list, fitn):

        i1, i2 = random.sample(list, 2)
        # Selection process - It want to minimize
        if fitn[i1] > fitn[i2]:
            pai1 = i2
        elif fitn[i1] < fitn[i2]:
            pai1 = i1
        else:
            pai1 = i1  # depois colocar uma escolha aleatória caso algo de errado
        i1, i2 = random.sample(list, 2)
        if fitn[i1] > fitn[i2]:
            pai2 = i2
        elif fitn[i1] < fitn[i2]:
            pai2 = i1
        else:
            pai1 = i1
        return pai1, pai2

    def plot(self, individual):
        ###################################
        # TODO
        ###################################
        pass
