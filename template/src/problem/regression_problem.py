import numpy as np
from template.src.problem.problem_interface import ProblemInterface
import random
import matplotlib.pyplot as plt


class RegressionProblem(ProblemInterface):
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
        population, individual = [], []
        for j in range(0, population_size):
            for i in range(0, 9):
                individual.append(random.uniform(-100, 100))
            population.append(individual)
            individual = []
        return population

    def fitness(self, population):
        # Calcular a função fitness para toda população
        fitn = []
        for i in range(0, len(population)):
            erro = 0
            # Representa a fitness the um indivíduo
            for k in range(0, len(self.x)):
                yo = self.y[k]
                fxi = (population[i][0]
                       + population[i][1] * np.sin(self.x[k]) + population[i][2] * np.cos(self.x[k])
                       + population[i][3] * np.sin(2 * self.x[k]) + population[i][4] * np.cos(2 * self.x[k])
                       + population[i][5] * np.sin(3 * self.x[k]) + population[i][6] * np.cos(3 * self.x[k])
                       + population[i][7] * np.sin(4 * self.x[k]) + population[i][8] * np.cos(4 * self.x[k])
                       )
                erro = erro + (yo - fxi) ** 2
            erro_mq = float(erro / len(self.x))
            fitn.append(erro_mq)
        best_fit = min(fitn)
        best_pos = fitn.index(best_fit)
        path = population[best_pos]
        return best_fit, best_pos, path, fitn

    def elitism(self, newpopulation, bestindividualold, fitn):

        # Encontrar a pior solução, para depois
        # substitui-la pela melhor solução da geração anterior
        bigger_fitn = max(fitn)
        pos = fitn.index(bigger_fitn)
        newpopulation[pos] = bestindividualold
        return newpopulation

    def mutation(self, individual, mutation_rate):
        lista = []

        # Em 50% dos casos:
        # muda uma coordenada para outro valor no intervalo de (-100 a 100)
        prob = random.random()
        if prob < 0.5:

            print("Individual before the mutation: ")
            print("    ", individual)

            random.uniform(0, 8)


            for i in range(0, 20):
                lista.append(np.random.randint(9))

            # Randomly select two positions
            rand_pos1, rand_pos2 = random.sample(lista, 2)

            # Swap the position of two cities
            aux = individual[rand_pos1]
            individual[rand_pos1] = individual[rand_pos2]
            individual[rand_pos2] = aux
        else:
            print("Individual after the mutation: ")
            print("    ",individual)
        return individual

    def crossover(self, p1, p2):
        # Randomly select a value for alpha
        son1, son2, alpha = ([0] * len(p1)), ([0] * len(p1)), []

        alpha = random.random()
        for k in range(0, len(p1) - 1):
            pai1 = p1[k]
            pai2 = p2[k]
            son1[k] = pai1 * alpha + pai2 * (1 - alpha)
            son2[k] = pai2 * alpha + pai1 * (1 - alpha)
        """print(p1, "\n", p2)
        print("Alpha: ", alpha)
        print(son1)"""
        return son1, son2

    def subcrossover(self, start_point, end_point, p1, p2, son1):
        pass

    def selection_process(self, fitn):
        lista = []
        for i in range(0, len(fitn)):
            lista.append(i)

        # Escolhe aleatoriamente dois individuos para competirem
        # e o que tiver menor valor de fitness vence, tornando-se
        # o pai 1
        i1, i2 = random.sample(lista, 2)
        # Selection process - It want to minimize
        if fitn[i1] > fitn[i2]:
            pai1 = i2
        elif fitn[i1] < fitn[i2]:
            pai1 = i1
        else:
            pai1 = i1  # depois colocar uma escolha aleatória caso algo de errado

        # Escolhe aleatoriamente outros dois individuos para competirem
        # e o que tiver menor valor de fitness vence, tornando-se
        # o pai 2
        i1, i2 = random.sample(lista, 2)
        if fitn[i1] > fitn[i2]:
            pai2 = i2
        elif fitn[i1] < fitn[i2]:
            pai2 = i1
        else:
            pai2 = i1
        return pai1, pai2

    def plot(self, best_fitness, ngeracoes):

        best_fitness = np.asarray(best_fitness)
        ngeracoes = np.asarray(ngeracoes)
        x = ngeracoes
        y = best_fitness
        plt.plot(x, y)
        plt.title("TSP - 30 cities")
        plt.xlabel("Nº Generations")
        plt.ylabel("Fitness")
        plt.show()
        pass

    def plot_bestfit(self, best_individual):
        xp, yp = [], []
        best_individual.insert(0, 0)
        best_individual.insert(len(best_individual), 0)
        for k in range(0, len(best_individual)):
            indice = best_individual[k]
            xp.append(float(self.x[indice]))
            yp.append(float(self.y[indice]))

        x = np.asarray(xp)
        y = np.asarray(yp)
        plt.plot(x, y, marker="o", markerfacecolor="r")
        plt.show()
        pass
