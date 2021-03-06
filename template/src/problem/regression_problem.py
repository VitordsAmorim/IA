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
        fitn, fx = [], []
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
        return best_fit, best_pos, path, fitn,

    def elitism(self, newpopulation, bestindividualold, fitn):
        # Encontra o pior individuo da geralção atual e o substitui
        # pelo melhor indivíduo da geração anterior
        bigger_fitn = max(fitn)
        pos = fitn.index(bigger_fitn)
        newpopulation[pos] = bestindividualold
        return newpopulation

    def mutation(self, individual, mutation_rate):
        mutation = np.random.random_sample()
        ind = individual
        if mutation < mutation_rate:
            # Em 50% dos casos: altera um valor para outro no intervalo de (-100 a 100)
            prob = random.random()
            if prob < 0.5:
                rand_pos1 = np.random.randint(9)
                ind[rand_pos1] = random.uniform(-100, 100)
            # Nos outros 50% dos casos somasse a cada um dos 9 valores um valor
            # calculado pela distribuição normal, de -1 a 1
            else:
                for i in range(0, 9):
                    xm = np.random.normal(scale=1)
                    ind[i] = ind[i] + xm
        return ind

    def crossover(self, p1, p2):
        son1, son2, alpha = ([0] * len(p1)), ([0] * len(p1)), []
        alpha = random.random()
        for k in range(0, len(p1) - 1):
            pai1 = p1[k]
            pai2 = p2[k]
            son1[k] = pai1 * alpha + pai2 * (1 - alpha)
            son2[k] = pai2 * alpha + pai1 * (1 - alpha)
        return son1, son2

    def subcrossover(self, start_point, end_point, p1, p2, son1):
        pass

    def selection_process(self, fitn):
        lista = []
        for i in range(0, len(fitn)):
            lista.append(i)

        # Escolhe aleatoriamente dois individuos para competirem
        # e o que tiver menor valor de fitness vence, tornando-se o pai1
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

        best_f, generation = best_fitness, ngeracoes
        ngeracoes = np.asarray(ngeracoes)
        xi = ngeracoes

        y0 = np.asarray(best_f[0])
        y1 = np.asarray(best_f[1])
        y2 = np.asarray(best_f[2])
        y3 = np.asarray(best_f[3])
        y4 = np.asarray(best_f[4])

        media = (y0 + y1 + y2 + y3 + y4) / 5

        fig, ax = plt.subplots()
        # ax.plot(xi, yi)
        ax.plot(xi, y0, 'b-',
                xi, y1, 'b-',
                xi, y2, 'b-',
                xi, y3, 'b-',
                xi, y4, 'b-',
                xi, media, 'r-',
                )
        ax.set(xlabel='gerações', ylabel='fitness',
               title='Regressão com função periodica')
        ax.grid()

        plt.savefig("Image/regression_fitness")
        plt.clf()
        pass

    def plot_bestfit(self, best_individual, rnd):
        yfx = []
        population = best_individual
        for k in range(0, len(self.x)):
            fxi = (population[0]
                   + population[1] * np.sin(self.x[k]) + population[2] * np.cos(self.x[k])
                   + population[3] * np.sin(2 * self.x[k]) + population[4] * np.cos(2 * self.x[k])
                   + population[5] * np.sin(3 * self.x[k]) + population[6] * np.cos(3 * self.x[k])
                   + population[7] * np.sin(4 * self.x[k]) + population[8] * np.cos(4 * self.x[k])
                   )
            yfx.append(fxi)

        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.plot(self.x, self.y, 'o', label='Experimental data')  # Plot some data on the axes.
        ax.plot(self.x, yfx, label='Regression')  # Plot more data on the axes...
        ax.set_xlabel('x')  # Add an x-label to the axes.
        ax.set_ylabel('y')  # Add a y-label to the axes.
        ax.set_title("Regressão com função periodica")  # Add a title to the axes.
        ax.legend()  # Add a legend.
        #plt.show()

        name = "regression_" + str(rnd)
        plt.savefig("Image/" + name)
        plt.clf()
        pass
