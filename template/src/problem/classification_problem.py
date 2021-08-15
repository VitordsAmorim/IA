import random
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from template.src.problem.problem_interface import ProblemInterface
import matplotlib.pyplot as plt

class ClassificationProblem(ProblemInterface):
    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()

        # For each line l, remove the "\n" at the end using
        # rstrip and then split the string the spaces. After
        # this instruction, lines is a list in which each element
        # is a sublist of strings [s1, s2, s3, ..., sn].
        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.int32)

        # Split features (x) and labels (y). The notation x[:, i]
        # returns all values of column i. To learn more about numpy indexing
        # see https://numpy.org/doc/stable/reference/arrays.indexing.html .
        x = lines[:, :-1]
        y = lines[:, -1]

        # Split the data in two sets without intersection.
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(x, y, test_size=0.30,
                             stratify=y, random_state=871623)

        # number of features
        self.n_features = self.X_train.shape[1]

        # search space for the values of k and metric
        self.Ks = [1, 3, 5, 7, 9, 11, 13, 15]
        self.metrics = ["euclidean", "hamming", "canberra", "braycurtis"]

    def initial_population(self, population_size):
        population, individual = [], []
        for j in range(0, population_size):
            for i in range(0, 24):
                individual.append(random.randint(0, 1))
            individual.append(random.choice(self.Ks))
            individual.append(random.choice(self.metrics))
            population.append(individual)
            individual = []
        return population

    def fitness(self, population):

        fitn = []
        for i in range(0, len(population)):
            binary_pattern = population[i][:-2]
            K = population[i][-2]
            metric = population[i][-1]

            # return the indices of the features that are not zero.
            indices = np.nonzero(binary_pattern)[0]

            # check if there is at least one feature available
            if len(indices) == 0:
                return 1e6

            # select a subset of columns given their indices
            x_tr = self.X_train[:, indices]
            x_val = self.X_val[:, indices]

            # build the classifier
            knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
            # train
            knn = knn.fit(x_tr, self.y_train)
            # predict the classes for the validation set
            y_pred = knn.predict(x_val)
            # measure the accuracy
            acc = np.mean(y_pred == self.y_val)

            # since the optimization algorithms minimize,
            # the fitness is defiend as the inverse of the accuracy
            fitness = -acc
            fitn.append(fitness)
        best_fit = min(fitn)
        best_pos = fitn.index(best_fit)
        path = population[best_pos]
        return best_fit, best_pos, path, fitn

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
            rand_pos1 = np.random.randint(24)
            if ind[rand_pos1] == 1:
                ind[rand_pos1] = 0
            else:
                ind[rand_pos1] = 1
            ind[-2] = random.choice(self.Ks)
            ind[-1] = random.choice(self.metrics)
        return ind

    def crossover(self, p1, p2):
        cut = random.randint(1, len(p1)-3)
        son1, son2 = ([0] * len(p1)), ([0] * len(p1))
        for k in range(0, cut):
            son1[k] = p1[k]
            son2[k] = p2[k]
        for k in range(cut, len(p1)-2):
            son1[k] = p2[k]
            son2[k] = p1[k]
        son1[-2], son1[-1] = p1[-2], p2[-1]
        son2[-2], son2[-1] = p2[-2], p1[-1]
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
        best_fitness = np.asarray(best_fitness)
        ngeracoes = np.asarray(ngeracoes)
        x = ngeracoes
        y = best_fitness
        plt.plot(x, y)
        plt.title("Hyperparameter optimization")
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.show()
        pass

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
               title='"Busca de Hyperparametros"')
        ax.grid()

        plt.savefig("Image/Hyperparametros_fitness")
        plt.clf()
        pass

    def plot_bestfit(self, best_individual, rnd):
        f = open('Output.csv', 'a', newline='', encoding='utf-8')
        w = csv.writer(f)
        w.writerow(['Hyperparametros:', best_individual[-2:]])
        f.close()
        pass