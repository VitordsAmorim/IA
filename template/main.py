import csv
import argparse
from src.problem.regression_problem import RegressionProblem
from src.problem.classification_problem import ClassificationProblem
from src.problem.tsp_problem import TSPProblem
from src.algorithm.genetic_algorithm import genetic_algorithm
import time
import numpy as np


def generate_report(output, problem, n_generations, best_fitness, deltat, typeproblem):
    problem.plot(output, n_generations)
    menorfit = min(best_fitness)
    maiorfit = max(best_fitness)
    media = sum(best_fitness)
    media = media/5
    delta_time = deltat/5

    dp = 0.0
    for k in range(0, 5):
        dp = dp + (best_fitness[k] - media)**2
    dp = dp/5
    dp = np.sqrt(dp)

    f = open('Output.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow([typeproblem, menorfit, maiorfit, media, dp, delta_time])
    f.close()
    pass


def build_problem(problem_name):
    if problem_name == "classification":
        return ClassificationProblem("data/german_statlog/german.data-numeric")
    elif problem_name == "regression":
        return RegressionProblem("data/regression/data-3.txt")
    elif problem_name == "tsp":
        return TSPProblem("data/tsp/tsp-30.txt")
    else:
        raise NotImplementedError()


def read_command_line_args():
    parser = argparse.ArgumentParser(
        description='Optimization with genetic algorithms.')

    parser.add_argument('-p', '--problem', default='classification',
                        choices=["classification", "regression", "tsp"])
    parser.add_argument('-n', '--n_generations', type=int,
                        default=10, help='number of generations.')
    parser.add_argument('-s', '--population_size', type=int,
                        default=10, help='population size.')
    parser.add_argument('-m', '--mutation_rate', type=float,
                        default=0.2, help='mutation rate.')

    args = parser.parse_args()
    return args


def main():

    '''args = read_command_line_args()
    problem = build_problem(args.problem)'''
    f = open('Output.csv', 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['Tipo do Problema', 'Menor fitness', 'maior fitness', 'media', 'desvio padrao', 'intervalo de tempo'])
    f.close()

    typeproblem = ["tsp", "regression", "classification"]
    population_s = [200, 200, 100]
    ngeracoes = [1000, 1000, 250]

    for t in range(0, 3):
        problem = build_problem(typeproblem[t])
        graph, best_fitness = [], []
        t_inicial = time.time()
        for i in range(0, 5):
            output, n = genetic_algorithm(
                problem,
                population_size=population_s[t],
                n_generations=ngeracoes[t],
                round=i)
            graph.append(output)
            best_fitness.append(output[ngeracoes[t] - 1])
        t_final = time.time()
        deltat = t_final - t_inicial
        generate_report(graph, problem, n, best_fitness, deltat, typeproblem[t])
    print("OK!")


if __name__ == "__main__":
    main()
