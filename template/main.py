
import argparse
from src.problem.regression_problem import RegressionProblem
from src.problem.classification_problem import ClassificationProblem
from src.problem.tsp_problem import TSPProblem
from src.algorithm.genetic_algorithm import genetic_algorithm
import time


def generate_report(output, problem, n_generations, best_fitness):
    problem.plot(output, n_generations)
    menorfit = min(best_fitness)
    meiorfit = max(best_fitness)
    media = sum(best_fitness)/n_generations
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

    parser.add_argument('-p', '--problem', default='tsp',
                        choices=["classification", "regression", "tsp"])
    parser.add_argument('-n', '--n_generations', type=int,
                        default=100, help='number of generations.')
    parser.add_argument('-s', '--population_size', type=int,
                        default=50, help='population size.')
    parser.add_argument('-m', '--mutation_rate', type=float,
                        default=0.2, help='mutation rate.')

    args = parser.parse_args()
    return args


def main():

    args = read_command_line_args()
    problem = build_problem(args.problem)

    graph, best_fitness = [], []
    for i in range(0, 5):
        output, n = genetic_algorithm(
            problem,
            population_size=args.population_size,
            n_generations=args.n_generations,
            round=i)
        graph.append(output)
        best_fitness.append(output[args.n_generations - 1])
    generate_report(graph, problem, n, best_fitness)

    print("OK!")


if __name__ == "__main__":
    main()
