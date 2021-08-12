import random

def genetic_algorithm(problem, population_size, n_generations, mutation_rate=0.1):

    # Create a initial population
    ini_population = problem.initial_population(population_size)
    best_fit, best_pos, path, fitn = problem.fitness(ini_population)

    list = []
    for i in range(0, len(fitn)): list.append(i)

    for j in range(0, len(fitn)):

        # retorna a posição de onde os pais estão dentro da lista da população
        pai1, pai2 = problem.selection_process(list, fitn)

        # Realiza o Crossover - Order 1
        problem.crossover(ini_population[pai1], ini_population[pai2])

    return []
