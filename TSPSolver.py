#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
# import numpy as np
from TSPClasses import *
from queue import LifoQueue
import copy
from random import randint
from operator import attrgetter
from statistics import mean


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    # Use this as a template for implementing other versions of the tour.
    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
            This is the entry point for the greedy solver, which you must implement for 
            the group project (but it is probably a good idea to just do it for the branch-and
            bound project as a way to get your feet wet).  Note this could be used to find your
            initial BSSF.
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of best solution, 
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this 
            algorithm</returns> 
        '''

    def greedy(self, time_allowance=60.0):
        # This is part of Project 6 but is helpful to do for Project 5.
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        distances = self.calculateEdges(ncities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        for i in range(ncities):
            # Implement Greedy approach here.
            # From the starting city, follow the shortest edges to undiscovered nodes
            while not foundTour and time.time() - start_time < time_allowance:
                curr = cities[i]
                curr_id = curr.getIndex()
                route = [curr]
                while len(route) < ncities:
                    index = 0
                    shortest = np.inf
                    for j in range(ncities):
                        if distances[curr_id][j] < shortest and not cities[j] in route:
                            shortest = distances[curr_id][j]
                            index = j
                    if distances[index] == np.inf:
                        break
                    else:
                        curr = cities[index]
                        curr_id = curr.getIndex()
                        route.append(curr)
                curr_solution = TSPSolution(route)
                if curr_solution.cost < np.inf:
                    foundTour = True
                    count += 1
                    if bssf is None:
                        bssf = curr_solution
                    elif curr_solution.cost < bssf.cost:
                        bssf = curr_solution
                break

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

        ''' <summary>
            This is the entry point for the branch-and-bound algorithm that you will implement
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of best solution, 
            time spent to find best solution, total number solutions found during search (does
            not include the initial BSSF), the best solution found, and three more ints: 
            max queue size, total number of states created, and number of pruned states.</returns> 
        '''

    def branchAndBound(self, time_allowance=60.0):
        pruned = 0
        max_states = 0
        total_states = 0

        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        # Firstly, run the greedy algorithm to get the result. That solution is the BSSF
        greedy_resuts = self.greedy()
        bssf = greedy_resuts['soln'].cost
        best_solution = copy.copy(greedy_resuts['soln'])
        start_time = time.time()
        # Initialize rcm
        # then pick a node. calculate the reduced cost matrix (RCM) for it which returns the lowerbound.
        start_state = self.init_RCM(ncities)
        # init returns a tuple of [lower bound, rcm, and route]
        start_state[2].append(cities[0])
        stateq = LifoQueue()
        stateq.put(start_state)
        # and time.time() - start_time < time_allowance
        while stateq.qsize() != 0 and time.time() - start_time < time_allowance:
            # pop state off of the queue
            curr_state = stateq.get()

            # compare lower_bound to BSSF and ditch it if necessary
            if curr_state[0] > best_solution.cost:
                pruned += 1
                continue
            curr_city = curr_state[2][-1]
            curr_city_id = curr_city.getIndex()
            new_states = []
            # follow each edge and update
            # RCM and update the lowerbound for each new state.

            for dest in range(ncities):
                # give it the current state, with the next city ID, and BSSF

                cost = curr_state[1][curr_city_id][dest]
                if cost != np.inf:
                    new_state = update_RCM(curr_state, cities[dest], cost, bssf, ncities)
                    total_states += 1
                    if new_state is not None:
                        if len(new_state[2]) == ncities:
                            curr_soln = TSPSolution(new_state[2])
                            if curr_soln.cost != np.inf:
                                if curr_soln.cost < best_solution.cost:
                                    best_solution = copy.copy(curr_soln)
                                count += 1
                                foundTour = True
                                continue
                            else:
                                pruned += 1
                        else:
                            new_states.append(new_state)
                    else:
                        pruned += 1
            new_states.sort(key=lambda new_states: new_states[0])
            # push new states onto the queue
            while len(new_states) != 0:
                popped = new_states.pop()
                stateq.put(popped)
            if stateq.qsize() > max_states:
                max_states = stateq.qsize()
        # Repeat until the queue is empty.
        end_time = time.time()
        results['cost'] = best_solution.cost if foundTour else bssf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = best_solution
        results['max'] = max_states
        results['total'] = total_states
        results['pruned'] = pruned
        return results

    ''' <summary>
    Genetic Algorithm implementation. Thanks to this article for help: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found during search, the 
    best solution found.  You may use the other three field however you like.
    algorithm</returns> 
    '''
    def fancy(self, time_allowance=60.0):
        # Needed variables
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        pop_size = 15
        elite_size = int(pop_size / 2)
        count = 0
        mutation_rate = 0.15

        # Initialize population
        if ncities < 50:
            population = self.initializePopulation(pop_size)
        else:
            population = self.initializeLargePopulation(pop_size, ncities)

        population.sort(key=lambda p: p.cost)
        bssf = TSPSolution(population[0].route)

        start_time = time.time()

        stagnant_generations = 0
        prev_generation_leader = np.inf
        # The number of generations/iterations of the genetic algorithm
        while stagnant_generations < 20000 and time.time() - start_time < time_allowance:
            # cull population back down to size.
            population = self.cullPopulation(population, pop_size, elite_size)
            # Make some chilluns
            population = self.breedPopulation(population, elite_size)
            # Mutate the population
            self.mutatePopulation(population, pop_size, ncities, mutation_rate)
            # Sort it again
            population.sort(key=lambda p: p.cost)

            curr_leader = population[0].cost
            if curr_leader < np.inf:
                if curr_leader < bssf.cost:
                    bssf = population[0]
                    stagnant_generations = 0
                    count += 1
                elif curr_leader == bssf.cost:
                    stagnant_generations += 1
                elif curr_leader == prev_generation_leader:
                    stagnant_generations += 1
            else:
                stagnant_generations = 0

            prev_generation_leader = copy.copy(population[0].cost)

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    '''The top elite will always survive. Then randomly choose some peeps to make babies.'''
    def breedPopulation(self, pop, elites):
        pop.sort(key=lambda p: p.cost)
        length = len(pop) - elites
        # Right now this is taking everybody. It shouldn't be taking everybody to the party.
        forest_dwellers = random.sample(pop, len(pop))
        survivors = []

        # The best handful always get passed on.
        for i in range(0, elites):
            survivors.append(pop[i])

        for i in range(0, length):
            oops = self.mate(forest_dwellers[i], forest_dwellers[len(pop) - i - 1])
            survivors.append(oops)

        return survivors

    '''take a random subset of cities from mom, and fill in the missing pieces from dad in the order they appear.
    This is a special ritual called ordered crossover.'''
    def mate(self, mom, dad):
        final_product = []
        moms_contribution = []
        dads_contribution = []

        # find two random indices
        geneA = int(random.random() * len(mom.route))
        geneB = int(random.random() * len(mom.route))
        # figure out which is smaller
        beginCity = min(geneA, geneB)
        endCity = max(geneA, geneB)

        # Start with a subset from mom
        for i in range(beginCity, endCity):
            moms_contribution.append(mom.route[i])
        # Find the rest from dad
        dads_contribution = [city for city in dad.route if city not in moms_contribution]

        # put them together and build a TSP Solution
        final_product = moms_contribution + dads_contribution
        baby_solution = TSPSolution(final_product)
        return baby_solution

    '''Cull population back down to size.'''
    def cullPopulation(self, pop, pop_size, elites):
        culled_pop = []
        pop.sort(key=lambda x: x.cost)
        for i in range(0, elites):
            culled_pop.append(pop[i])
        # now choose some randos to fill in the extra space. They got lucky.
        while len(culled_pop) < pop_size:
            rando = randint(elites, len(pop) - 1)
            if pop[rando] not in culled_pop:
                culled_pop.append(pop[rando])
        return culled_pop

    '''Takes population size, makes random solutions.
    The first elites are added to a new list, 
    and then random entries are chosen to reach the desired list length'''

    def initializePopulation(self, pop_size):
        init_pop = []
        for i in range(pop_size):
            default_results = self.defaultRandomTour()
            # TSPSolution object
            init_pop.append(default_results['soln'])
        return init_pop

    def initializeLargePopulation(self, pop_size, ncities):
        init_pop = []
        half = int(ncities/2) - 1
        # for i in range(3):
        #     default_results = self.defaultRandomTour()
        #     # TSPSolution object
        #     init_pop.append(default_results['soln'])
        greedy_results = self.greedy()

        init_pop.append(greedy_results['soln'])
        while len(init_pop) < pop_size:
            perm = self.randomPerm()
            init_pop.append(perm)
        return init_pop
    '''mutateGene is called if a solution is randomly selected to undergo mutation.'''

    def randomPerm(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        bssf = None
        # create a random permutation
        perm = np.random.permutation(ncities)
        route = []
        # Now build the route using the random permutation
        for i in range(ncities):
            route.append(cities[perm[i]])
        bssf = TSPSolution(route)
        return bssf

    def mutateGene(self, tsp_soln, ncities):
        soln = tsp_soln.route

        # Randomly pick two cities
        rand_num_1 = randint(0, ncities - 1)
        rand_num_2 = randint(0, ncities - 1)
        if rand_num_1 != rand_num_2:
            # Swap cities
            soln[rand_num_1], soln[rand_num_2] = soln[rand_num_2], soln[rand_num_1]
        mutated_soln = TSPSolution(soln)
        return mutated_soln

    def mutatePopulation(self, pop, pop_size, ncities, mutation_rate):
        for i in range(1, pop_size):
            # random.random() returns a random value between 0.0 and 1.0
            if random.random() < mutation_rate:
                # Mutate this entry
                pop[i] = self.mutateGene(pop[i], ncities)

    ''' Use this method to calculate the edges of the graph. 
        Row is the source and cols are the edges.
    '''

    def calculateEdges(self, size):
        cities = self._scenario.getCities()
        distances = [[np.inf] * size for x in range(size)]
        for source in range(size):
            for dest in range(size):
                distances[source][dest] = cities[source].costTo(cities[dest])
        return distances

    def init_RCM(self, size):
        distances = self.calculateEdges(size)
        rcm = np.array([x for x in distances])
        lower_bound = 0

        lower_bound = reduce_rows(lower_bound, rcm, size)
        lower_bound = reduce_cols(lower_bound, rcm, size)

        return_state = [lower_bound, rcm, []]
        return return_state


def update_RCM(state, next_city, cost, bssf, size):
    lower_bound = copy.copy(state[0])
    rcm = copy.copy(state[1])
    route = copy.copy(state[2])
    from_city = route[-1]
    from_city_id = from_city.getIndex()
    to_city_id = next_city.getIndex()
    lower_bound += cost

    for i in range(size):
        rcm[from_city_id][i] = np.inf
        rcm[i][to_city_id] = np.inf

    # set the back edge to inf.
    rcm[to_city_id][from_city_id] = np.inf
    # Reduce cols and rows again.
    lower_bound = reduce_cols(lower_bound, rcm, size)
    lower_bound = reduce_rows(lower_bound, rcm, size)

    # compare lowerbound to BSSF and prune if necessary
    if lower_bound >= bssf:
        return None

    route.append(copy.copy(next_city))
    ret_state = [lower_bound, rcm, route]
    return ret_state


def reduce_rows(lower_bound, rcm, size):
    smallest_of_each_row = np.amin(rcm, 1)
    # reduce rows
    for i in range(size):
        small_num = smallest_of_each_row[i]
        if small_num > 0 and small_num != np.inf:
            lower_bound += small_num
            for j in range(size):
                rcm[i][j] -= small_num
    return lower_bound


def reduce_cols(lower_bound, rcm, size):
    # reduce cols
    smallest_of_each_col = np.amin(rcm, 0)
    for j in range(size):
        small_num = smallest_of_each_col[j]
        if small_num > 0 and small_num != np.inf:
            lower_bound += small_num
            for i in range(size):
                rcm[i][j] -= small_num
    return lower_bound
