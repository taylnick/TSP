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
    This is the entry point for the algorithm you'll write for your group project.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found during search, the 
    best solution found.  You may use the other three field however you like.
    algorithm</returns> 
    '''
    def fancy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        pop_size = 15
        past_gen_maxes = [5]

        generations = 50
        population = self.initializePopulation(pop_size)
        population.sort(key=lambda p: p.cost)
        # TODO: use this or get rid of it
        bssf = TSPSolution(population[0].route)
        past_gen_maxes.append(bssf)
        start_time = time.time()

        # The number of generations/iterations of the genetic algorithm
        still_improving = True
        while still_improving and time.time() - start_time < time_allowance:
            population = self.createNewPopulation(population, pop_size, ncities, bssf, past_gen_maxes, time_allowance, start_time)
            # min_cost = min(population, key=attrgetter('cost'))
            still_improving = self.is_improving(population, past_gen_maxes)

        # Sort population by cost
        population.sort(key=lambda p: p.cost)
        end_time = time.time()
        results['cost'] = population[0].cost
        results['time'] = end_time - start_time
        results['count'] = pop_size
        results['soln'] = TSPSolution(population[0].route)
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def is_improving(self, pop, max_list):

        # pop.sort(key=lambda p: p.cost)
        # min_this_iteration = pop[0]
        # if len(max_list) < 5:
        #     max_list.append(min_this_iteration)
        # else:
        #     max_list.sort(key=lambda l: l.cost)
        #     if min_this_iteration.cost < max_list[-1].cost:
        #         max_list.pop(-1)
        #         max_list.append(min_this_iteration)
        # average_cost = mean(sol.cost for sol in max_list)
        # if average_cost < (max_list[])
        return pop[0].cost

    '''Takes population size, makes random solutions. 
        returns a list of TSPSolution objs'''
    def initializePopulation(self, pop_size):
        init_pop = []
        for i in range(pop_size*2):
            default_results = self.defaultRandomTour()
            # Tuple of (cost, solution) # TSPSolution object
            init_pop.append(default_results['soln'])
        init_pop.sort(key=lambda l: l.cost)
        return init_pop[:pop_size]

    def mutateGene(self, tsp_soln, ncities):
        soln = tsp_soln.route

        # Percentage of mutations performed on the solution
        mutation_rate = 0.20
        # Number of mutations to make on the solution
        num_of_mutations = np.ceil(ncities * mutation_rate)
        i = 0
        while i < num_of_mutations:
            # Randomly pick two cities
            rand_num_1 = randint(0, ncities-1)
            rand_num_2 = randint(0, ncities-1)
            if rand_num_1 != rand_num_2:
                # Swap cities
                soln[rand_num_1], soln[rand_num_2] = soln[rand_num_2], soln[rand_num_1]
                i += 1
        return soln

    def calculateFitness(self, soln):
        results = TSPSolution(soln)
        return results.cost

    def createNewPopulation(self, init_pop, pop_size, ncities, bssf, best_past, time_allowance, start_time):
        new_pop = []

        # To create a new population of equal size
        for i in range(pop_size):
            if time.time() - start_time < time_allowance:
                parent = init_pop[i]
                # Mutate parent
                isFound = False
                while not isFound:
                    if time.time() - start_time < time_allowance:
                        child_soln = self.mutateGene(parent, ncities)
                        # Calculate Fitness
                        child_cost = self.calculateFitness(child_soln)
                        # Add the parent or the child with the better cost
                        if child_cost < parent.cost:
                            # Tuple of (cost, solution)
                            new_pop.append(TSPSolution(child_soln))
                            isFound = True
                    else:
                        break
            else:
                return init_pop
        return new_pop

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

