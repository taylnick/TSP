#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq as hq
import itertools
from operator import itemgetter


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
        # TODO: implement this function
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        # Firstly, run the greedy algorithm to get the result. That solution is the BSSF
        bssf = self.greedy()['soln'].cost
        start_time = time.time()
        # Initialize rcm
        # then pick a node. calculate the reduced cost matrix (RCM) for it which returns the lowerbound.
        start = self.init_RCM(ncities)
        # init returns a tuple of [lower bound, rcm, and route]
        start[2].append(cities[0])
        stateq = [start]
        while not foundTour and stateq.__len__() != 0:
            # pop state off of the queue
            next_state = hq.heappop(stateq)
            # compare lowerbound to BSSF and ditch it if necessary
            if next_state[0] > bssf:
                continue
            next_city = next_state[2][-1]
            new_states = []
            # follow each edge and update RCM and update the lowerbound for each new state.
            # TODO: Update RCM
            for dest in range(ncities):
                # give it the current state, with the next city ID, and BSSF
                if next_city.costTo(cities[dest]) != np.inf:
                    new_state = self.update_RCM(next_state, cities[dest], bssf, ncities)
                    if new_state is not None:
                        new_states.append(new_state)
            new_states.sort(key=itemgetter[0])
            # push new states onto the queue
            for next in new_states:
                popped = new_states.pop(0)
                hq.heappush(stateq, popped)
        # Repeat until the queue is empty.

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        pass

        ''' Use this method to calculate the edges of the graph. 
            Row is the source and cols are the edges.
        '''
        # TODO: create a np_calculateEdges that returns a np array

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
        # reduce cols
        smallest_of_each_col = np.amin(rcm, 0)
        for j in range(size):
            small_num = smallest_of_each_col[j]
            lower_bound += small_num
            for i in range(size):
                val = rcm[i][j]
                if val == np.inf:
                    continue
                elif val >= small_num:
                    rcm[i][j] -= small_num
                else:
                    rcm[i][j] = 0
        smallest_of_each_row = np.amin(rcm, 1)
        # reduce rows
        for i in range(size):
            small_num = smallest_of_each_row[i]
            lower_bound += small_num
            for j in range(size):
                val = rcm[i][j]
                if val == np.inf:
                    continue
                elif val >= small_num:
                    rcm[i][j] -= small_num
                else:
                    rcm[i][j] = 0

        return [lower_bound, rcm, []]

    def update_RCM(self, state, next_city, bssf, size):
        lower_bound = state[0]
        rcm = state[1]
        route = state[2]
        from_city = route[-1]
        from_city_id = from_city.getIndex()
        to_city_id = next_city.getIndex()
        lower_bound += (rcm[from_city_id][to_city_id])
        # compare lowerbound to BSSF and prune if necessary
        lower_bound += rcm[from_city_id][to_city_id]
        if lower_bound >= bssf:
            return None

        for i in range(size):
            rcm[from_city_id][i] = np.inf
            rcm[i][to_city_id] = np.inf
        route.append(next_city)

        return [lower_bound, rcm, route]
