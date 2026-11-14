"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 7: Aerial Firefighting
Spring 2026 Semester

"""
from abc import ABC, abstractmethod
import numpy as np

class GeneticAlgorithm(ABC):

    def __init__(self, parameters):
        self.parameters = parameters
        for key, value in parameters.items():
            setattr(self, key, value)

        assert self.P >= 2, f"Need at least 2 parents to breed (P={self.P})"
        assert self.K % 2 == 0, f"Children are bred in pairs, so K must be even (K={self.K})"
        assert self.P + self.K <= self.S, f"P + K must not exceed S (P={self.P}, K={self.K}, S={self.S})"

        self.strings = [np.array([])]*self.S
        self.costs_of_current_generation = [0]*self.S
        self.best_cost = []
        self.average_cost = []
        self.parents_average_cost = []
        self.best_p_strings = []


    @abstractmethod
    def initialize_special_params(self):
        pass
    
    @abstractmethod
    def _generate_design_string(self):
        pass

    @abstractmethod
    def _evaluate_costs(self, start):
        pass


    def _sort_strings_and_costs(self):
        self.sorted_indices = np.argsort(self.costs_of_current_generation)
        self.strings = [self.strings[i] for i in self.sorted_indices]
        self.costs_of_current_generation = [self.costs_of_current_generation[i] for i in self.sorted_indices]

    def _mix_parents(self):
        rng = np.random.default_rng() # create default generator
        # Generate K children into slots P..P+K-1; cycle through parent pairs if K > P.
        for i in range(0, self.K, 2):
            p1, p2 = i % self.P, (i+1) % self.P
            phi = rng.random(size=self.numLam)
            psi = rng.random(size=self.numLam)
            self.strings[self.P+i] = phi*self.strings[p1] + (1-phi)*self.strings[p2]
            self.strings[self.P+i+1] = psi*self.strings[p1] + (1-psi)*self.strings[p2]

    def _add_immigrants(self):
        random_count = self.S - self.P - self.K
        for i in range(random_count):
            self.strings[i+self.P + self.K] = self._generate_design_string()

    def _store_data(self):
        self.average_cost.append(sum(self.costs_of_current_generation)/self.S)
        self.best_cost.append(self.costs_of_current_generation[0])
        self.parents_average_cost.append(sum(self.costs_of_current_generation[:self.P]) / self.P)
        self.best_p_strings = self.strings[:self.P]

    def GA(self, print_bool=False):
        for i in range(self.S):
            self.strings[i] = self._generate_design_string()

        # Generation 1: evaluate every design string.
        self._evaluate_costs(0)
        self._sort_strings_and_costs()
        self._store_data()
        if self.best_cost[-1] < self.tolerance:
            data = {}
            data['best_cost'] = self.best_cost
            data['average_cost'] = self.average_cost
            data['parents_average_cost'] = self.parents_average_cost
            data['best_p_strings'] = self.best_p_strings
            return data

        # Later generations: preserve parent costs, evaluate only new strings.
        g = 2
        while g <= self.G:
            if print_bool:
                print(f'Generation : {g}')
                print(f'Best Cost  : {self.best_cost[-1]}')
            self._mix_parents()
            self._add_immigrants()
            self._evaluate_costs(self.P)
            self._sort_strings_and_costs()
            self._store_data()
            if self.best_cost[-1] < self.tolerance:
                break
            g+=1
        
        data = {}
        data['best_cost'] = self.best_cost
        data['average_cost'] = self.average_cost
        data['parents_average_cost'] = self.parents_average_cost
        data['best_p_strings'] = self.best_p_strings
        return data
    
    @abstractmethod
    def specialGA(self):
        pass