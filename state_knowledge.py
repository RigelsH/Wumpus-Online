from typing import Iterable
import itertools

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import wumpus as wws

from utils import rotate_clockwise, rotate_counterclockwise


class Knowledge:
    """
    This class updates states of environment and hunter for each action.
    """
    def __init__(self):

        # agent related variables
        self.agent_x_coord = 0
        self.agent_y_coord = 0
        self.agent_orientation = (0, 1)
        self.is_arrow_available = True
        self.agent_legal_locations = {(0, 0)}
        self.visited_locations = set()

        # world related variables
        self.world_size_has_determined = False
        self.world_size = 1
        
        # pit
        self.pit_locations = set()
        self.no_pit_locations = set()
        
        # wumpus
        self.is_wumpus_alive = True
        self.wumpus_location = None
        self.wumpus_location_has_determined = False
        self.no_wumpus_locations = set()
        self.possible_wumpus_locations = set()

        # stench
        self.stench_locations = set()
        self.no_stench_locations = set()

        # breeze
        self.breeze_locations = set()
        self.no_breeze_locations = set()
        
        # adjacent_cell
        self.adjacent_cell = None

        # gold
        self.gold_has_grabbed = False

    def __str__(self):
        return f"" \
            + f"------------------------------------------------\n" \
            + f"-------------------- WORLD ---------------------\n" \
            + f"------------------------------------------------\n" \
            + f"world_size_has_determined = {self.world_size_has_determined}\n" \
            + f"world_size = {(self.world_size, self.world_size)}\n" \
            + f"------------------------------------------------\n" \
            + f"-------------------- HUNTER --------------------\n" \
            + f"------------------------------------------------\n" \
            + f"agent_location = {(self.agent_x_coord, self.agent_y_coord)}\n" \
            + f"agent_orientation = {self.agent_orientation}\n" \
            + f"visited_locations = {sorted(self.visited_locations)}\n" \
            + f"agent_legal_locations = {sorted(self.agent_legal_locations)}\n" \
            + f"adjacent_cell = {self.adjacent_cell}\n" \
            + f"gold_has_grabbed = {self.gold_has_grabbed}\n" \
            + f"------------------------------------------------\n" \
            + f"----------------- BREEZE/PITS ------------------\n" \
            + f"------------------------------------------------\n" \
            + f"breeze_locations = {sorted(self.breeze_locations)}\n" \
            + f"no_breeze_locations = {sorted(self.no_breeze_locations)}\n" \
            + f"no_pit_locations = {sorted(self.no_pit_locations)}\n" \
            + f"------------------------------------------------\n" \
            + f"------------------- WUMPUS ---------------------\n" \
            + f"------------------------------------------------\n" \
            + f"is_wumpus_alive = {self.is_wumpus_alive}\n" \
            + f"wumpus_location_has_determined = {self.wumpus_location_has_determined}\n" \
            + f"stench_locations = {self.stench_locations}\n" \
            + f"no_stench_locations = {self.no_stench_locations}\n" \
            + f"no_wumpus_locations = {sorted(self.no_wumpus_locations)}\n" \
            + f"possible_wumpus_locations = {sorted(self.possible_wumpus_locations)}\n" \
            + f"wumpus_location = {self.wumpus_location}\n" \
            + f"------------------------------------------------\n"
            

    def filter_out_of_bound_locations(self, locations):
        """
        Filters out of bounds location outside of the current world size.
        If we have already determined the world size, we will limit inner location
        by world size, otherwise we will limit it only by not negative number.
        """
        if self.world_size_has_determined:
            inner_locations = set(filter(lambda loc: 
                                         loc[0] >= 0 and loc[0] < self.world_size and 
                                         loc[1] >= 0 and loc[1] < self.world_size, locations))
        else:
            inner_locations = set(filter(lambda loc: 
                                         loc[0] >= 0 and loc[1] >= 0, locations))
        return inner_locations

    @staticmethod
    def get_adjacent_locations(x, y):
        """
        Get adjucent location for given location.
        """
        adjacent_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        adjacent_locations = set((x + dx, y + dy) for dx, dy in adjacent_offsets)
        return adjacent_locations
    
    def update_after_perception(self, previous_agent_action: wws.Agent.Actions, current_percept: wws.Percept):
        """
        This class updates knowledge based on hunters' percept.
        """
        if previous_agent_action is not None:

            if previous_agent_action == wws.Hunter.Actions.GRAB:
                # update gold state condition
                self.gold_has_grabbed = True
            
            if previous_agent_action == wws.Hunter.Actions.MOVE and not current_percept.bump:
                # update agent location based on orientation
                self.agent_x_coord += self.agent_orientation[0]
                self.agent_y_coord += self.agent_orientation[1]

            elif previous_agent_action == wws.Hunter.Actions.RIGHT:
                # turn orientation on the right 
                self.agent_orientation = rotate_clockwise(self.agent_orientation) 

            elif previous_agent_action == wws.Hunter.Actions.LEFT:
                # turn orientation on the left
                self.agent_orientation = rotate_counterclockwise(self.agent_orientation)

            elif previous_agent_action == wws.Hunter.Actions.SHOOT:
                self.is_arrow_available = False

                if current_percept.scream:
                    self.wumpus_location = set()
                    self.stench_locations = set()
                    self.wumpus_location_has_determined = True
                    self.is_wumpus_alive = False
                    self.possible_wumpus_locations = set()

                else:
                    self.adjacent_cell = (self.agent_x_coord + self.agent_orientation[0], 
                                          self.agent_y_coord + self.agent_orientation[1])

        if (self.agent_x_coord, self.agent_y_coord) in self.visited_locations:
            if current_percept.bump and ((self.agent_orientation == (0, 1)) or (self.agent_orientation == (1, 0))):
                self.world_size_has_determined = True
         
        if not self.world_size_has_determined:
            if not current_percept.bump and ((self.agent_x_coord >= self.world_size) or (self.agent_y_coord >= self.world_size)):
                self.world_size += 1

        # since until now we survived
        # we will update no-pit, no-wumpus locations and visited cells
        self.no_wumpus_locations.add((self.agent_x_coord, self.agent_y_coord))
        self.no_pit_locations.add((self.agent_x_coord, self.agent_y_coord))
        self.visited_locations.add((self.agent_x_coord, self.agent_y_coord))

        # define current neigbours (adjacent locations)
        neighbour_locations = self.get_adjacent_locations(x=self.agent_x_coord, y=self.agent_y_coord)
        neighbour_locations = self.filter_out_of_bound_locations(neighbour_locations)
        # print('neighbour_locations', neighbour_locations)

        # define current neigbours (adjacent locations) without visited locations
        not_visited_neighbour_locations = set(filter(lambda x: x not in self.visited_locations, neighbour_locations))
        not_visited_neighbour_locations = self.filter_out_of_bound_locations(locations=not_visited_neighbour_locations)
        # print('not_visited_neighbour_locations', not_visited_neighbour_locations)
        
        # update agent possible locations to move in the current world
        if (self.agent_x_coord, self.agent_y_coord) in self.agent_legal_locations:
            self.agent_legal_locations.remove((self.agent_x_coord, self.agent_y_coord))
        self.agent_legal_locations = self.agent_legal_locations.union(not_visited_neighbour_locations)

        if current_percept.breeze:
            # update breeze_locations (add current agent location)
            self.breeze_locations.add((self.agent_x_coord, self.agent_y_coord))
        else:
            # update no_breeze_locations (add current agent location)
            self.no_breeze_locations.add((self.agent_x_coord, self.agent_y_coord))
            # update no_pit_locations (add current neqigbours)
            self.no_pit_locations = self.no_pit_locations.union(not_visited_neighbour_locations)

        if current_percept.stench:
            # update stench_locations (add current agent location)
            self.stench_locations.add((self.agent_x_coord, self.agent_y_coord))
            # update possible_wumpus_locations
            if not self.wumpus_location_has_determined:
                self.possible_wumpus_locations = self.possible_wumpus_locations.union(not_visited_neighbour_locations)
                self.possible_wumpus_locations = self.possible_wumpus_locations.difference(self.no_wumpus_locations)
                self.possible_wumpus_locations = self.possible_wumpus_locations.difference({(self.agent_x_coord, self.agent_y_coord)})
                self.filter_wumpus_possible_locations()
                self.possible_wumpus_locations = self.filter_out_of_bound_locations(self.possible_wumpus_locations)
                self.check_if_wumpus_location_is_determined_by_stench()
                self.check_if_wumpus_location_is_determined_by_size()

                if self.adjacent_cell:
                    if self.adjacent_cell in self.possible_wumpus_locations:
                        self.possible_wumpus_locations.remove(self.adjacent_cell)
                        self.wumpus_location = self.possible_wumpus_locations.pop()
        else:
            # update no_stench_locations (add current agent location)
            self.no_stench_locations.add((self.agent_x_coord, self.agent_y_coord))
            # update no_wumpus_locations (add current neqigbours)
            self.no_wumpus_locations = self.no_wumpus_locations.union(not_visited_neighbour_locations)
            # update possible_wumpus_locations
            if not self.wumpus_location_has_determined:
                self.possible_wumpus_locations = self.possible_wumpus_locations.difference(not_visited_neighbour_locations)
                self.possible_wumpus_locations = self.possible_wumpus_locations.difference({(self.agent_x_coord, self.agent_y_coord)})
                self.filter_wumpus_possible_locations()
                self.possible_wumpus_locations = self.filter_out_of_bound_locations(self.possible_wumpus_locations)
                self.check_if_wumpus_location_is_determined_by_stench()
                self.check_if_wumpus_location_is_determined_by_size()

    def check_if_wumpus_location_is_determined_by_size(self):
        """
        Determine Wumpus location if there is only one possible location for Wumpus,
        taken in account that there is only one Wumpus.
        """
        if len(self.possible_wumpus_locations) == 1:
            self.wumpus_location_has_determined = True
            self.wumpus_location = self.possible_wumpus_locations.pop()
            self.no_wumpus_locations = self.no_wumpus_locations.union(self.possible_wumpus_locations.difference(set([self.wumpus_location])))
            self.possible_wumpus_locations = set()

    def check_if_wumpus_location_is_determined_by_stench(self):
        """
        If know that stench locations are at a distance of 2 along a single axis,
        then Wumpus location is determined as location beetween these two points.
        """
        if len(self.stench_locations) > 1:
            for (x1, y1) in self.stench_locations:
                for (x2, y2) in self.stench_locations:
                    if (abs(x2 - x1) == 2 and y2 - y1 == 0):
                        self.wumpus_location = ((x1 + x2) // 2, y1)
                    if (x2 - x1 == 0 and abs(y2 - y1) == 2):
                        self.wumpus_location = (x1, (y1 + y2) // 2)

            if self.wumpus_location:
                self.wumpus_location_has_determined = True
                self.no_wumpus_locations = (self.no_wumpus_locations
                    .union(self.possible_wumpus_locations.difference(set([self.wumpus_location]))))
                self.possible_wumpus_locations = set()

    def filter_wumpus_possible_locations(self):
        """
        If 2 stench locations are in diagonally adjusent locations, then 
        the possible locations for the Wumpus are limited to just two possibilities.
        """
        if len(self.stench_locations) > 1:
            conditional_possible_wumpus_locations = set()

            for (x1, y1) in self.stench_locations:
                for (x2, y2) in self.stench_locations:
                    if abs(x2 - x1) == 1 and abs(y2 - y1) == 1:
                        conditional_possible_wumpus_locations.add((x1, y2)) 
                        conditional_possible_wumpus_locations.add((x2, y1))

            if conditional_possible_wumpus_locations:
                self.possible_wumpus_locations = self.possible_wumpus_locations.intersection(conditional_possible_wumpus_locations)

    def get_safe_locations(self):
        """
        Returns the full list of safe locations.
        """
        if not self.is_wumpus_alive:
            return self.no_pit_locations
        else:
            if self.possible_wumpus_locations:
                return self.no_pit_locations.difference(self.possible_wumpus_locations)
            else:
                return self.no_pit_locations.intersection(self.no_wumpus_locations)

    def make_bayesian_proposition(self, name, values, probabilities):
        """
        Helper function to define a conditional probability distribution table.
        """
        return TabularCPD(variable=name,
                          variable_card=len(values), 
                          values=[[probability] for probability in probabilities], 
                          state_names={name: values})
    
    def make_evidence_bool_bayesian_proposition(self, name, values, evidence, bool_function):
        """
        Helper function to make an evidence conditional probability distribution table.
        """
        def cpd_values(bool_fn, arity):
            def bool_to_prob(*args) -> bool:
                return (1.0, 0.0) if bool_fn(*args) else (0.0, 1.0)
            return tuple(zip(*[bool_to_prob(*ps) for ps in itertools.product((True, False), repeat=arity)]))

        return TabularCPD(variable=name, 
                          variable_card=2,
                          values=cpd_values(eval(bool_function), len(evidence)),
                          evidence=evidence, 
                          evidence_card=[2] * len(evidence),
                          state_names={n: values for n in [name] + evidence})

    def make_bayesian_bool_propositions_dict(self, locations, name_prefix, probabilities):
        """
        Generates the bayesian bool propositions dict.
        """
        cpds = {}
        for loc in locations:
            cpds[loc[0], loc[1]] = \
                self.make_bayesian_proposition(
                name=f'{name_prefix}{loc[0]}{loc[1]}', 
                values=[True, False], 
                probabilities=probabilities
            )
        return cpds
    
    def get_bayesian_model(self, pit_probability=0.2):
        """
        Make a bayesian model.
        """
        pit_cpds_dict = {}

        # add CPDs of agent_legal_locations without no_pit_locations 
        # (this locations a have pit probability 0.2)
        pit_cpds_dict.update(self.make_bayesian_bool_propositions_dict(
            locations=self.agent_legal_locations.difference(self.no_pit_locations), 
            name_prefix='P',
            probabilities=[pit_probability, 1 - pit_probability])
        )
        # add CPDs of no_pit_locations
        # (this locations have a pit probability 0.0)
        pit_cpds_dict.update(self.make_bayesian_bool_propositions_dict(
            locations=self.no_pit_locations,
            name_prefix='P',
            probabilities=[0.0, 1.0])
        )

        # for k,v in pit_cpds_dict.items():
        #     print(k,v)

        breeze_cpds_dict = {}
        for breeze_location in self.breeze_locations.union(self.no_breeze_locations):
            evidence = [f'P{inner_loc[0]}{inner_loc[1]}' for inner_loc in 
                        self.filter_out_of_bound_locations(locations=self.get_adjacent_locations(x=breeze_location[0], y=breeze_location[1]))]
            breeze_cpds_dict[(breeze_location[0], breeze_location[1])] = \
                self.make_evidence_bool_bayesian_proposition(
                    name=f"B{breeze_location[0]}{breeze_location[1]}", 
                    values=[True, False], 
                    evidence=evidence, 
                    bool_function=f'lambda {", ".join(evidence)}: ({" or ".join(evidence)})'
                )
        
        # for k,v in breeze_cpds_dict.items():
        #     print(k,v)

        model = BayesianModel()
        for breeze_cpd in breeze_cpds_dict.values():
            for evidence in breeze_cpd.get_evidence():
                model.add_edge(evidence, breeze_cpd.variable)

        model.add_nodes_from([pit_cpd.variable for pit_cpd in pit_cpds_dict.values()])
        model.add_cpds(*breeze_cpds_dict.values())
        model.add_cpds(*pit_cpds_dict.values())

        return model
    
    def get_evidence_dict(self):
        """
        Generates the evidence dict.
        """
        evidence_dict = {}

        # add pit (P) evidence
        for loc in self.no_pit_locations:
            evidence_dict[f'P{loc[0]}{loc[1]}'] = False
        
        # add breeze (B) evidence
        for loc in self.breeze_locations:
            evidence_dict[f'B{loc[0]}{loc[1]}'] = True
        for loc in self.no_breeze_locations:
            evidence_dict[f'B{loc[0]}{loc[1]}'] = False

        return evidence_dict

    def check_pit_probability(self, location, bayesian_model, evidence_dict):
        """
        Returns the probability of being a pit for the input location.
        """
        inference = VariableElimination(bayesian_model)
        return inference.query(variables=[f'P{location[0]}{location[1]}'], 
                               evidence=evidence_dict, 
                               show_progress=False).values[0]
