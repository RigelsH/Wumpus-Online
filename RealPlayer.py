import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Iterable
import networkx as nx

import wumpus as wws

from state_knowledge import Knowledge
from utils import navigate, find_lowest_indexes
from heuristics import heuristic_minmax


class AstarPlayer(wws.OnlinePlayer):
    """
    This player automatically makes the next move.
    """
    def __init__(self, name):
        super().__init__(name=name)
        self.knowledge = Knowledge()
        self.previous_agent_action = None
        self.climb_command = False

    def play(self, percept: wws.Percept, actions: Iterable[wws.Agent.Actions], reward: int) -> wws.Agent.Actions:
        
        # update knowledge about environment ans hunter
        self.knowledge.update_after_perception(previous_agent_action=self.previous_agent_action, current_percept=percept)
        
        if percept.glitter:
            # if hunter feels GLITTER we will GRAB gold
            self.previous_agent_action = wws.Hunter.Actions.GRAB
            return wws.Hunter.Actions.GRAB
        
        if percept.scream:
            # if hunter hears scream we will MOVE, meaning wumpus is dead
            # and the location in from is safe
            self.previous_agent_action = wws.Hunter.Actions.MOVE
            return wws.Hunter.Actions.MOVE

        bayesian_model = self.knowledge.get_bayesian_model()
        evidence_dict = self.knowledge.get_evidence_dict()
        safe_locations = self.knowledge.get_safe_locations()

        # initialize agent legal (unseen) locations with zero pit probabilities 
        agent_legal_locations_pit_prob = {loc: 0.0 for loc in self.knowledge.agent_legal_locations}

        # and update pit probabilities using bayes net
        # (make difference because model and evidence will have the same nodes)
        for loc in self.knowledge.agent_legal_locations.difference(self.knowledge.no_pit_locations):
            agent_legal_locations_pit_prob[loc] = \
                self.knowledge.check_pit_probability(
                    location=loc, 
                    bayesian_model=bayesian_model, 
                    evidence_dict=evidence_dict
                )

        # filter agent legal (unseen) locations by threshold probability
        agent_legal_locations_pit_prob_filtered = set([key for key, value in agent_legal_locations_pit_prob.items() if value < 0.2])
        agent_legal_locations_pit_prob_filtered = agent_legal_locations_pit_prob_filtered.difference(self.knowledge.possible_wumpus_locations)
        
        # forbid bumping forever
        if self.knowledge.world_size_has_determined:
            agent_legal_locations_pit_prob_filtered = [
                (i,j) for (i,j) in agent_legal_locations_pit_prob_filtered
                if  i >= 0 and j >= 0 and i < self.knowledge.world_size and j < self.knowledge.world_size]

        if self.knowledge.wumpus_location:
            agent_legal_locations_pit_prob_filtered.remove(self.knowledge.wumpus_location)

        agent_legal_locations_pit_prob_filtered = list(agent_legal_locations_pit_prob_filtered)
        
        # for the graph we make union of safe locations and
        # agent_legal_locations_pit_prob_filtered (with difference of possible wumpus locations)
        graph_locations = set(agent_legal_locations_pit_prob_filtered).union(safe_locations)

        G = nx.Graph()
        G.add_nodes_from(nodes_for_adding=graph_locations)

        # Connect adjacent nodes with edges
        for i in range(len(list(G.nodes))):
            for j in range(i+1, len(list(G.nodes))):
                node1 = list(G.nodes)[i]
                node2 = list(G.nodes)[j]
                if abs(node1[0] - node2[0]) + abs(node1[1] - node2[1]) == 1:
                    G.add_edge(node1, node2)

        if self.knowledge.gold_has_grabbed:
            # if we grabbed a gold

            if (self.knowledge.agent_x_coord, self.knowledge.agent_y_coord) == (0, 0):
                # if hunter reached starting location, we will CLIMB
                self.previous_agent_action = wws.Hunter.Actions.CLIMB
                self.climb_command = True
                return wws.Hunter.Actions.CLIMB
            
            else:
                # we need to find path back
                shortest_path = nx.astar_path(G, 
                                              source=(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord),
                                              target=(0, 0),
                                              heuristic=heuristic_minmax)
                actions = navigate(path=shortest_path, orientation=self.knowledge.agent_orientation)
                self.previous_agent_action = actions
                return actions

        else:
            # if we didnt grab gold, we will explore environment

            closest_target = None

            if len(agent_legal_locations_pit_prob_filtered) > 0:
                # if there is at least one safe location to explore

                if len(agent_legal_locations_pit_prob_filtered) == 1:
                    closest_target = agent_legal_locations_pit_prob_filtered[0]
                
                else:
                    # if cell in front of agent is available, we keep going ahead
                    if (self.knowledge.agent_x_coord + self.knowledge.agent_orientation[0], 
                        self.knowledge.agent_y_coord + self.knowledge.agent_orientation[1]) in agent_legal_locations_pit_prob_filtered:
                        closest_target = (self.knowledge.agent_x_coord + self.knowledge.agent_orientation[0], 
                                          self.knowledge.agent_y_coord + self.knowledge.agent_orientation[1])
                    else:
                        routes_lengths = []
                        for possible_target_loc in agent_legal_locations_pit_prob_filtered:
                            trip_length = nx.shortest_path_length(
                                G, 
                                source=(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord), 
                                target=possible_target_loc
                            )
                            routes_lengths.append(trip_length)

                        closest_targets_ids = find_lowest_indexes(routes_lengths)
                        closest_targets = [list(agent_legal_locations_pit_prob_filtered)[i] for i in closest_targets_ids]

                        if len(closest_targets) > 0:
                            if len(closest_targets) > 1:
                                xymin = 1e6
                                closest_target = closest_targets[0]
                                for num, (x, y) in enumerate(closest_targets):
                                    if min(x, y) < xymin:
                                        xymin = min(xymin, x**2 + y**2)
                                        closest_target = closest_targets[num]
                                    
                            else:
                                closest_target = closest_targets[0]
                        else:
                            closest_target = (0, 0)

                self.climb_command = False
            
            else:
                # there is NO safe locatioon

                # first, we look if there are wumpus locations available 
                # (possible several locations or one determined location)
                # in this a case we go to wumpus!
                if self.knowledge.is_wumpus_alive:
                    if self.knowledge.wumpus_location:
                        # in case if we know exactly its location

                        euclidean_dist_to_wumpus = \
                            (self.knowledge.wumpus_location[0] - self.knowledge.agent_x_coord)**2 + \
                            (self.knowledge.wumpus_location[1] - self.knowledge.agent_y_coord)**2
                        
                        if euclidean_dist_to_wumpus == 1:
                            # if we are close, check agent orientation w.r.t. wumpus location
                            # (we need to be in from to shoot and kill wumpus)
                            
                            if (self.knowledge.agent_x_coord + self.knowledge.agent_orientation[0], 
                                self.knowledge.agent_y_coord + self.knowledge.agent_orientation[1]) == self.knowledge.wumpus_location:
                                # if we are in front of wumpus, we will SHOOT
                                self.previous_agent_action = wws.Hunter.Actions.SHOOT
                                return wws.Hunter.Actions.SHOOT

                            else:
                                # otherwise we will turn to be in front of wumpus
                                path2wumpus = [(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord), self.knowledge.wumpus_location]
                                actions = navigate(path=path2wumpus, orientation=self.knowledge.agent_orientation)
                                self.previous_agent_action = actions
                                return actions

                        else:
                            closest_target = self.knowledge.wumpus_location

                    elif len(self.knowledge.possible_wumpus_locations) > 0:
                        # if we are not sure about his location
                        # and if we feel STENCH, we gonna SHOOT
                        if percept.stench:
                            self.previous_agent_action = wws.Hunter.Actions.SHOOT
                            return wws.Hunter.Actions.SHOOT
                    else:
                        # wumpus locations are not explored (not reachable)
                        pass

                closest_target = (0, 0)

                if (self.knowledge.agent_x_coord, self.knowledge.agent_y_coord) == (0, 0):
                    # we CLIMB without gold
                    self.previous_agent_action = wws.Hunter.Actions.CLIMB
                    self.climb_command = True
                    return wws.Hunter.Actions.CLIMB

            if not self.climb_command:
                shortest_path = nx.astar_path(
                    G, 
                    source=(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord),
                    target=closest_target,
                    heuristic=heuristic_minmax
                )
                actions = navigate(path=shortest_path, orientation=self.knowledge.agent_orientation)
                self.previous_agent_action = actions
                return actions