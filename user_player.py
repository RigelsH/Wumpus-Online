import argparse
import random
random.seed(10)
import json
import sys
from typing import Iterable
import networkx as nx

import wumpus as wws

from state_knowledge import Knowledge
from utils import navigate, find_lowest_indexes
from heuristics import heuristic_minmax


class UserPlayer(wws.OnlinePlayer):
    """This player asks the user for the next move, if it's not ambiguous it accepts also commands initials and ignores the case."""

    def __init__(self, name):
        super().__init__(name=name)
        self.knowledge = Knowledge()
        self.previous_agent_action = None
        self.climb_command = False

    def play(self, percept: wws.Percept, actions: Iterable[wws.Agent.Actions], reward: int) -> wws.Agent.Actions:
        
        # UPDATE KNOWLEDGE
        self.knowledge.update_after_perception(previous_agent_action=self.previous_agent_action, current_percept=percept)
        
        if percept.glitter:
            print('GO GRAB AFTER GLITTER')

        if percept.scream:
            print('MOVE AFTER SCREAM')
            
        actions_dict = {a.name: a for a in actions}
        bayesian_model = self.knowledge.get_bayesian_model()
        evidence_dict = self.knowledge.get_evidence_dict()
        safe_locations = self.knowledge.get_safe_locations()

        print(f'percept: {self.name} {str(percept)}')
        print(self.knowledge.__str__())
        # print("bayesian_model.modes:", bayesian_model.nodes())
        # print("evidence_dict", evidence_dict)
        print("previous_agent_action", self.previous_agent_action)
        print("safe_locations", safe_locations)

        # initialize agent legal (unseen) locations with zero pit probabilities 
        agent_legal_locations_pit_prob = {loc: 0.0 for loc in self.knowledge.agent_legal_locations}

        # and update pit probabilities using bayes net
        # (make idfference because model and evidence will have the same nodes)
        for loc in self.knowledge.agent_legal_locations.difference(self.knowledge.no_pit_locations):
            agent_legal_locations_pit_prob[loc] = \
                self.knowledge.check_pit_probability(
                    location=loc, 
                    bayesian_model=bayesian_model, 
                    evidence_dict=evidence_dict
                )
        print("agent_legal_locations_pit_prob", agent_legal_locations_pit_prob)

        # filter agent legal (unseen) locations by threshold probability
        agent_legal_locations_pit_prob_filtered = set([key for key, value in agent_legal_locations_pit_prob.items() if value < 0.2])
        agent_legal_locations_pit_prob_filtered = agent_legal_locations_pit_prob_filtered.difference(self.knowledge.possible_wumpus_locations)
        
        # forbid bumping forever
        # TODO: why can it be out of bounds?
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
        print('graph_locations', graph_locations)

        G = nx.Graph()
        G.add_nodes_from(nodes_for_adding=graph_locations)

        # Connect adjacent nodes with edges
        for i in range(len(list(G.nodes))):
            for j in range(i+1, len(list(G.nodes))):
                node1 = list(G.nodes)[i]
                node2 = list(G.nodes)[j]
                if abs(node1[0] - node2[0]) + abs(node1[1] - node2[1]) == 1:
                    G.add_edge(node1, node2)

        # print(G)
        # print("graph NODES", G.nodes)
        # print(nx.to_dict_of_dicts(G))

        if self.knowledge.gold_has_grabbed:
            # if we grabbed a gold

            if (self.knowledge.agent_x_coord, self.knowledge.agent_y_coord) == (0, 0):
                # if we reached starting location
                print("GO CLIBM")
                # return wws.Hunter.Actions.CLIMB
            else:
                # we need to find path back
                shortest_path = nx.astar_path(G, 
                                              source=(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord),
                                              target=(0, 0),
                                              heuristic=heuristic_minmax)
                actions = navigate(path=shortest_path, orientation=self.knowledge.agent_orientation)
                print('shortest_path', shortest_path)
                print('actions back', actions)
                #return shortest_path[0]

        else:
            # if we didnt grab gold, we will explore environment

            closest_target = None

            if len(agent_legal_locations_pit_prob_filtered) > 0:
                # if there is at least one safe location to explore

                if len(agent_legal_locations_pit_prob_filtered) == 1:
                    print('OPTION 1', agent_legal_locations_pit_prob_filtered)
                    closest_target = agent_legal_locations_pit_prob_filtered[0]
                
                else:
                    # if cell in front of agent is available, we keep going ahead
                    if (self.knowledge.agent_x_coord + self.knowledge.agent_orientation[0], 
                        self.knowledge.agent_y_coord + self.knowledge.agent_orientation[1]) in agent_legal_locations_pit_prob_filtered:
                        print('OPTION 2', agent_legal_locations_pit_prob_filtered, closest_target)
                        closest_target = (self.knowledge.agent_x_coord + self.knowledge.agent_orientation[0], 
                                          self.knowledge.agent_y_coord + self.knowledge.agent_orientation[1])
                    else:
                        print('OPTION 345', agent_legal_locations_pit_prob_filtered)
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
                                # closest_target = random.choice(closest_targets)
                                print('OPTION 3', agent_legal_locations_pit_prob_filtered)
                                xymin = 999
                                closest_target = closest_targets[0]
                                for num, (x, y) in enumerate(closest_targets):
                                    if min(x, y) < xymin:
                                        xymin = min(x, y)
                                        # TODO: try x**2 + y**2
                                        closest_target = closest_targets[num]
                                    
                            else:
                                print('OPTION 4', agent_legal_locations_pit_prob_filtered)
                                closest_target = closest_targets[0]
                        else:
                            print('OPTION 5', agent_legal_locations_pit_prob_filtered)
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
                                # if we are in front of wumpus, we will shoot!
                                print('SHOOOT IT')

                            else:
                                # otherwise we will turn to be in front of wumpus
                                path2wumpus = [(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord), self.knowledge.wumpus_location]
                                actions = navigate(path=path2wumpus, orientation=self.knowledge.agent_orientation)
                                print(actions)
                        else:
                            closest_target = self.knowledge.wumpus_location

                    elif len(self.knowledge.possible_wumpus_locations) > 0:
                        # if we are not sure about his location
                        # and if we feel stench, we gonna shoot!
                        if percept.stench:
                            print('SHOOT AFTER SMELL')
                    else:
                        # wumpus locations are not explored (not reachable)
                        pass

                print('OPTION 6', agent_legal_locations_pit_prob_filtered)

                # if self.knowledge.is_wumpus_alive and (len(self.knowledge.possible_wumpus_locations) > 0 or self.knowledge.wumpus_location):
                #     # if wumpus alive and (reachable or determined) -> go shoot him!
                #     # TODO: write WUMPUS SHOOTING logic
                #     pass

                closest_target = (0, 0)

                if (self.knowledge.agent_x_coord, self.knowledge.agent_y_coord) == (0, 0):
                    print('CLIMB WITHOUT GOLD !!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print("===============================================")
                    
                    self.climb_command = True

            if not self.climb_command:
                shortest_path = nx.astar_path(
                    G, 
                    source=(self.knowledge.agent_x_coord, self.knowledge.agent_y_coord),
                    target=closest_target,
                    heuristic=heuristic_minmax
                )
                actions = navigate(path=shortest_path, orientation=self.knowledge.agent_orientation)
                print('shortest_path', shortest_path)
                print('actions forward', actions)


        while True:
            answer = input('{}: select an action {} and press enter, or empty to stop: '.format(
                self.name, list(actions_dict.keys()))).strip()
            if len(answer) < 1:
                return None
            elif answer in actions_dict:
                self.previous_agent_action = actions_dict[answer]
                return actions_dict[answer]
            else:
                print('Cannot understand <{}>'.format(answer))


def play_fixed_informed():
    """Play on a given world described in JSON format."""
    
    with open("data/map01.json") as fd:
        world_list_dict = json.load(fd)
    world = wws.WumpusWorld.from_JSON(world_list_dict)
    wws.run_episode(world=world, player=UserPlayer(name="Artem")) 


EXAMPLES=(play_fixed_informed, play_fixed_informed)


def main(*cargs):
    """Demonstrate the use of the wumpus API on selected worlds"""
    ex_names = {ex.__name__.lower(): ex for ex in EXAMPLES}
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('example', nargs='?', help='select one of the available example', choices=list(ex_names.keys()))
    args = parser.parse_args(cargs)
    
    if args.example:
        ex = ex_names[args.example.lower()]
    else:
        ex = random.choice(EXAMPLES)

    print('Example {}:'.format(ex.__name__))
    print('  ' + ex.__doc__)
    ex()

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
