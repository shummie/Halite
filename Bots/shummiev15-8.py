# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:24:03 2016

@author: Shummie
"""

###########
# Imports #
###########
import math
import itertools
import sys
import logging
import numpy
import random
import time
import scipy.sparse



#############
# Variables #
#############

botname = "shummie v15.8"

strength_buffer = 1
buildup_multiplier = 7

early_game_value_threshold = 0.75

future_map_value = 3
owned_production_value = -1
enemy_production_value = 4
neutral_production_value = 3

combat_zone_value = 6
enemy_border_value = 4

is_owned_value = -2


        
#==============================================================================
# Game Class
#==============================================================================

class Game:
    def __init__(self):
        self.initialize_game()
        
    def initialize_game(self):
        # This should only be called once and at the beginning of the game
        self.my_id = int(get_string())
        map_size_string = get_string()
        production_map_string = get_string()
        
        self.width, self.height = tuple(map(int, map_size_string.split()))
        self.starting_player_count = None # Will automatically be initialized in the first GameFrame __init__ call.                   
        
        self.phase = 0
        self.frame = 0
        
        # Distance maps are expensive to create and are static.
        self.distance_map = self.create_distance_map()
        self.distance_map_no_decay = self.create_distance_map(1)
        
        self.production_map = numpy.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()
        
        self.game_frame = [GameFrame(self)]
                           
        send_string(botname)
                           
    def update_next_frame(self):
        # updates the game to the next frame
        self.frame += 1
        self.game_frame.append(GameFrame(self))
        self.game_frame[game.frame].update_maps()
        
        if game.phase == 0:
            if numpy.sum(self.game_frame[self.frame].is_owned_map) > (10*(self.width * self.height)**.5) / ((self.starting_player_count**0.5) * 10):
                game.phase = 1
        
        # Run the AI functions
        # Assign border cell priorities

    def create_distance_map(self, falloff = 1):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.
        
        # Create the base map for 0, 0
        zero_zero_map = numpy.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                dist_x = min(x, -x % self.width)
                dist_y = min(y, -y % self.width)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        zero_zero_map = zero_zero_map ** falloff
        
        distance_map = numpy.zeros((self.width, self.height, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)
                
        return distance_map       
        
    def get_moves(self):
        self.game_frame[self.frame].get_moves()
        

   
#==============================================================================
# GameFrame Class
#==============================================================================

class GameFrame():
    # Stores all information about a single frame of the game so that it can be historically referenced if desired
    def __init__(self, game, map_string = None):
        self.game = game
        self.production_map = game.production_map
        self.frame = game.frame
        self.width = game.width
        self.height = game.height
        self.my_id = game.my_id
        
        
        self.get_frame(map_string)
        
        if self.game.starting_player_count == None:
            self.game.starting_player_count = numpy.amax(self.owner_map)

    def __iter__(self):
        # Allows direct iteration over all squares
        return itertools.chain.from_iterable(self.squares)
        
    def get_frame(self, map_string = None):
        # Reads the input information. This should only be called once.
        
        if map_string is None:
            map_string = get_string()
        split_string = map_string.split()

        # The state of the map (including owner and strength values, but excluding production values) is sent in the following way:
        # One integer, COUNTER, representing the number of tiles with the same owner consecutively.
        # One integer, OWNER, representing the owner of the tiles COUNTER encodes.
        # The above repeats until the COUNTER total is equal to the area of the map. 
        # It fills in the map from row 1 to row HEIGHT and within a row from column 1 to column WIDTH. 
        # Please be aware that the top row is the first row, as Halite uses screen-type coordinates.
        owners = list()
        
        while len(owners) < self.width * self.height:
            counter = int(split_string.pop(0))
            owner = int(split_string.pop(0))
            owners.extend([owner] * counter)
        assert len(owners) == self.width * self.height        
        self.owner_map = numpy.array(owners).reshape((self.height, self.width)).transpose()
                
        assert len(split_string) == self.width * self.height
        str_list = list(map(int, split_string))        
        self.strength_map = numpy.array(str_list).reshape((self.height, self.width)).transpose()
        
        self.squares = numpy.empty((self.width, self.height), dtype = numpy.object)
        
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y] = Square(self.game, x, y, self.owner_map[x, y], self.strength_map[x, y], self.production_map[x, y])
        
        # Reset the move_map
        self.move_map = numpy.ones((self.width, self.height)) * -1  # Could possibly expand this in the future to consider enemy moves...         

    def update_maps(self):
        # Updates all maps for this frame.
        
        # Create informational maps
        self.update_owner_maps()
        self.update_distance_maps()
        self.update_border_maps()
        
        self.update_recover_map()
        
        self.update_heuristic_map()
        
    def update_owner_maps(self):
        # Creates a 3-d owner map from self.owner_map
        # self.is_owner_map[# of players, width, height]
        # self.is_owner_map[player_id] returns a width x height matrix of 0's or 1's depending on whether or not that player owns that square
        # self.is_owner_map[0] returns 1 for all Neutral squares, 0 otherwise
        # self.is_owner_map[game_map.my_id] returns 1 for all player owned squares, 0 otherwise
        self.is_owner_map = numpy.zeros((self.game.starting_player_count + 1, self.width, self.height))
        # I can probably speed this up by doing something like self.is_owner_map[x] = (self.owner_map == x) or something like that but w/e.
        for x in range(self.width):
            for y in range(self.height):
                self.is_owner_map[self.owner_map[x, y], x, y] = 1      
        
        self.is_enemy_map = numpy.zeros((self.width, self.height), dtype = bool)
        self.is_neutral_map = numpy.zeros((self.width, self.height), dtype = bool)
        self.is_owned_map = numpy.zeros((self.width, self.height), dtype = bool)
        
        self.is_neutral_map[numpy.where(self.owner_map == 0)] = True
        self.is_owned_map[numpy.where(self.owner_map == self.my_id)] = True
        self.is_enemy_map[numpy.where((self.owner_map != 0) * (self.owner_map != self.my_id))] = True
        

    def update_border_maps(self):
        # The border maps are squares that are NEXT to the territory. 
        # self.border_map[x, y] = 1 if the square is NEXT to OUR territory but is NOT our territory (NEUTRAL)
        # self.enemy_border_map[x, y] = 1 if the square is NEXT to an ENEMY territory but is NEUTRAL territory      
        self.border_map = self.distance_from_owned == 1
        self.enemy_border_map = self.distance_from_enemy == 1
    
        self.inner_border_map = spread_n(self.border_map * 1.0, 1)
        self.inner_border_map = numpy.multiply(self.inner_border_map, self.is_owned_map) # Do we need this anymore?
        
        self.combat_zone_map = numpy.multiply(self.border_map, numpy.multiply(self.is_neutral_map, self.strength_map == 0))
        
        
    def update_distance_maps(self):
        self.distance_from_owned = distance_from_owned(self.game.distance_map_no_decay, self.is_owned_map)
        self.distance_from_owned[numpy.nonzero(self.is_owned_map)] = 0 # Any territory we own has a distance of 0.
        
        self.distance_from_border = distance_from_owned(self.game.distance_map_no_decay, 1 - self.is_owned_map)
        self.distance_from_border[numpy.nonzero(1 - self.is_owned_map)] = 0
        
        self.distance_from_enemy = distance_from_owned(self.game.distance_map_no_decay, self.is_enemy_map)                          
        self.distance_from_enemy[numpy.nonzero(self.is_enemy_map)] = 0
        
        
    def update_recover_map(self):
        max_distance = 50
        self.recover_map = numpy.zeros((max_distance + 1, self.width, self.height))
        self.recover_map[0] = numpy.divide(self.strength_map, numpy.maximum(self.production_map, 0.01))
        self.recover_map[0] = numpy.multiply(self.recover_map[0], self.is_neutral_map)
        self.recover_map[0] += (self.is_owned_map + self.is_enemy_map) * 999
        
        for distance in range(1, max_distance + 1):
            dir_map = numpy.zeros((4, self.width, self.height))
            dir_map[0] = roll_xy(self.recover_map[distance - 1], 0, 1)
            dir_map[1] = roll_xy(self.recover_map[distance - 1], 0, -1)
            dir_map[2] = roll_xy(self.recover_map[distance - 1], 1, 0)
            dir_map[3] = roll_xy(self.recover_map[distance - 1], -1, 0)
            
            self.recover_map[distance] = numpy.add(self.recover_map[distance - 1], numpy.amin(dir_map, 0))
        
        for d in range(2, max_distance + 1):
            self.recover_map[d] = self.recover_map[d] / d
        
    def get_moves(self):
        self.assign_border_cell_priority()
        self.attack_production_squares()
        self.attack_combat_squares()
        self.reinforce_borders()
        
  
    def rebase_map(self, map_a):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.width * self.height
        factor = (self.width * self.height) / numpy.sum(map_a)
        return numpy.multiply(map_a, factor)

    def send_frame(self):
        # Goes through each square and get the list of moves.
        move_list = []
        for sq in itertools.chain.from_iterable(self.squares):
            if sq.owner == self.my_id:
                if sq.move == -1:
                    # In the event we didn't actually assign a move, make sure it's coded to STILL
                    sq.move = 4
                move_list.append(sq)
        
        send_string(' '.join(str(square.x) + ' ' + str(square.y) + ' ' + str(translate_cardinal(square.move)) for square in move_list))

    def assign_border_cell_priority(self):
        # This function assigns each border cell to 1 of three categories:
        # 1) Production focused
        # 2) Combat focused
        # 3) High(enough) valued targets
        # 4) Worthless
        # Get a list of all border cells
        border_square_list = [s for s in self if self.border_map[s.x, s.y]]
        self.production_border_square_list = []
        self.combat_border_square_list = []
        self.other_border_square_list = []
        self.worthless_border_square_list = []

        for border in border_square_list:
            # Here is where we determine which list to place each square into.
            # Production squares fit the following criteria:
            # Must be at least 4 squares away from enemy controlled territory
            if self.distance_from_enemy[border.x, border.y] > 4:
                self.production_border_square_list.append(border)
            else:
            #elif self.distance_from_enemy[border.x, border.y] <= 4:
                self.combat_border_square_list.append(border)
            #else:
            #    self.other_border_square_list.append(border)
                
        # We have a tentative list of what squares should be focused on now.
    
    def attack_production_squares(self):
        if len(self.production_border_square_list) == 0:
            return False
        # Attempts to attack all squares in the production square list. Squares which are successful are removed, cells that aren't get moved to the other_border_square_list.
        def get_recover_value(square):
            #return self.recover_map[self.distance_from_enemy[square.x, square.y] - 1, square.x, square.y]
            return self.recover_map[5, square.x, square.y]
        self.production_border_square_list.sort(key = lambda b: get_recover_value(b))
        
        threshold = get_recover_value(self.production_border_square_list[0]) / early_game_value_threshold
        
        for border in self.production_border_square_list:
            find_cell = False
            if get_recover_value(border) <= threshold:
                find_cell = self.attack_cell(border, 5)
                if find_cell:
                    self.production_border_square_list.remove(border)
    
    def attack_combat_squares(self):
        # Simple combat routine for now. Just try to attack any neighbors that are in a combat zone!
        if self.game.phase == -1:
            if len(self.combat_border_square_list) == 0:
                return False
                
            for border in self.combat_border_square_list:
                find_cell = self.attack_cell(border, 3)
                if find_cell:
                    self.combat_border_square_list.remove(border)
        else:
            if len(self.combat_border_square_list) > 0:
                for border in self.combat_border_square_list:
                    find_cell = self.attack_cell(border, 1)
                    if find_cell:
                        self.combat_border_square_list.remove(border)            
            if len(self.production_border_square_list) > 0:
                for border in self.production_border_square_list:
                    find_cell = self.attack_cell(border, 3)
                    if find_cell:
                        self.production_border_square_list.remove(border)            
                
    def reinforce_borders(self):
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.game.my_id and square.move == -1 and not self.inner_border_map[square.x, square.y]:
                if square.strength > square.production * buildup_multiplier:
                    self.find_nearest_non_owned_border(square)
                    
    def reinforce_borders2(self):
        #if len(self.other_border_square_list) == 0:
        #    return False
        
        for square in itertools.chain.from_iterable(self.squares):
            if square.move == -1 and not self.inner_border_map[square.x, square.y]:
                if square.strength > square.production * buildup_multiplier:
                    target_val = 0
                    target = self.get_target(square, NORTH)
                    for n in square.neighbors():
                        if self.heuristic_map[n.x, n.y] > target_val:
                            target_val = self.heuristic_map[n.x, n.y]
                            target = n
                    square.move_to_target(target, True)
                        

    def find_nearest_non_owned_border(self, square):
                
        current_distance = self.distance_from_border[square.x, square.y]
        for n in square.neighbors():
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_border[n.x, n.y] < current_distance:
                    success = square.move_to_target(n, True)
                    if success:
                        break
                    
            
                    

    def attack_cell(self, target, max_cells_out = 1):
        # Will only attack the cell if sufficient strength
        # Otherwise, will attempt to move cells by cells_out so that it can gather enough strength.
        # Returns True if we have successfully found something to attack this
        # Returns False otherwise.
        if target.owner == self.my_id:
            # This function shouldn't try to attack our own cell
            return False
        
        cells_out = 1
        while cells_out <= max_cells_out:
            if cells_out > 1 and target.owner != 0:
                # Don't try to coordinate multi-turn attacks into a combat zone
                return False
            
            available_squares = (self.move_map == -1) * 1
            distance_matrix = self.friendly_flood_fill(target, cells_out)
            distance_matrix[distance_matrix == -1] = 0

            available_strength = numpy.sum(numpy.multiply(numpy.multiply(self.strength_map, numpy.minimum(distance_matrix, 1)), available_squares))
            
            # If all available cells stay STILL then do we have enough strength?
            distance_matrix_prod = cells_out - distance_matrix
            distance_matrix_prod[distance_matrix_prod == cells_out] = 0
            available_production = numpy.sum(numpy.multiply(numpy.multiply(self.production_map, distance_matrix_prod), available_squares))
            
            if available_strength + available_production > target.strength:
                # We have sufficient strength. Let's attack!
                # Get a list of all friendly neighbors
                attacking_cells = [x for x in target.neighbors(cells_out) if x.owner == self.my_id and x.move == -1]
                still_cells = []
                if cells_out > 1:
                    # We need to tell cells to stay still.
                    still_cells = [x for x in target.neighbors(cells_out - 1) if x.owner == self.my_id and x.move == -1]
                moving_cells = list(set(attacking_cells) - set(still_cells))
                
                # Ok, since we are doing this iteratively, we know that all cells in still_cellls must stay still, else an earlier cells_out would have worked
                for square in still_cells:
                    self.make_move(square, STILL)
                
                still_strength = numpy.sum(numpy.multiply(numpy.multiply(self.strength_map, numpy.minimum(distance_matrix_prod, 1)), available_squares))
                needed_strength_from_movers = target.strength - available_production - still_strength
                
                if needed_strength_from_movers > 0:
                    # We don't necessarily want the highest strength piece to capture this. but if we start with the smallest, we might be wasting moves / production.
                    # See if we need more than 1 piece to capture.
                    moving_cells.sort(key = lambda x: x.strength, reverse = True)
                    for square in moving_cells:
                        if cells_out == 1:
                            square.move_to_target(target, through_friendly = False)
                        else:
                            square.move_to_target(target, through_friendly = True)
                        needed_strength_from_movers -= square.strength
                        if needed_strength_from_movers < 0:
                            break
                
                # Moves have been pended, let's see if we can actually resolve them.
                return True
            else:
                cells_out += 1
        return False
    

    def make_move(self, square, direction):
        self.move_map[square.x, square.y] = direction

        if direction == -1:
            # Reset the move
            if square.target != None:
                square.target.moving_here.remove(square)
                square.target = None
            square.move = -1
            return
                
        if square.move != -1:
            square.target.moving_here.remove(square)
            
        square.move = direction
        if direction != STILL:
            square.target = self.get_target(square, direction)
            square.target.moving_here.append(square)
        
    def get_target(self, square, direction):
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
        return self.squares[(square.x + dx) % self.width][(square.y + dy) % self.height]
        
        
    def friendly_flood_fill(self, source, up_to):
        queue = [(source, 0)]
        flood_fill_map = numpy.ones((self.width, self.height)) * -1
        while len(queue) > 0:
            target = queue.pop(0)
            target_square = target[0]
            if flood_fill_map[target_square.x, target_square.y] == -1:
                # We haven't visited this yet.
                flood_fill_map[target_square.x, target_square.y] = target[1]
                # Add neighbors to the queue
                if target[1] < up_to:
                    # Not yet at the max distance, let's add friendly neighbors to the queue
                    # Should we limit only to cells which are staying still? I think so... remove if it screws things up.
                    neighbors = [n for n in target_square.neighbors() if n.owner == self.my_id and n.move == -1]
                    for n in neighbors:
                        queue.append((n, target[1] + 1))
            else:
                # We could have duplicates but visited the long way...
                if flood_fill_map[target_square.x, target_square.y] > target[1]:
                    # This is a shorter route. Replace!
                    flood_fill_map[target_square.x, target_square.y] = target[1]
                    # Need to add neighbors back in.
                    neighbors = [n for n in target_square.neighbors() if n.owner == self.my_id and n.move == -1]
                    for n in neighbors:
                        queue.append((n, target[1] + 1))
        return flood_fill_map            

    def update_heuristic_map(self):
        # Creates a heuristic map which has the value for every cell.
        
        # Start with future production value
        self.heuristic_map = numpy.zeros((self.width, self.height))
        
        rebased_future_map = self.rebase_map(1 / numpy.copy(self.recover_map[5]))        
        self.heuristic_map += rebased_future_map * future_map_value
        
        rebased_production_map = self.rebase_map(self.production_map)
        self.heuristic_map += numpy.multiply(rebased_production_map, self.is_owned_map) * owned_production_value
        self.heuristic_map += numpy.multiply(rebased_production_map, self.is_enemy_map) * enemy_production_value
        self.heuristic_map += numpy.multiply(rebased_production_map, self.is_neutral_map) * neutral_production_value

        self.heuristic_map += self.combat_zone_map * combat_zone_value
        self.heuristic_map += self.enemy_border_map * enemy_border_value
        
        self.heuristic_map += self.is_owned_map * is_owned_value
        
        self.heuristic_map = spread_n(self.heuristic_map, 10, decay = 0.3)
        
        
        
        





    
#==============================================================================
# Square class        
#==============================================================================
class Square:
    def __init__(self, game, x, y, owner, strength, production):
        
        self.game = game
        self.x = x
        self.y = y
        self.owner = owner
        self.strength = strength
        self.production = production
        
        #self.game_frame = self.game.game_frame[self.game.frame]
        
        self.move = -1
        self.target = None
        self.moving_here = []
        self._neighbors_1 = None
    
    def neighbors(self, n = 1, include_self = False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            if self._neighbors_1 != None:
                return self._neighbors_1
            else:
                combos = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
                self._neighbors_1 = [self.game.game_frame[self.game.frame].squares[(self.x + dx) % self.game.width][(self.y + dy) % self.game.height] for dx, dy in combos if include_self or dx or dy]
                return self._neighbors_1
        else:
            combos = ((dx, dy) for dy in range(-n, n+1) for dx in range(-n, n+1) if abs(dx) + abs(dy) <= n)
        return (self.game.game_frame[self.game.frame].squares[(self.x + dx) % self.game.width][(self.y + dy) % self.game.height] for dx, dy in combos if include_self or dx or dy)
        
    def is_inner_border(self):
        # Checks to see if this square is owned by us AND borders a square NOT owned by us
        return self.game.game_frame[self.game.frame].inner_border_map[self.x, self.y]

    def is_neutral_border(self):
        # Checks to see if this square is owned by neutral (playerid = 0) and next to a square we own
        return self.game.game_frame[self.game.frame].border_map[self.x, self.y]

    def is_contested(self):
        # A square is contested if the strength is 0, is neutral, and is bordered by both a player and enemy owned square
        return self.game.game_frame[self.game.frame].combat_zone_map[self.x, self.y]

    def move_to_target(self, target, through_friendly):
        game_frame = self.game.game_frame[self.game.frame]
        # Attempts to move to the designated target
        # Does so naively. Perhaps replace this with an A* or Dijkstra's algorithm at some point
    
        # Calculate cardinal direction distance to target.
        dist_w = (self.x - target.x) % self.game.width
        dist_e = (target.x - self.x) % self.game.width
        dist_n = (self.y - target.y) % self.game.height
        dist_s = (target.y - self.y) % self.game.height        
        
        if dist_w == 0 and dist_n == 0:
            return False

        w_neighbor = game_frame.get_target(self, WEST)            
        e_neighbor = game_frame.get_target(self, EAST)
        n_neighbor = game_frame.get_target(self, NORTH)
        s_neighbor = game_frame.get_target(self, SOUTH)
        
        if dist_w < dist_e:
            if through_friendly and w_neighbor.owner != self.game.my_id:
                if e_neighbor.owner == self.game.my_id:
                    ew_move = (EAST, e_neighbor)
                else:
                    ew_move = None
            else: 
                ew_move = (WEST, w_neighbor)
        elif dist_e < dist_w:
            if through_friendly and e_neighbor.owner != self.game.my_id:
                if w_neighbor.owner == self.game.my_id:
                    ew_move = (WEST, w_neighbor)
                else:
                    ew_move = None
            else: 
                ew_move = (EAST, e_neighbor)
        elif dist_w == 0:
            ew_move = None
        elif dist_w == dist_e:
            if through_friendly and (w_neighbor.owner != self.game.my_id or e_neighbor.owner != self.game.my_id):
                if w_neighbor.owner != self.game.my_id and e_neighbor.owner != self.game.my_id:
                    ew_move = None
                elif w_neighbor.owner == self.game.my_id and e_neighbor.owner != self.game.my_id:
                    ew_move = (WEST, w_neighbor)
                else:
                    ew_move = (EAST, e_neighbor)
            else:
                # Prefer the move with lower production
                if e_neighbor.production < w_neighbor.production:
                    ew_move = (EAST, e_neighbor)
                else:
                    ew_move = (WEST, w_neighbor)
                               
        if dist_s < dist_n:
            if through_friendly and s_neighbor.owner != self.game.my_id:
                if n_neighbor.owner == self.game.my_id:
                    ns_move = (NORTH, n_neighbor)
                else:
                    ns_move = None
            else: 
                ns_move = (SOUTH, s_neighbor)
        elif dist_n < dist_s:
            if through_friendly and n_neighbor.owner != self.game.my_id:
                if s_neighbor.owner == self.game.my_id:
                    ns_move = (SOUTH, s_neighbor)
                else:
                    ns_move = None
            else: 
                ns_move = (NORTH, n_neighbor)
        elif dist_s == 0:
            ns_move = None
        elif dist_s == dist_n:
            if through_friendly and (s_neighbor.owner != self.game.my_id or n_neighbor.owner != self.game.my_id):
                if s_neighbor.owner != self.game.my_id and n_neighbor.owner != self.game.my_id:
                    ns_move = None
                elif s_neighbor.owner == self.game.my_id and n_neighbor.owner != self.game.my_id:
                    ns_move = (SOUTH, s_neighbor)
                else:
                    ns_move = (NORTH, n_neighbor)
            else:
                # Prefer the move with lower production
                if n_neighbor.production < s_neighbor.production:
                    ns_move = (NORTH, n_neighbor)
                else:
                    ns_move = (SOUTH, s_neighbor)
                             
        if ns_move == None and ew_move == None:
            return False

        possible_moves = []            
        if ns_move == None:
            possible_moves.append(ew_move)
        elif ew_move == None:
            possible_moves.append(ns_move)
        else:
            if ew_move[1].production < ns_move[1].production:
                possible_moves.append(ew_move)
                possible_moves.append(ns_move)
            else:
                possible_moves.append(ns_move)
                possible_moves.append(ew_move)
                
        # We now have a list of possible moves of either length 1 or 2, and only along the x or y axis.
        # Can we safely move into one of these squares?
        #logging.debug(possible_moves)
        future_strength = self.strength
        future_strength += possible_moves[0][1].strength if (possible_moves[0][1].owner == self.game.my_id and (possible_moves[0][1].move == -1 or possible_moves[0][1].move == STILL)) else 0
        if possible_moves[0][1].moving_here != []:
            future_strength += sum(x.strength for x in possible_moves[0][1].moving_here)
        
        if future_strength <= 255 + strength_buffer:
            # We're good, make the move
            self.game.game_frame[self.game.frame].make_move(self, possible_moves[0][0])
            return True
            
        # Can we test the second move if it exists?
        if len(possible_moves) > 1:
            future_strength = self.strength + possible_moves[1][1].strength if (possible_moves[1][1].owner == self.game.my_id and (possible_moves[1][1].move == -1 or possible_moves[1][1].move == STILL)) else 0
            if possible_moves[1][1].moving_here != []:
                future_strength += sum(x.strength for x in possible_moves[1][1].moving_here)
            
            if future_strength <= 255 + strength_buffer:
                # We're good, make the move
                self.game.game_frame[self.game.frame].make_move(self, possible_moves[1][0])
                return True            
        
        # Ok, so moving here will result in too much strength. What are our options?
        # Can we move the cell that we are moving to?
        if possible_moves[0][1].owner == self.game.my_id and (possible_moves[0][1].move == -1 or possible_moves[0][1] == STILL):
            if self.strength + sum(x.strength for x in possible_moves[0][1].moving_here) <= 255 + strength_buffer:
                # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                # This is dangerous, make sure to UNDO the fake move.
                game_frame.make_move(self, possible_moves[0][0])
                success = possible_moves[0][1].move_to_target(target, False)
                if success:
                    return True
                else:
                    # UNDO THE MOVE
                    game_frame.make_move(self, -1)
                # Is there anywhere else we can move this cell?
                if possible_moves[0][1].moving_here != []:
                    for secondary_target in possible_moves[0][1].moving_here:
                        # Simulate the move
                        game_frame.make_move(self, possible_moves[0][0])
                        success = possible_moves[0][1].move_to_target(secondary_target.target, False)
                        if success:
                            return True
                        game_frame.make_move(self, -1)
                # Ok, can we just move the destination to a different square?
                neighbor_targets = []
                for n in possible_moves[0][1].neighbors():
                    neighbor_strength = n.strength if n.owner == self.game.my_id else 0
                    neighbor_strength += sum(x.strength for x in n.moving_here)
                    neighbor_targets.append((n, neighbor_strength))
                # Try to move to the lowest strength target.
                neighbor_targets.sort(key = lambda x: x[1])
                # Attempt to move to the lowest strength neighbor
                for n_t in neighbor_targets:
                    if n_t[0].owner != self.game.my_id:
                        # We're attempting to attack a cell
                        if n_t[0].strength < possible_moves[0][1].strength + sum(x.strength for x in n_t[0].moving_here):
                            if possible_moves[0][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                game_frame.make_move(self, possible_moves[0][0])
                                possible_moves[0][1].move_to_target(n_t[0], False)
                                return True
                    else:
                        future_n_strength = possible_moves[0][1].strength
                        future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                        future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                        if future_n_strength <= 255 + strength_buffer:
                            game_frame.make_move(self, possible_moves[0][0])
                            possible_moves[0][1].move_to_target(n_t[0], True)
                            return True
                        else:
                            break
        # Ok, the cell we are moving to isn't the problem. WE are. Let's try the secondary direction
        if len(possible_moves) > 1:
            if possible_moves[1][1].owner == self.game.my_id and (possible_moves[1][1].move == -1 or possible_moves[1][1] == STILL):
                if self.strength + sum(x.strength for x in possible_moves[1][1].moving_here) <= 255 + strength_buffer:
                    # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                    # This is dangerous, make sure to UNDO the fake move.
                    game_frame.make_move(self, possible_moves[1][0])
                    success = possible_moves[1][1].move_to_target(target, False)
                    if success:
                        return True
                    else:
                        # UNDO THE MOVE
                        game_frame.make_move(self, -1)
                    # Is there anywhere else we can move this cell?
                    if possible_moves[1][1].moving_here != []:
                        for secondary_target in possible_moves[0][1].moving_here:
                            # Simulate the move
                            game_frame.make_move(self, possible_moves[1][0])
                            success = possible_moves[1][1].move_to_target(secondary_target.target, False)
                            if success:
                                return True
                            game_frame.make_move(self, -1)
                    # Ok, can we just move the destination to a different square?
                    neighbor_targets = []
                    for n in possible_moves[1][1].neighbors():
                        neighbor_strength = n.strength if n.owner == self.game.my_id else 0
                        neighbor_strength += sum(x.strength for x in n.moving_here)
                        neighbor_targets.append((n, neighbor_strength))
                    # Try to move to the lowest strength target.
                    neighbor_targets.sort(key = lambda x: x[1])
                    # Attempt to move to the lowest strength neighbor
                    for n_t in neighbor_targets:
                        if n_t[0].owner != self.game.my_id:
                            # We're attempting to attack a cell
                            if n_t[0].strength < possible_moves[1][1].strength + sum(x.strength for x in n_t[0].moving_here):
                                if possible_moves[1][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                    game_frame.make_move(self, possible_moves[1][0])
                                    possible_moves[1][1].move_to_target(n_t[0], False)
                                    return True
                        else:
                            future_n_strength = possible_moves[1][1].strength
                            future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                            future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                            if future_n_strength <= 255 + strength_buffer:
                                game_frame.make_move(self, possible_moves[1][0])
                                possible_moves[1][1].move_to_target(n_t[0], True)
                                return True
                            else:
                                break
        # We can't do anything.
        return False
                    

#==============================================================================
# Functions for communicating with the Halite game environment (formerly contained in separate module networking.py        
#==============================================================================
def translate_cardinal(direction):
    # Cardinal index used by the framework is:
    # NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3, STILL = 4
    # Cardinal index used by the game is:
    # STILL = 0, NORTH = 1, EAST = 2, SOUTH = 3, WEST = 4
    return int((direction + 1) % 5)

def send_string(to_be_sent):
    sys.stdout.write(to_be_sent + "\n")
    sys.stdout.flush()

def get_string():
    return sys.stdin.readline().rstrip('\n')

        
#==============================================================================
# Helper Functions        
#==============================================================================

def get_distance(sq1, sq2):
    dx = abs(sq1.x - sq2.x)
    dy = abs(sq1.y - sq2.y)
    if dx > game.width / 2:
        dx = game.width - dx
    if dy > game.height / 2:
        dy = game.height - dy
    return dx + dy
        
def get_offset(direction):
    return ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]

def distance_between(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dx > game.width / 2:
        dx = game.width - dx
    if dy > game.height / 2:
        dy = game.height - dy
    return dx + dy        

def opposite_direction(direction):
    return (direction + 2) % 4 if direction != STILL else STILL

def roll_x(M, x):
    return numpy.roll(M, x, 0)

def roll_y(M, y):
    return numpy.roll(M, y, 1)

def roll_xy(M, x, y):
    return numpy.roll(numpy.roll(M, x, 0), y, 1)
    
def get_all_d_away(d):
    # Returns a list of dx, dy that are exactly D away from the origin
    combos = []
    for x in range(0, d+1):
        x_vals = list(set([x, -x]))
        y_vals = list(set([d-x, -(d-x)]))
        combos.extend(list(itertools.product(x_vals, y_vals)))
    return list(set(combos))    
    
def spread_n(M, n, decay = 0, include_self = True):
    # Takes a matrix M, and then creates an influence map by offsetting by N in every direction. 
    # Decay function is currently of the form exp(-decay * distance)
    if include_self == True:
        spread_map = numpy.copy(M)
    else:
        spread_map = numpy.zeros_like(M)
    distance = 1
    while distance <= n:
        combos = get_all_d_away(distance)
        decay_factor = math.exp(-decay * distance)
        for c in combos:
            spread_map += roll_xy(numpy.multiply(decay_factor, M), c[0], c[1])
        distance += 1
    return spread_map
    
def spread(M, decay = 0, include_self = True):
    # For now to save time, we'll use game_map.distance_map and assume that we'll always be using the same falloff distances to calculate offsets.
    
    # Takes the matrix M and then for each point (x, y), calculate the product of the distance map and the decay factor.
    decay_map = numpy.exp(numpy.multiply(game.distance_map, -decay))
    
    spread_map = numpy.sum(numpy.multiply(decay_map, M), (2, 3))
    return spread_map    
    
def distance_from_owned(M, mine):
    # Returns the minimum distance to get to any point if already at all points in xys using 4D array M
    return numpy.apply_along_axis(numpy.min, 0, M[numpy.nonzero(mine)])    
 
#==============================================================================
# Game Loop
#==============================================================================
def game_loop():
    
    game.update_next_frame()
#    game.update()

    #logging.debug("\nFrame: " + str(game_map.frame))
    #logging.debug("width:" + str(game_map.width) + " height: " + str(game_map.height))
#        
#    if game.phase == 0:
#        #start = time.time()
#        first_turns_heuristic()
#        #end = time.time()
#        #logging.debug("13.4.1 Frame: " + str(game_map.frame) + " : " + str(end - start))
#    elif game.phase == 1:
#        game.get_best_moves()
#    else:
#        #game_map.get_best_moves()
#        #game_map.late_game_attack()
#        game.get_best_moves_late_game()

    game.get_moves()

    #for square in square_move_list:
    #    game_map.get_best_move(square)
    # Project the state of the board assuming for now that enemy pieces do not move    
    #game_map.create_projection()    
    # Do stuff

    
    #over_count = game_map.width * game_map.height

    #new_over_count = game_map.prevent_overstrength()

    #while new_over_count < over_count:
    #    over_count = new_over_count
    #    new_over_count = game_map.prevent_overstrength()

    
    game.game_frame[game.frame].send_frame()

    
#==============================================================================
# Game run-time code
#==============================================================================
logging.basicConfig(filename='logging.log',level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)

game = Game()


while True:
    game_loop()