#==============================================================================
# Imports
#==============================================================================
import functools
import itertools
import logging
import math
import numpy as np
import random
import sys
import time

#==============================================================================
# Variables
#==============================================================================
botname = "shummie v35"

#buildup_multiplier = 11
strength_buffer = 0
#pre_combat_threshold = 2
#combat_radius = 8
#production_cells_out = 8

#==============================================================================
# Game Class
#==============================================================================

class Game:
    def __init__(self):
        # This should only be called once, and at the beginning of the game
        self.my_id = int(get_string())
        
        map_size_string = get_string()
        self.width, self.height = tuple(map(int, map_size_string.split()))
        
        production_map_string = get_string()
        self.production_map = np.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()
        
        self.create_squares_list()
        
        self.frame = -1
    
        self.get_frame()
            
        self.starting_player_count = np.amax(self.owner_map) # Note, for range you'd need to increase the range by 1
        
        # Create the distance map
        self.create_one_time_maps()
        
        self.max_turns = 10 * ((self.width * self.height) ** 0.5)
        
        self.set_configs()
        
        # Send the botname
        send_string(botname)

    def __iter__(self):
        # Allows direct iteration over all squares
        return itertools.chain.from_iterable(self.squares)        

    def get_frame(self, map_string = None):
        # Updates the map information from the latest frame provided by the game environment
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
    
        self.owner_map = np.array(owners).reshape((self.height, self.width)).transpose()
        
        # This is then followed by WIDTH * HEIGHT integers, representing the strength values of the tiles in the map. 
        # It fills in the map in the same way owner values fill in the map.        
        assert len(split_string) == self.width * self.height
        str_list = list(map(int, split_string))
        
        self.strength_map = np.array(str_list).reshape((self.height, self.width)).transpose()
        
        # Update all squares
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y].update(self.owner_map[x, y], self.strength_map[x, y])
        
        # Reset the move_map
        self.move_map = np.ones((self.width, self.height)) * -1  # Could possibly expand this in the future to consider enemy moves...
        
        self.frame += 1

    def send_frame(self):
        # Goes through each square and get the list of moves.
        move_list = []
        for sq in itertools.chain.from_iterable(self.squares):
            if sq.owner == self.my_id:
                if sq.strength == 0: # Squares with 0 strength shouldn't move.
                    sq.move = 4
                if sq.move == -1:
                    # In the event we didn't actually assign a move, make sure it's coded to STILL
                    sq.move = 4
                move_list.append(sq)
        
        send_string(' '.join(str(square.x) + ' ' + str(square.y) + ' ' + str(translate_cardinal(square.move)) for square in move_list))                

    def create_squares_list(self):
        self.squares = np.empty((self.width, self.height), dtype = np.object)
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y] = Square(self, x, y, self.production_map[x, y])
    
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y].after_init_update()
    
        
    def create_one_time_maps(self):
        self.distance_map = self.create_distance_map()        
        self.distance_map_no_decay = self.create_distance_map(1)
        
        self.production_map_01 = np.maximum(self.production_map, 0.1)            
        self.production_map_1 = np.maximum(self.production_map, 1)

    def create_distance_map(self, falloff = 1):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.
        
        # Create the base map for 0, 0
        zero_zero_map = np.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                dist_x = min(x, -x % self.width)
                dist_y = min(y, -y % self.width)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        zero_zero_map = zero_zero_map ** falloff
        
        distance_map = np.zeros((self.width, self.height, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)
                
        return distance_map
    
    def set_configs(self):
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 3), 10)
        self.pre_combat_threshold = 2
        self.combat_radius = 8
        self.production_cells_out = int(self.width / self.starting_player_count / 1.5)
    
    def update_configs(self):
        #self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 5), 5)
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 3), 6)
        self.buildup_multiplier = self.buildup_multiplier + (self.distance_from_border ** 0.5)
        #self.combat_radius = int(min(max(5, self.percent_owned * self.width / 2), self.width // 2))
        self.combat_radius = 8
        
        if np.sum(self.combat_zone_map) > 3:
            self.production_cells_out = int(self.width / self.starting_player_count / 2.5)
        
        if self.percent_owned > 0.6:
            self.buildup_multiplier -= 1  
            self.pre_combat_threshold = 0
            self.combat_radius = 10
        
        
    def update(self):
#        start = time.time()
        self.update_maps()
#        end = time.time()
#        logging.debug("update_maps Frame: " + str(game.frame) + " : " + str(end - start))
        self.update_stats()
        self.update_configs()

    def update_maps(self):
        self.update_calc_maps() 
        self.update_owner_maps()
        #start = time.time()
        self.update_distance_maps()
        #end = time.time()
        #logging.debug("update_dist_maps Frame: " + str(game.frame) + " : " + str(end - start))
        self.update_border_maps()
        #start = time.time()
        self.update_enemy_maps()
        #end = time.time()
        #logging.debug("update_enemymaps Frame: " + str(game.frame) + " : " + str(end - start))
        #start = time.time()
        
        self.update_recover_maps()
        #end = time.time()
        #logging.debug("update_recover Frame: " + str(game.frame) + " : " + str(end - start))
        self.update_value_production_map()
        self.update_controlled_influence_production_maps()
        
    def update_calc_maps(self):
        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)
    
    def update_owner_maps(self):
        self.is_owned_map = np.zeros((self.width, self.height))
        self.is_neutral_map = np.zeros((self.width, self.height))
        self.is_enemy_map = np.zeros((self.width, self.height))
        
        self.is_owned_map[np.where(self.owner_map == self.my_id)] = 1
        self.is_neutral_map[np.where(self.owner_map == 0)] = 1    
        self.is_enemy_map = 1 - self.is_owned_map - self.is_neutral_map

    def update_distance_maps(self):
        # Relatively expensive operation
        #self.distance_from_owned = distance_from_owned(self.distance_map_no_decay, self.is_owned_map)
        #self.distance_from_owned[self.is_owned_map == 1] = 0
        return
        #self.distance_from_border = distance_from_owned(self.distance_map_no_decay, 1 - self.is_owned_map)
        #self.distance_from_border[1 - (self.is_owned_map == 1)] = 0
                                  
        
                                  
        #self.distance_from_border = self.friendly_flood_fill_multiple_sources()

#        if self.starting_player_count > 1:  # Breaks in single player mode otherwise.
#            self.distance_from_enemy = distance_from_owned(self.distance_map_no_decay, self.is_enemy_map)
#            self.distance_from_enemy[self.is_enemy_map == 1] = 999
#        else:
#            self.distance_from_enemy = np.ones((self.width, self.height)) * 999    
#        
    def update_border_maps(self):
        self.border_map = np.zeros((self.width, self.height))
        #self.inner_border_map = np.zeros((self.width, self.height))
        self.combat_zone_map = np.zeros((self.width, self.height))
        
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == 0:
                for n in square.neighbors:
                    if n.owner == self.my_id:
                        self.border_map[square.x, square.y] = 1
                        continue
                    
        border_squares_indices = np.transpose(np.nonzero(self.border_map))
        border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        self.distance_from_border = self.friendly_flood_fill_multiple_sources(border_squares, max(self.width, self.height))

        owned_squares_indices = np.transpose(np.nonzero(self.is_owned_map))
        owned_squares = [self.squares[c[0], c[1]] for c in owned_squares_indices]
        self.distance_from_owned = self.friendly_flood_fill_multiple_sources(owned_squares, max(self.width, self.height))                    
        #self.border_map = (self.distance_from_owned == 1) * 1
        #self.border_indices = np.transpose(np.where(self.border_map == 1))

        #self.inner_border_map = (self.distance_from_border == 1) * 1
        #self.inner_border_indices = np.transpose(np.where(self.inner_border_map == 1))
        
        self.combat_zone_map = self.border_map * (self.strength_map == 0)
        
#        if self.starting_player_count > 1 and np.sum(self.combat_zone_map) >= 1:  # Breaks in single player mode otherwise.
#            self.distance_from_combat_zone = distance_from_owned(self.distance_map_no_decay, self.combat_zone_map)
#            self.distance_from_combat_zone += (1-self.is_owned_map) * 999
#        else:
#            self.distance_from_combat_zone = np.ones((self.width, self.height)) * 999

    def update_enemy_maps(self):
        self.enemy_strength_map = np.zeros((5, self.width, self.height))
        self.enemy_strength_map[0] = self.strength_map * self.is_enemy_map
    
        for x in range(len(self.enemy_strength_map)):
            self.enemy_strength_map[x] = spread_n(self.enemy_strength_map[0], x)

        self.own_strength_map = np.zeros((5, self.width, self.height))
        self.own_strength_map[0] = self.strength_map * self.is_owned_map
    
        for x in range(len(self.own_strength_map)):
            self.own_strength_map[x] = spread_n(self.own_strength_map[0], x)            
            
    def update_recover_maps(self):
        #max_distance = min(self.width // 2, self.height // 2)
        max_distance = int(min(0.8 * self.width, 0.8 * self.height))
        #self.recover_map = np.zeros((max_distance + 1, self.width, self.height))
        #self.recover_map[0] = np.divide(self.strength_map, self.production_map_01) * (self.is_neutral_map - self.combat_zone_map)
        
        self.prod_over_str_map = np.zeros((max_distance + 1, self.width, self.height))
        #self.prod_over_str_map[0] = np.divide(self.production_map, self.strength_map_01) * (self.is_neutral_map - self.combat_zone_map)
        new_str_map = np.copy(self.strength_map)
        new_str_map[new_str_map == 0] = 40
        #self.prod_over_str_map[0] = np.divide(self.production_map, self.strength_map_01) * (self.is_neutral_map - self.combat_zone_map)
        self.prod_over_str_map[0] = np.divide(self.production_map * 1.45, new_str_map) * (self.is_neutral_map - self.combat_zone_map)
        #self.recover_map[0] = 1 / np.maximum(self.prod_over_str_map[0], 0.01)
        
        for distance in range(1, max_distance + 1):
            self.prod_over_str_map[distance] = spread_n(self.prod_over_str_map[distance - 1], 1)
            self.prod_over_str_map[distance][self.prod_over_str_map[distance-1] == 0] = 0
            self.prod_over_str_map[distance] = self.prod_over_str_map[distance] / 5
            #self.recover_map[distance] = 1 / np.maximum(self.prod_over_str_map[distance], 0.01)

        self.prod_over_str_max_map = np.apply_along_axis(np.max, 0, self.prod_over_str_map)
        #self.recover_max_map = 1 / np.maximum(self.prod_over_str_max_map, 0.01)
        self.prod_over_str_avg_map = np.apply_along_axis(np.mean, 0, self.prod_over_str_map)
        #self.recover_avg_map = 1 / np.maximum(self.prod_over_str_avg_map, 0.01)
        self.prod_over_str_wtd_map = (self.prod_over_str_max_map + self.prod_over_str_avg_map) / 2
        self.recover_wtd_map = 1 / np.maximum(self.prod_over_str_wtd_map, 0.01)

    def update_value_production_map(self):
        self.value_production_map = (self.border_map - self.combat_zone_map * (self.enemy_strength_map[1] == 0)) * self.recover_wtd_map
        #self.value_production_map = (self.border_map - self.combat_zone_map) * self.recover_wtd_map
        self.value_production_map[self.value_production_map == 0] = 9999
        turns_left = self.max_turns - self.frame
        recover_threshold = turns_left * 0.6
        self.value_production_map[self.value_production_map > recover_threshold] == 9999
        bx, by = np.unravel_index(self.value_production_map.argmin(), (self.width, self.height))
        best_cell_value = self.value_production_map[bx, by]

        avg_recov_threshold = 2
        avg_map_recovery = np.sum(self.strength_map * self.border_map) / np.sum(self.production_map * self.border_map)
        self.value_production_map[self.value_production_map > (avg_recov_threshold * avg_map_recovery)] = 9999
                                  
    def update_controlled_influence_production_maps(self):
        max_distance = 9
        self.controlled_production_influence_map = np.zeros((max_distance + 1, self.width, self.height))
        self.controlled_production_influence_map[0] = self.production_map * (self.is_enemy_map + self.is_owned_map)
        for distance in range(1, max_distance + 1):
            self.controlled_production_influence_map[distance] = spread_n(self.controlled_production_influence_map[distance - 1], 1)
            self.controlled_production_influence_map[distance] = rebase_map(self.controlled_production_influence_map[distance - 1], False)
                                  
    def get_moves(self):
        # This is the main logic controlling code.
        # Find super high production cells
        self.get_pre_combat_production()
        # 1 - Find combat zone cells and attack them.
#        start = time.time()

        self.get_moves_attack()
#        end = time.time()
#        logging.debug("get_move_attack Frame: " + str(game.frame) + " : " + str(end - start))
        # 2 - Find production zone cells and attack them
#        start = time.time()
        self.get_moves_production()
#        end = time.time()
#        logging.debug("get production moves Frame: " + str(game.frame) + " : " + str(end - start))
        # 3 - Move all other unassigned cells.
#        start = time.time()
        self.get_moves_other()
#        end = time.time()
#        logging.debug("get other moves Frame: " + str(game.frame) + " : " + str(end - start))    
        
    def get_pre_combat_production(self):
        # In the event we are trying to fight in a very high production zone, reroute some attacking power to expand in this area.
        potential_targets_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices if (self.recover_wtd_map[c[0], c[1]] < self.pre_combat_threshold)]
        if len(potential_targets) == 0: 
            return
            
        potential_targets.sort(key = lambda sq: self.recover_wtd_map[sq.x, sq.y])
        
        best_target_value = self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y]
        # anything with X of the best_value target should be considered. Let's set this to 4 right now.
        while len(potential_targets) > 0 and self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y] <= (best_target_value + 2):
            target = potential_targets.pop(0)
            self.attack_cell(target, 2)        
        
    def get_moves_attack(self):
        # Attempts to attack all border cells that are in combat
        potential_targets_indices = np.transpose(np.nonzero(self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]
        #potential_targets.sort(key = lambda x: self.distance_from_enemy[x.x, x.y])
        potential_targets.sort(key = lambda x: self.enemy_strength_map[2, x.x, x.y], reverse = True)
        
        # TODO: Should sort by amount of overkill damage possible.
        for square in potential_targets:
            self.attack_cell(square, 1)
    
        self.get_moves_breakthrough()
        # Get a list of all squares within 5 spaces of a combat zone.
        # TODO: This causes bounciness, i should probably do a floodfill of all combat zone squares instead?
        combat_zone_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]
        combat_distance_matrix = self.friendly_flood_fill_multiple_sources(combat_zone_squares, self.combat_radius)
        combat_distance_matrix[combat_distance_matrix == -1] = 0
        combat_distance_matrix[combat_distance_matrix == 1] = 0
        combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]        
        combat_squares.sort(key = lambda x: x.strength, reverse = True)
#        combat_squares_indices = np.transpose(np.nonzero((self.distance_from_combat_zone <= combat_radius) * (self.move_map == -1)))
#        combat_squares = [self.squares[c[0], c[1]] for c in combat_squares_indices]
        
        for square in combat_squares:
            if (square.strength > square.production * self.buildup_multiplier[square.x, square.y]) and ((square.x + square.y) % 2 == self.frame % 2) and square.move == -1:
#                self.move_towards_map(square, self.distance_from_combat_zone)
                self.move_towards_map_old(square, combat_distance_matrix)
            else:
                self.make_move(square, STILL, None)

    def find_nearest_non_owned_border(self, square):
                
        current_distance = self.distance_from_border[square.x, square.y]
        for n in square.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_border[n.x, n.y] < current_distance:
                    success = self.move_square_to_target(square, n, True)
                    if success:
                        break                    
                
    def move_towards_map(self, square, distance_map):
        current_distance = distance_map[square.x, square.y]
        queue = [square]
        targets = []
        while len(queue) > 0:
            current = queue.pop(0)
            current_distance = distance_map[current.x, current.y]
            for n in current.neighbors:
                if distance_map[n.x, n.y] == 0:
                    targets.append(n)
                elif distance_map[n.x, n.y] <= current_distance - 1:
                    queue.append(n)
        random.shuffle(targets)
        target = targets.pop(0)
        success = self.move_square_to_target(square, target, True)
#        while len(targets) > 0:
#            target = targets.pop(0)
#            success = self.move_square_to_target(square, target, True)
#            if success:
#                return
                
    def move_towards_map_old(self, square, distance_map, through_friendly = True):
        current_distance = distance_map[square.x, square.y]
        possible_moves = []
        for n in square.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if distance_map[n.x, n.y] < current_distance:
                    possible_moves.append(n)
        if len(possible_moves) > 0:
            random.shuffle(possible_moves)
            possible_moves.sort(key = lambda sq: self.enemy_strength_map[4, sq.x, sq.y], reverse = True)
            self.move_square_to_target(square, possible_moves[0], True)
        
    def get_moves_production(self):
        # Tries to find the best cells to attack from a production standpoint.
        # Does not try to attack cells that are in combat zones.
        # potential_targets_indices = np.transpose(np.nonzero((self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] == 0)))
        potential_targets_indices = np.transpose(np.nonzero((self.value_production_map != 9999)))
        potential_targets = [(self.squares[c[0], c[1]], self.value_production_map[c[0], c[1]], 1) for c in potential_targets_indices]

        potential_targets = []
        for c in potential_targets_indices:
            target = self.squares[c[0], c[1]]
            value = self.value_production_map[c[0], c[1]]
            cells_out = 1
            while cells_out <= self.production_cells_out:
                potential_targets.append((target, value, cells_out))
                cells_out += 1
        
        if len(potential_targets) == 0: 
            return
        potential_targets.sort(key = lambda x: x[0].strength)    
        potential_targets.sort(key = lambda x: x[1] + (x[2] * 2))
        
        # Keep only the top 80ile?
        #potential_targets = potential_targets[0:int(len(potential_targets) * .9)]
        
#        best_target_value = potential_targets[0][1]
        # anything with X of the best_value target should be considered. Let's set this to 4 right now.
        while len(potential_targets) > 0: # and potential_targets[0][1] <= (best_target_value + 4000):
            target = potential_targets.pop(0)
            success = self.attack_cell(target[0], target[2], target[2])
            if success and target[2] < self.production_cells_out:
                potential_targets = list(filter(lambda sq: sq[0] != target[0], potential_targets))


    def get_moves_breakthrough(self):
        # Determine if we should bust through and try to open up additional lanes of attack into enemy territory
        # Best to have a separate lane. so we should evaluate squares that are not next to already open channels.
        # We are only looking at squares which are next to the enemy already.
        potential_squares_indices = np.transpose(np.nonzero((self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] > 0)))
        potential_squares = [self.squares[c[0], c[1]] for c in potential_squares_indices]
        # We only want to bust through if we have a lot of strength here.
        #logging.debug(str(self.own_strength_map[4]))
        for square in potential_squares:
            if self.own_strength_map[4, square.x, square.y] > 1750 and (self.own_strength_map[4, square.x, square.y] > 1.5 * self.enemy_strength_map[4, square.x, square.y]):
                self.attack_cell(square, 1)
                
        
    def get_moves_other(self):
        # Tries to move to 
        idle_squares_indices = np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))
        idle_squares = [self.squares[c[0], c[1]] for c in idle_squares_indices]

        if len(idle_squares) == 0:
            return
        
        # Move squares closer to the border first.
        idle_squares.sort(key = lambda sq: self.distance_from_border[sq.x, sq.y])
        
        for square in idle_squares:
            if square.strength > square.production * self.buildup_multiplier[square.x, square.y] and square.move == -1:
                if self.percent_owned > 0.65:
                    self.find_nearest_non_owned_border(square)
                    #self.move_towards_map(square, self.distance_from_border)
                else:
                    # Move towards the closest border
                    #if not self.inner_border_map[square.x, square.y]:
                        # For now, move to the square with the lowest recovery
                    value_map = (self.value_production_map + self.distance_map_no_decay[square.x, square.y] * 1) * self.border_map
                    #best_target_value = (self.recover_wtd_map * (self.border_map - self.combat_zone_map)).argmin()
                    #value_map = value_map * (1 - self.combat_zone_map)
                    value_map[np.nonzero(self.combat_zone_map)] = 0
                    value_map += self.distance_map_no_decay[square.x, square.y] * 0.85 * self.combat_zone_map 
                    value_map -= self.controlled_production_influence_map[6, square.x, square.y] * 5 * self.combat_zone_map 
                    #value_map[self.combat_zone_map == 1] = self.distance_map_no_decay[square.x, square.y] * .8
                    value_map[value_map == 0] = 9999
                    #tx, ty = np.unravel_index(value_map.argmin(), (self.width, self.height))
                    tx, ty = np.unravel_index(value_map.argmin(), (self.width, self.height))
                    target = self.squares[tx, ty]
                    # We're targeting either a combat square, or a production square. Don't move towards close production squares.
                    if self.combat_zone_map[tx, ty]:
                        if self.distance_between(square, target) > 14:
                            self.move_square_to_target_simple(square, target, True)
                        elif self.distance_between(square, target) > 1:
                            self.move_square_to_target(square, target, True)
                    else:
                        if self.distance_between(square, target) > 14:
                            self.move_square_to_target_simple(square, target, True)
                        elif self.distance_between(square, target) > self.production_cells_out - 1:
                            self.move_square_to_target(square, target, True)
                    
    def distance_between(self, sq1, sq2):
        dx = abs(sq1.x - sq2.x)
        dy = abs(sq1.y - sq2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy
            
            
            
    def attack_cell(self, target, max_cells_out, min_cells_out = 1):
        # Attempts to coordinate attack to a specific cell.
        cells_out = min_cells_out
        
        while cells_out <= max_cells_out:
            # If we're trying to attack a combat zone cell, this isn't the function to do it. cancel.
            if cells_out > 1 and self.combat_zone_map[target.x, target.y]:
                return False
            
            free_squares = self.is_owned_map * (self.move_map == -1)
            target_distance_matrix = self.friendly_flood_fill(target, cells_out)
            target_distance_matrix[target_distance_matrix == -1] = 0
            target_distance_matrix = target_distance_matrix * free_squares
            available_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix, 1))
            
            target_distance_matrix_production = cells_out - target_distance_matrix
            target_distance_matrix_production[target_distance_matrix_production == cells_out] = 0 # Cells furthest out would be moving so no production
            target_distance_matrix_production = target_distance_matrix_production * free_squares
            available_production = np.sum(self.production_map * target_distance_matrix_production)
            
            if available_strength + available_production > target.strength + 0:
                attacking_cells_indices = np.transpose(np.nonzero(target_distance_matrix > 0))
                attacking_cells = [self.squares[c[0], c[1]] for c in attacking_cells_indices]

                still_cells = []
                if cells_out > 1:
                    still_cells_indices = np.transpose(np.nonzero(target_distance_matrix_production> 0))
                    still_cells = [self.squares[c[0], c[1]] for c in still_cells_indices]
                moving_cells = list(set(attacking_cells) - set(still_cells))
                
                for square in still_cells:
                    self.make_move(square, STILL, None)
                    
                still_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix_production, 1))
                needed_strength_from_movers = target.strength - available_production - still_strength + 1
                
                if needed_strength_from_movers > 0:
                    # Handle movement here
                    moving_cells.sort(key = lambda x: x.strength, reverse = True)
                    # There are probably ways to do this more efficiently, for now just start with the highest strength cell 
                    # and work backwards to minimize the # of cells that need to be moved.
                    for square in moving_cells:
                        if square.strength > 0:
                            if cells_out == 1:
                                self.move_square_to_target(square, target, False)
                            else:
                                self.move_square_to_target(square, target, True)
                            needed_strength_from_movers -= square.strength
                            if needed_strength_from_movers < 0:
                                break
                return True
            else:
                cells_out += 1
        return False
    
    def make_move(self, square, direction, far_target):
        self.move_map[square.x, square.y] = direction

        if direction == -1: # Reset the square move
            if square.target != None:
                square.target.moving_here.remove(square)
                square.target = None
                square.far_target = None
            square.move = -1
            square.far_target = None
            return
        
        if square.move != -1:
            if square.target != None:
                square.target.moving_here.remove(square)
                square.target = None
            square.far_target = None
        
        square.move = direction
        if direction != STILL:
            square.target = square.neighbors[direction]
            square.target.moving_here.append(square)
            square.far_target = far_target
            
    def move_square_to_target(self, source, destination, through_friendly):
        # Get the distance matrix that we will use to determine movement.
        
        distance_matrix = self.flood_fill_until_target(source, destination, through_friendly)
        source_distance = distance_matrix[source.x, source.y]
        if source_distance == -1 or source_distance == 0:
            # We couldn't find a path to the destination or we're trying to move STILL
            return False
        
        path_choices = []
        for d in directions:
            if d != STILL:
                neighbor = source.neighbors[d]
                if distance_matrix[neighbor.x, neighbor.y] == (source_distance - 1):
                    path_choices.append((d, neighbor))
        
        # There should be at most 2 cells in path_choices
        path_choices.sort(key = lambda x: x[1].production)
                
        # Implement collision detection later.
        
        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength #+ target.production
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= 255 + strength_buffer:
                self.make_move(source, direction, destination)
                return True
                
        for (direction, target) in path_choices:
            # Ok, can we move the cell that we are moving to:
            if target.owner == self.my_id:
                # Yes. We can, but is the cell staying still? If not, then we can't do anything
                if target.move == STILL or target.move == -1:
                    # Ok, let's make sure that moving this piece actually does something.
                    future_strength = source.strength
                    for sq in target.moving_here:
                        future_strength += sq.strength
                    if future_strength <= 255 + strength_buffer:
                    # Ok, let's move the target square.
                    # Start with trying to move to the same destination as someone moving here.
                        self.make_move(source, direction, destination) # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        for n in target.moving_here:
                            #n = target.neighbors[n_d]
                            if n.owner == self.my_id and n.far_target != None: # The n.owner check is redundant, but just in case.
                                success = self.move_square_to_target(target, n.far_target, True)
                                if success: 
                                    return True
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for n_d in n_directions:
                            n = target.neighbors[n_d]
                            if n.owner == self.my_id:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength # + n.production
                                for n_moving in n.moving_here:
                                    future_n_t_strength += n_moving.strength
                                if future_n_t_strength <= 255 + strength_buffer:
                                    success = self.move_square_to_target_simple(target, n, True)
                                    if success: 
                                        return True
                        # TODO: Logic to attempt to capture a neutral cell if we want.
                        self.make_move(source, -1, None)
        # Nothing to do left
        return False
 

    def move_square_to_target_simple(self, source, destination, through_friendly):
        # For large distances, we can probably get away with simple movement rules.
        dist_w = (source.x - destination.x) % self.width
        dist_e = (destination.x - source.x) % self.width
        dist_n = (source.y - destination.y) % self.height
        dist_s = (destination.y - source.y) % self.height

        if dist_w == 0 and dist_n == 0:
            return False
            
        ew_swap = False
        ns_swap = False
        
        w_neighbor = source.neighbors[WEST]
        e_neighbor = source.neighbors[EAST]
        n_neighbor = source.neighbors[NORTH]
        s_neighbor = source.neighbors[SOUTH]

        if dist_w < dist_e:
            if through_friendly and w_neighbor.owner != self.my_id:
                if e_neighbor.owner == self.my_id:
                    ew_move = (EAST, e_neighbor)
                    ew_swap = True
                else:
                    ew_move = None
            else:
                ew_move = (WEST, w_neighbor)
        elif dist_e < dist_w:
            if through_friendly and e_neighbor.owner != self.my_id:
                if w_neighbor.owner == self.my_id:
                    ew_move = (WEST, w_neighbor)
                    ew_swap = True
                else:
                    ew_move = None
            else:
                ew_move = (EAST, e_neighbor)
        elif dist_w == 0:
            ew_move = None
        elif dist_w == dist_e:
            if through_friendly and (w_neighbor.owner != self.my_id or e_neighbor.owner != self.my_id):
                if w_neighbor.owner != self.my_id and e_neighbor.owner != self.my_id:
                    ew_move = None
                elif w_neighbor.owner == self.my_id and e_neighbor.owner != self.my_id:
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
            if through_friendly and s_neighbor.owner != self.my_id:
                if n_neighbor.owner == self.my_id:
                    ns_move = (NORTH, n_neighbor)
                    ns_swap = True
                else:
                    ns_move = None
            else: 
                ns_move = (SOUTH, s_neighbor)
        elif dist_n < dist_s:
            if through_friendly and n_neighbor.owner != self.my_id:
                if s_neighbor.owner == self.my_id:
                    ns_move = (SOUTH, s_neighbor)
                    ns_swap = True
                else:
                    ns_move = None
            else: 
                ns_move = (NORTH, n_neighbor)
        elif dist_s == 0:
            ns_move = None
        elif dist_s == dist_n:
            if through_friendly and (s_neighbor.owner != self.my_id or n_neighbor.owner != self.my_id):
                if s_neighbor.owner != self.my_id and n_neighbor.owner != self.my_id:
                    ns_move = None
                elif s_neighbor.owner == self.my_id and n_neighbor.owner != self.my_id:
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
            
        path_choices = []
        if ns_move == None:
            path_choices.append(ew_move)
        elif ew_move == None:
            path_choices.append(ns_move)
        elif ns_swap == True and ew_swap == False:
            path_choices.append(ew_move)
            path_choices.append(ns_move)
        elif ns_swap == False and ew_swap == True:
            path_choices.append(ns_move)
            path_choices.append(ew_move)
        else:
            if ew_move[1].production < ns_move[1].production:
                path_choices.append(ew_move)
                path_choices.append(ns_move)
            else:
                path_choices.append(ns_move)
                path_choices.append(ew_move)
                
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength #+ target.production
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= 255 + strength_buffer:
                self.make_move(source, direction, destination)
                return True
                
        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength #+ target.production 
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= 255 + strength_buffer:
                self.make_move(source, direction, destination)
                return True
                
        for (direction, target) in path_choices:
            # Ok, can we move the cell that we are moving to:
            if target.owner == self.my_id:
                # Yes. We can, but is the cell staying still? If not, then we can't do anything
                if target.move == STILL or target.move == -1:
                    # Ok, let's make sure that moving this piece actually does something.
                    future_strength = source.strength
                    for sq in target.moving_here:
                        future_strength += sq.strength
                    if future_strength <= 255 + strength_buffer:
                    # Ok, let's move the target square.
                    # Start with trying to move to the same destination as someone moving here.
                        self.make_move(source, direction, destination) # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        for n in target.moving_here:
                            #n = target.neighbors[n_d]
                            if n.owner == self.my_id and n.far_target != None: # The n.owner check is redundant, but just in case.
                                success = self.move_square_to_target(target, n.far_target, True)
                                if success: 
                                    return True
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for n_d in n_directions:
                            n = target.neighbors[n_d]
                            if n.owner == self.my_id:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength # + n.production
                                for n_moving in n.moving_here:
                                    future_n_t_strength += n_moving.strength
                                if future_n_t_strength <= 255 + strength_buffer:
                                    success = self.move_square_to_target_simple(target, n, True)
                                    if success: 
                                        return True
                        # TODO: Logic to attempt to capture a neutral cell if we want.
                        self.make_move(source, -1, None)
        # Nothing to do left
        return False
                
            

    def flood_fill_until_target(self, source, destination, friendly_only):
        # Does a BFS flood fill to find shortest distance from source to target.
        # Starts the fill AT destination and then stops once we hit the target.
        q = [destination]
        distance_matrix = np.ones((self.width, self.height)) * -1
        distance_matrix[destination.x, destination.y] = 0
        while len(q) > 0 and distance_matrix[source.x, source.y] == -1:
            current = q.pop(0)
            current_distance = distance_matrix[current.x, current.y]
            for neighbor in current.neighbors:
                if distance_matrix[neighbor.x, neighbor.y] == -1:
                    if not friendly_only or (friendly_only and neighbor.owner == self.my_id):
                        distance_matrix[neighbor.x, neighbor.y] = current_distance + 1
                        q.append(neighbor)

        return distance_matrix
            
    def friendly_flood_fill(self, source, max_distance):
        # Returns a np.array((self.width, self.height)) that contains the distance to the target by traversing through friendly owned cells only.
        # q is a queue(list) of items (cell, distance)
        q = [source]
        distance_matrix = np.ones((self.width, self.height)) * -1
        distance_matrix[source.x, source.y] = 0

        while len(q) > 0:
            current = q.pop(0)
            current_distance = distance_matrix[current.x, current.y]
            for neighbor in current.neighbors:
                if distance_matrix[neighbor.x, neighbor.y] == -1 and neighbor.owner == self.my_id:
                    distance_matrix[neighbor.x, neighbor.y] = current_distance + 1
                    if current_distance < max_distance - 1:
                        q.append(neighbor)
        
        return distance_matrix
            
    def friendly_flood_fill_multiple_sources(self, sources, max_distance):
        # Returns a np.array((self.width, self.height)) that contains the distance to the target by traversing through friendly owned cells only.
        # q is a queue(list) of items (cell, distance). sources is a list that contains the source cells.
        q = sources
        distance_matrix = np.ones((self.width, self.height)) * -1
        for source in q:
            distance_matrix[source.x, source.y] = 0

        while len(q) > 0:
            current = q.pop(0)
            current_distance = distance_matrix[current.x, current.y]
            for neighbor in current.neighbors:
                if (distance_matrix[neighbor.x, neighbor.y] == -1 or distance_matrix[neighbor.x, neighbor.y] > (current_distance + 1)) and neighbor.owner == self.my_id:
                    distance_matrix[neighbor.x, neighbor.y] = current_distance + 1
                    if current_distance < max_distance - 1:
                        q.append(neighbor)
        
        return distance_matrix        

    def last_resort_strength_check(self):
        # Calculates the projected strength map and identifies squares that are violating it.
        # Ignore strength overloads due to production for now
        # Validate moves
        projected_strength_map = np.zeros((self.width, self.height))
        # We only care about our moves.
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.my_id:
                if square.move == -1 or square.move == STILL:
                    projected_strength_map[square.x, square.y] += square.strength # + square.production
                else:
                    dx, dy = get_offset(square.move)
                    projected_strength_map[(square.x + dx) % self.width, (square.y + dy) % self.height] += square.strength

        # Get a list of squares that are over the cap
        violation_indices = np.transpose(np.nonzero((projected_strength_map > 255 + strength_buffer)))
        violation_squares = [self.squares[c[0], c[1]] for c in violation_indices]
        violation_count = len(violation_squares)

        for square in violation_squares:
            if square.owner == self.my_id and (square.move == -1 or square.move == STILL):
                # We can try to move this square to an neighbor.
                possible_paths = []
                for d in range(0, 4):
                    # Move to the lowest strength neighbor. this might cause a collision but we'll resolve it with multiple iterations
                    n = square.neighbors[d]
                    if n.owner == self.my_id:
                        possible_paths.append((d, n, projected_strength_map[n.x, n.y]))
                    else:
                        # Do we want to attack a bordering cell? Let's try taking over a cell IF it's next to a combat zone cell.
                        for n_n in n.neighbors:
                            if self.enemy_strength_map[1, n_n.x, n_n.y] > 0 and square.strength > n.strength:
                                possible_paths.append((d, n, n.strength))
                        
                possible_paths.sort(key = lambda x: x[2])
                # Force a move there
                self.make_move(square, d, n)
            else:
                # We aren't the problem. one of the squares that's moving here is going to collide with us.
                # How do we resolve this?
                options_list = []
                for n in square.neighbors:
                    if n.owner == self.my_id:
                        options_list.append((n, projected_strength_map[n.x, n.y]))
                options_list.sort(key = lambda x: x[1])
                # Let's try having the smallest one stay still instead
                self.make_move(options_list[0][0], STILL, None)
        
        return violation_count
                    
        
        

    def update_stats(self):
        # Updates various stats used for tracking
        self.turns_left = self.max_turns - self.frame
        self.percent_owned = np.sum(self.is_owned_map) / (self.width * self.height)
    

#==============================================================================
# Square class                
#==============================================================================
class Square:
    def __init__(self, game, x, y, production):
        self.game = game
        self.x = x
        self.y = y
        self.production = production
        self.height = game.height
        self.width = game.width
        self.vertex = x * self.height + y
        self.target = None
        self.moving_here = []
        self.far_target = None
    
    def after_init_update(self):
        # Should only be called after all squares in game have been initialized.
        self.north = self.game.squares[(self.x + 0) % self.width, (self.y - 1) % self.height]
        self.east = self.game.squares[(self.x + 1) % self.width, (self.y + 0) % self.height]
        self.south = self.game.squares[(self.x + 0) % self.width, (self.y + 1) % self.height]
        self.west = self.game.squares[(self.x - 1) % self.width, (self.y + 0) % self.height]
        self.neighbors = [self.north, self.east, self.south, self.west] # doesn't include self

    def get_neighbors(self, n = 1, include_self = False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            if not include_self:
                return self.neighbors
                
        combos = ((dx, dy) for dy in range(-n, n+1) for dx in range(-n, n+1) if abs(dx) + abs(dy) <= n)
        return (self.game.squares[(self.x + dx) % self.width][(self.y + dy) % self.height] for dx, dy in combos if include_self or dx or dy)        
    
    def update(self, owner, strength):
        # updates the square with the new owner and strength. Also resets movement variables
        self.owner = owner
        self.strength = strength
        self.reset_move()

    def reset_move(self):
        # Resets the move information
        # Note, the target's moving_here is NOT reset so this should really only be used if all squares are being reset.
        self.move = -1
        self.target = None
        self.moving_here = []
        self.far_target = None
        
        
####################
# Helper Functions #
####################

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
    return np.roll(M, x, 0)

def roll_y(M, y):
    return np.roll(M, y, 1)

def roll_xy(M, x, y):
    return np.roll(np.roll(M, x, 0), y, 1)

def spread_n(M, n, decay = 0, include_self = True):
    # Takes a matrix M, and then creates an influence map by offsetting by N in every direction. 
    # Decay function is currently of the form exp(-decay * distance)
    if include_self == True:
        spread_map = np.copy(M)
    else:
        spread_map = np.zeros_like(M)
    distance = 1
    while distance <= n:
        combos = get_all_d_away(distance)
        decay_factor = math.exp(-decay * distance)
        for c in combos:
            spread_map += roll_xy(np.multiply(decay_factor, M), c[0], c[1])
        distance += 1
    return spread_map
    
def spread(M, decay = 0, include_self = True):
    # For now to save time, we'll use game_map.distance_map and assume that we'll always be using the same falloff distances to calculate offsets.
    
    # Takes the matrix M and then for each point (x, y), calculate the product of the distance map and the decay factor.
    decay_map = np.exp(np.multiply(game.distance_map, -decay))
    
    spread_map = np.sum(np.multiply(decay_map, M), (2, 3))
    return spread_map
        
def get_all_d_away(d):
    combos = []
    for x in range(0, d+1):
        x_vals = list(set([x, -x]))
        y_vals = list(set([d-x, -(d-x)]))
        combos.extend(list(itertools.product(x_vals, y_vals)))
    return list(set(combos))
        
def distance_from_owned(M, mine):
    # Returns the minimum distance to get to any point if already at all points in xys using 4D array M
    return np.apply_along_axis(np.min, 0, M[np.nonzero(mine)])  

def rebase_map(map_a, total = True):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.width * self.height
    # If Total = False, rebases to the # of non-zero squares 
    if total:
        size = functools.reduce(lambda x, y: x*y, map_a.shape)
    else:
        size = np.sum(map_a != 0)
    factor = size / np.sum(map_a)
    return np.multiply(map_a, factor)      

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
# Game Loop
#==============================================================================
def game_loop():
    game.get_frame()
    logging.debug("Frame: " + str(game.frame))
    
    game.update()
    
    game.get_moves()
    
    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()
    game.send_frame()
    
######################
# Game run-time code #
######################

logging.basicConfig(filename='logging.log',level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)
directions = [NORTH, EAST, SOUTH, WEST, STILL]

game = Game()


while True:
    game_loop()