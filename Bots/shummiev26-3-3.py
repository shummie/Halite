#==============================================================================
# Imports
#==============================================================================
import functools
import itertools
import logging
import math
import numpy as np
import random
import scipy.sparse
import sys
import time

#==============================================================================
# Variables
#==============================================================================
botname = "shummie v26.3-3"

buildup_multiplier = 6
strength_buffer = 0

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
        
    def update(self):
        self.update_maps()
        self.update_stats()

    def update_maps(self):
        self.update_calc_maps() 
        
        self.update_owner_maps()
        self.update_distance_maps()
        self.update_border_maps()
        
        self.update_recover_maps()
       
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
        self.distance_from_owned = distance_from_owned(self.distance_map_no_decay, self.is_owned_map)
        self.distance_from_owned[self.is_owned_map == 1] = 0
        
        self.distance_from_border = distance_from_owned(self.distance_map_no_decay, 1 - self.is_owned_map)
        self.distance_from_border[1 - (self.is_owned_map == 1)] = 0

        if self.starting_player_count > 1:  # Breaks in single player mode otherwise.
            self.distance_from_enemy = distance_from_owned(self.distance_map_no_decay, self.is_enemy_map)
            self.distance_from_enemy[self.is_enemy_map == 1] = 999
        else:
            self.distance_from_enemy = np.ones((self.width, self.height)) * 999

        
        
    def update_border_maps(self):
        self.border_map = np.zeros((self.width, self.height))
        self.inner_border_map = np.zeros((self.width, self.height))
        self.combat_zone_map = np.zeros((self.width, self.height))
        
        self.border_map = (self.distance_from_owned == 1) * 1
        self.border_indices = np.transpose(np.where(self.border_map == 1))

        self.inner_border_map = (self.distance_from_border == 1) * 1
        self.inner_border_indices = np.transpose(np.where(self.inner_border_map == 1))
        
        self.combat_zone_map = self.border_map * (self.strength_map == 0)
        
        if self.starting_player_count > 1 and np.sum(self.combat_zone_map) >= 1:  # Breaks in single player mode otherwise.
            self.distance_from_combat_zone = distance_from_owned(self.distance_map_no_decay, self.combat_zone_map)
            self.distance_from_combat_zone += (self.is_enemy_map + self.is_neutral_map - self.combat_zone_map) * 999
        else:
            self.distance_from_combat_zone = np.ones((self.width, self.height)) * 999
        
    def update_recover_maps(self):
        max_distance = self.width // 2
        self.recover_map = np.zeros((max_distance + 1, self.width, self.height))
        self.recover_map[0] = np.divide(self.strength_map, self.production_map_01) * (self.is_neutral_map - self.combat_zone_map)
        
        self.prod_over_str_map = np.zeros((max_distance + 1, self.width, self.height))
        #self.prod_over_str_map[0] = np.divide(self.production_map, self.strength_map_01) * (self.is_neutral_map - self.combat_zone_map)
        new_str_map = np.copy(self.strength_map)
        new_str_map[new_str_map == 0] = 2
        #self.prod_over_str_map[0] = np.divide(self.production_map, self.strength_map_01) * (self.is_neutral_map - self.combat_zone_map)
        self.prod_over_str_map[0] = np.divide(self.production_map, new_str_map) * (self.is_neutral_map - self.combat_zone_map)
        self.recover_map[0] = 1 / np.maximum(self.prod_over_str_map[0], 0.01)
        
        for distance in range(1, max_distance + 1):
            self.prod_over_str_map[distance] = spread_n(self.prod_over_str_map[distance - 1], 1)
            self.prod_over_str_map[distance][self.prod_over_str_map[distance-1] == 0] = 0
            self.prod_over_str_map[distance] = self.prod_over_str_map[distance] / 5
            self.recover_map[distance] = 1 / np.maximum(self.prod_over_str_map[distance], 0.01)

        self.prod_over_str_max_map = np.apply_along_axis(np.max, 0, self.prod_over_str_map)
        self.recover_max_map = 1 / np.maximum(self.prod_over_str_max_map, 0.01)
        self.prod_over_str_avg_map = np.apply_along_axis(np.mean, 0, self.prod_over_str_map)
        self.recover_avg_map = 1 / np.maximum(self.prod_over_str_avg_map, 0.01)
        self.prod_over_str_wtd_map = (self.prod_over_str_max_map + self.prod_over_str_avg_map) / 2
        self.recover_wtd_map = 1 / np.maximum(self.prod_over_str_wtd_map, 0.01)

    
    def get_moves(self):
        # This is the main logic controlling code.
        # Find super high production cells
        self.get_pre_combat_production()
        # 1 - Find combat zone cells and attack them.
        self.get_moves_attack()
        # 2 - Find production zone cells and attack them
        self.get_moves_production()
        # 3 - Move all other unassigned cells.
        self.get_moves_other()
        
    def get_pre_combat_production(self):
        # In the event we are trying to fight in a very high production zone, reroute some attacking power to expand in this area.
        pre_combat_threshold = 2
        potential_targets_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices if (self.recover_wtd_map[c[0], c[1]] < pre_combat_threshold)]
        if len(potential_targets) == 0: 
            return
            
        potential_targets.sort(key = lambda sq: self.recover_wtd_map[sq.x, sq.y])
        
        best_target_value = self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y]
        # anything with X of the best_value target should be considered. Let's set this to 4 right now.
        while len(potential_targets) > 0 and self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y] <= (best_target_value + 2):
            target = potential_targets.pop(0)
            logging.debug("Frame: " + str(self.frame) + " x/y: " + str(target.x) + "/" + str(target.y) + " : " + str(self.recover_wtd_map[target.x, target.y]))
            logging.debug("Frame: " + str(self.frame) + " x/y: " + str(target.x) + "/" + str(target.y) + " : prod " + str(self.prod_over_str_wtd_map[target.x, target.y]))
            self.attack_cell(target, 3)        
        
    def get_moves_attack(self):
        # Attempts to attack all border cells that are in combat
        potential_targets_indices = np.transpose(np.nonzero(self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]
        potential_targets.sort(key = lambda x: self.distance_from_enemy[x.x, x.y])

        for square in potential_targets:
            self.attack_cell(square, 1)
        
        # Get a list of all squares within 5 spaces of a combat zone.
        combat_radius = 5
        combat_squares_indices = np.transpose(np.nonzero((self.distance_from_combat_zone <= combat_radius) * (self.move_map == -1)))
        combat_squares = [self.squares[c[0], c[1]] for c in combat_squares_indices]
        
        for square in combat_squares:
            if (square.strength > square.production * buildup_multiplier) and ((square.x + square.y) % 2 == self.frame % 2):
                self.move_towards_map(square, self.distance_from_combat_zone)
            else:
                self.make_move(square, STILL)

    def move_towards_map(self, square, distance_map, through_friendly = True):
        current_distance = distance_map[square.x, square.y]
        for n in square.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if distance_map[n.x, n.y] < current_distance:
                    success = self.move_square_to_target(square, n, True)
                    if success:
                        break

        
    def get_moves_production(self):
        # Tries to find the best cells to attack from a production standpoint.
        # Does not try to attack cells that are in combat zones.
        potential_targets_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]
        if len(potential_targets) == 0: 
            return
            
        potential_targets.sort(key = lambda sq: self.recover_wtd_map[sq.x, sq.y])
        
        best_target_value = self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y]
        # anything with X of the best_value target should be considered. Let's set this to 4 right now.
        while len(potential_targets) > 0 and self.recover_wtd_map[potential_targets[0].x, potential_targets[0].y] <= (best_target_value + 8000):
            target = potential_targets.pop(0)
            self.attack_cell(target, 4)
        
    def get_moves_other(self):
        # Tries to move to 
        idle_squares_indices = np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))
        idle_squares = [self.squares[c[0], c[1]] for c in idle_squares_indices]

        if len(idle_squares) == 0:
            return
            
        for square in idle_squares:
            if square.strength > square.production * buildup_multiplier:
                # Move towards the closest border
                #if not self.inner_border_map[square.x, square.y]:
                    # For now, move to the square with the lowest recovery
                value_map = (self.recover_wtd_map + self.distance_map_no_decay[square.x, square.y] * 1.2) * self.border_map
                #best_target_value = (self.recover_wtd_map * (self.border_map - self.combat_zone_map)).argmin()
                #value_map = value_map * (1 - self.combat_zone_map)
                value_map[np.nonzero(self.combat_zone_map)] = 0
                value_map += self.distance_map_no_decay[square.x, square.y] * 0.8 * self.combat_zone_map
                #value_map[self.combat_zone_map == 1] = self.distance_map_no_decay[square.x, square.y] * .8
                value_map[value_map == 0] = 9999
                tx, ty = np.unravel_index(value_map.argmin(), (self.width, self.height))
                target = self.squares[tx, ty]
                if self.distance_between(square, target) > 10:
                    self.move_square_to_target_simple(square, target, True)
                elif self.distance_between(square, target) > 1:
                    self.move_square_to_target(square, target, True)
                
                    
    def distance_between(self, sq1, sq2):
        dx = abs(sq1.x - sq2.x)
        dy = abs(sq1.y - sq2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy
            
            
            
    def attack_cell(self, target, max_cells_out):
        # Attempts to coordinate attack to a specific cell.
        cells_out = 1
        
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
            
            if available_strength + available_production > target.strength:
                attacking_cells_indices = np.transpose(np.nonzero(target_distance_matrix > 0))
                attacking_cells = [self.squares[c[0], c[1]] for c in attacking_cells_indices]

                still_cells = []
                if cells_out > 1:
                    still_cells_indices = np.transpose(np.nonzero(target_distance_matrix_production> 0))
                    still_cells = [self.squares[c[0], c[1]] for c in still_cells_indices]
                moving_cells = list(set(attacking_cells) - set(still_cells))
                
                for square in still_cells:
                    self.make_move(square, STILL)
                    
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
    
    def make_move(self, square, direction):
        self.move_map[square.x, square.y] = direction

        if direction == -1: # Reset the square move
            if square.target != None:
                square.target.moving_here.remove(square)
                square.target = None
            square.move = -1
            return
        
        if square.move != -1:
            if square.target != None:
                square.target.moving_here.remove(square)
        
        square.move = direction
        if direction != STILL:
            square.target = square.neighbors[direction]
            square.target.moving_here.append(square)
            
    def move_square_to_target(self, source, destination, through_friendly):
        # Get the distance matrix that we will use to determine movement.
        
        distance_matrix = self.flood_fill_until_target(source, destination, through_friendly)
        source_distance = distance_matrix[source.x, source.y]
        if source_distance == -1:
            # We couldn't find a path to the destination
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
        future_strength = source.strength
        future_strength += path_choices[0][1].strength if (path_choices[0][1].owner == self.my_id and (path_choices[0][1].move == -1 or path_choices[0][1].move == STILL)) else 0
        if path_choices[0][1].moving_here != []:
            future_strength += sum(x.strength for x in path_choices[0][1].moving_here)
        
        if future_strength <= 255 + strength_buffer:
            # We're good, make the move
            self.make_move(source, path_choices[0][0])
            return True
            
        # Can we test the second move if it exists?
        if len(path_choices) > 1:
            future_strength = source.strength + path_choices[1][1].strength if (path_choices[1][1].owner == self.my_id and (path_choices[1][1].move == -1 or path_choices[1][1].move == STILL)) else 0
            if path_choices[1][1].moving_here != []:
                future_strength += sum(x.strength for x in path_choices[1][1].moving_here)
            
            if future_strength <= 255 + strength_buffer:
                # We're good, make the move
                self.make_move(source, path_choices[1][0])
                return True

        # Ok, so moving here will result in too much strength. What are our options?
        # Can we move the cell that we are moving to?
        if path_choices[0][1].owner == self.my_id and (path_choices[0][1].move == -1 or path_choices[0][1] == STILL):
            if source.strength + sum(x.strength for x in path_choices[0][1].moving_here) <= 255 + strength_buffer:
                # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                # This is dangerous, make sure to UNDO the fake move.
                self.make_move(source, path_choices[0][0])
                success = self.move_square_to_target(path_choices[0][1], destination, through_friendly)
                if success:
                    return True
                else:
                    # UNDO THE MOVE
                    self.make_move(source, -1)
                # Is there anywhere else we can move this cell?
                if path_choices[0][1].moving_here != []:
                    for secondary_target in path_choices[0][1].moving_here:
                        # Simulate the move
                        self.make_move(source, path_choices[0][0])
                        success = self.move_square_to_target(path_choices[0][1], secondary_target.target, through_friendly)
                        if success:
                            return True
                        self.make_move(source, -1)
                # Ok, can we just move the destination to a different square?
                neighbor_targets = []
                for n in path_choices[0][1].neighbors:
                    neighbor_strength = n.strength if n.owner == self.my_id else 0
                    neighbor_strength += sum(x.strength for x in n.moving_here)
                    neighbor_targets.append((n, neighbor_strength))
                # Try to move to the lowest strength target.
                neighbor_targets.sort(key = lambda x: x[1])
                # Attempt to move to the lowest strength neighbor
                for n_t in neighbor_targets:
                    if n_t[0].owner != self.my_id:
                        # We're attempting to attack a cell
                        if n_t[0].strength < path_choices[0][1].strength + sum(x.strength for x in n_t[0].moving_here):
                            if path_choices[0][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                self.make_move(source, path_choices[0][0])
                                self.move_square_to_target(path_choices[0][1], n_t[0], through_friendly)
                                return True
                    else:
                        future_n_strength = path_choices[0][1].strength
                        future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                        future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                        if future_n_strength <= 255 + strength_buffer:
                            self.make_move(source, path_choices[0][0])
                            self.move_square_to_target(path_choices[0][1], n_t[0], through_friendly)
                            return True
                        else:
                            break
        # Ok, the cell we are moving to isn't the problem. WE are. Let's try the secondary direction
        if len(path_choices) > 1:
            if path_choices[1][1].owner == self.my_id and (path_choices[1][1].move == -1 or path_choices[1][1] == STILL):
                if source.strength + sum(x.strength for x in path_choices[1][1].moving_here) <= 255 + strength_buffer:
                    # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                    # This is dangerous, make sure to UNDO the fake move.
                    self.make_move(source, path_choices[1][0])
                    success = self.move_square_to_target(path_choices[1][1], destination, through_friendly)
                    if success:
                        return True
                    else:
                        # UNDO THE MOVE
                        self.make_move(source, -1)
                    # Is there anywhere else we can move this cell?
                    if path_choices[1][1].moving_here != []:
                        for secondary_target in path_choices[0][1].moving_here:
                            # Simulate the move
                            self.make_move(source, path_choices[1][0])
                            success = self.move_square_to_target(path_choices[1][1], secondary_target.target, through_friendly)
                            if success:
                                return True
                            self.make_move(source, -1)
                    # Ok, can we just move the destination to a different square?
                    neighbor_targets = []
                    for n in path_choices[1][1].neighbors:
                        neighbor_strength = n.strength if n.owner == self.my_id else 0
                        neighbor_strength += sum(x.strength for x in n.moving_here)
                        neighbor_targets.append((n, neighbor_strength))
                    # Try to move to the lowest strength target.
                    neighbor_targets.sort(key = lambda x: x[1])
                    # Attempt to move to the lowest strength neighbor
                    for n_t in neighbor_targets:
                        if n_t[0].owner != self.my_id:
                            # We're attempting to attack a cell
                            if n_t[0].strength < path_choices[1][1].strength + sum(x.strength for x in n_t[0].moving_here):
                                if path_choices[1][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                    self.make_move(source, path_choices[1][0])
                                    self.move_square_to_target(path_choices[1][1], n_t[0], through_friendly)
                                    return True
                        else:
                            future_n_strength = path_choices[1][1].strength
                            future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                            future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                            if future_n_strength <= 255 + strength_buffer:
                                self.make_move(source, path_choices[1][0])
                                self.move_square_to_target(path_choices[1][1], n_t[0], through_friendly)
                                return True
                            else:
                                break
        # We can't do anything.
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
                
        future_strength = source.strength
        future_strength += path_choices[0][1].strength if (path_choices[0][1].owner == self.my_id and (path_choices[0][1].move == -1 or path_choices[0][1].move == STILL)) else 0
        if path_choices[0][1].moving_here != []:
            future_strength += sum(x.strength for x in path_choices[0][1].moving_here)
        
        if future_strength <= 255 + strength_buffer:
            # We're good, make the move
            self.make_move(source, path_choices[0][0])
            return True
            
        # Can we test the second move if it exists?
        if len(path_choices) > 1:
            future_strength = source.strength + path_choices[1][1].strength if (path_choices[1][1].owner == self.my_id and (path_choices[1][1].move == -1 or path_choices[1][1].move == STILL)) else 0
            if path_choices[1][1].moving_here != []:
                future_strength += sum(x.strength for x in path_choices[1][1].moving_here)
            
            if future_strength <= 255 + strength_buffer:
                # We're good, make the move
                self.make_move(source, path_choices[1][0])
                return True

        # Ok, so moving here will result in too much strength. What are our options?
        # Can we move the cell that we are moving to?
        if path_choices[0][1].owner == self.my_id and (path_choices[0][1].move == -1 or path_choices[0][1] == STILL):
            if source.strength + sum(x.strength for x in path_choices[0][1].moving_here) <= 255 + strength_buffer:
                # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                # This is dangerous, make sure to UNDO the fake move.
                self.make_move(source, path_choices[0][0])
                success = self.move_square_to_target_simple(path_choices[0][1], destination, False)
                if success:
                    return True
                else:
                    # UNDO THE MOVE
                    self.make_move(source, -1)
                # Is there anywhere else we can move this cell?
                if path_choices[0][1].moving_here != []:
                    for secondary_target in path_choices[0][1].moving_here:
                        # Simulate the move
                        self.make_move(source, path_choices[0][0])
                        success = self.move_square_to_target_simple(path_choices[0][1], secondary_target.target, False)
                        if success:
                            return True
                        self.make_move(source, -1)
                # Ok, can we just move the destination to a different square?
                neighbor_targets = []
                for n in path_choices[0][1].neighbors:
                    neighbor_strength = n.strength if n.owner == self.my_id else 0
                    neighbor_strength += sum(x.strength for x in n.moving_here)
                    neighbor_targets.append((n, neighbor_strength))
                # Try to move to the lowest strength target.
                neighbor_targets.sort(key = lambda x: x[1])
                # Attempt to move to the lowest strength neighbor
                for n_t in neighbor_targets:
                    if n_t[0].owner != self.my_id:
                        # We're attempting to attack a cell
                        if n_t[0].strength < path_choices[0][1].strength + sum(x.strength for x in n_t[0].moving_here):
                            if path_choices[0][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                self.make_move(source, path_choices[0][0])
                                self.move_square_to_target_simple(path_choices[0][1], n_t[0], False)
                                return True
                    else:
                        future_n_strength = path_choices[0][1].strength
                        future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                        future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                        if future_n_strength <= 255 + strength_buffer:
                            self.make_move(source, path_choices[0][0])
                            self.move_square_to_target_simple(path_choices[0][1], n_t[0], True)
                            return True
                        else:
                            break
        # Ok, the cell we are moving to isn't the problem. WE are. Let's try the secondary direction
        if len(path_choices) > 1:
            if path_choices[1][1].owner == self.my_id and (path_choices[1][1].move == -1 or path_choices[1][1] == STILL):
                if source.strength + sum(x.strength for x in path_choices[1][1].moving_here) <= 255 + strength_buffer:
                    # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                    # This is dangerous, make sure to UNDO the fake move.
                    self.make_move(source, path_choices[1][0])
                    success = self.move_square_to_target_simple(path_choices[1][1], destination, False)
                    if success:
                        return True
                    else:
                        # UNDO THE MOVE
                        self.make_move(source, -1)
                    # Is there anywhere else we can move this cell?
                    if path_choices[1][1].moving_here != []:
                        for secondary_target in path_choices[0][1].moving_here:
                            # Simulate the move
                            self.make_move(source, path_choices[1][0])
                            success = self.move_square_to_target_simple(path_choices[1][1], secondary_target.target, False)
                            if success:
                                return True
                            self.make_move(source, -1)
                    # Ok, can we just move the destination to a different square?
                    neighbor_targets = []
                    for n in path_choices[1][1].neighbors:
                        neighbor_strength = n.strength if n.owner == self.my_id else 0
                        neighbor_strength += sum(x.strength for x in n.moving_here)
                        neighbor_targets.append((n, neighbor_strength))
                    # Try to move to the lowest strength target.
                    neighbor_targets.sort(key = lambda x: x[1])
                    # Attempt to move to the lowest strength neighbor
                    for n_t in neighbor_targets:
                        if n_t[0].owner != self.my_id:
                            # We're attempting to attack a cell
                            if n_t[0].strength < path_choices[1][1].strength + sum(x.strength for x in n_t[0].moving_here):
                                if path_choices[1][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                    self.make_move(source, path_choices[1][0])
                                    self.move_square_to_target_simple(path_choices[1][1], n_t[0], False)
                                    return True
                        else:
                            future_n_strength = path_choices[1][1].strength
                            future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                            future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                            if future_n_strength <= 255 + strength_buffer:
                                self.make_move(source, path_choices[1][0])
                                self.move_square_to_target_simple(path_choices[1][1], n_t[0], True)
                                return True
                            else:
                                break
        # We can't do anything.
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
            
        


        

    def update_stats(self):
        # Updates various stats used for tracking
        self.turns_left = self.max_turns - self.frame
    
    

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
    
    def after_init_update(self):
        # Should only be called after all squares in game have been initialized.
        self.north = self.game.squares[(self.x + 0) % self.width, (self.y - 1) % self.height]
        self.east = self.game.squares[(self.x + 1) % self.width, (self.y + 0) % self.height]
        self.south = self.game.squares[(self.x + 0) % self.width, (self.y + 1) % self.height]
        self.west = self.game.squares[(self.x - 1) % self.width, (self.y + 0) % self.height]
        self.neighbors = [self.north, self.east, self.south, self.west] # We might want to remove self...

    def get_neighbors(self, n = 1, include_self = False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            if include_self:
                return self.neighbors # broken.
            else:
                return self.neighbors[0:4]            
        else:
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

def rebase_map(map_a):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.width * self.height
    size = functools.reduce(lambda x, y: x*y, map_a.shape)
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
    game.update()
    game.get_moves()
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