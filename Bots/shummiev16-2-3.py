# TODO LIST:
# -- Dijkstra's Algorithm for pathfinding?
# -- How to identify which direction to focus growth? Looking at production map at beginning to see.
# -- Attack patterns? What's the best strategy for attacking / retreating / reinforcing?
# -- Very early game production focus strategy

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


#############
# Variables #
#############

botname = "shummie v16.2-2"

production_decay = 0.7
production_influence_max_distance = 12
buildup_multiplier = 6
early_game_buildup_multiplier = 6
early_game_value_threshold = 0.80
strength_buffer = 0
border_distance_decay_factor = 1.3
border_target_percentile = .3

production_self_factor = 0
production_neutral_factor = 1
production_enemy_factor = 1
production_influence_factor = 4 # Sample values are around 50-80
production_square_influence_factor = 30
prod_over_str_influence_factor = 10 # Sample values are around 0.5 - 1.0
prod_over_str_self_factor = -2
prod_over_str_neutral_factor = 1.25
prod_over_str_enemy_factor = 3
enemy_strength_0_influence_factor = 12
enemy_strength_1_influence_factor = 8 # Sample values around 4-20
enemy_strength_2_influence_factor = 4 # Sample values around 2-40
enemy_strength_3_influence_factor = 1 # Sample values around 2-50
enemy_territory_1 = 50
enemy_territory_2 = 25
enemy_territory_3 = 10

late_game_buildup_multiplier = 7
late_game_production_self_factor = -1
late_game_production_neutral_factor = 2
late_game_production_enemy_factor = 5
late_game_production_influence_factor = 3 # Sample values are around 50-80
late_game_production_square_influence_factor = 5
late_game_prod_over_str_influence_factor = 10 # Sample values are around 0.5 - 1.0
late_game_prod_over_str_self_factor = -2
late_game_prod_over_str_neutral_factor = 2
late_game_prod_over_str_enemy_factor = 3
late_game_enemy_strength_0_influence_factor = 5
late_game_enemy_strength_1_influence_factor = 1 # Sample values around 4-20
late_game_enemy_strength_2_influence_factor = 2 # Sample values around 2-40
late_game_enemy_strength_3_influence_factor = -2 # Sample values around 2-50
late_game_enemy_territory_1 = 20
late_game_enemy_territory_2 = 10
late_game_enemy_territory_3 = -4

        
#################
# GameMap Class #
#################

# There is a TON of information stored in here. I may have to break it out into additional classes at some point:
#
# self.production_map[width, height]: The raw production found in cell width, height
# self.strength_map[width, height]: The strength located in cell width, height
# self.owner_map[width, height]: The owner located in cell width, height
# self.distance_map[x, y, i, j]: Stores the distance from (x, y) to (i, j) with falloff as a distance modifier. Note that (x, y) to (x, y) = 1. 
#                                Useful for multiplying influence maps and modifying based on distance.
# self.is_owner_map[id, x, y]: 1 if cell x, y belongs to player id. 0 otherwise.
# self.border_map[x, y]: 1 if the cell's owner == 0 AND there is a friendly neighboring cell. 0 otherwise
# self.is_enemy_map[x, y]: 1 if the cell's owner is NOT 0 AND NOT self.my_id.
# self.enemy_border_map[x, y]: 1 if the cell is NEXT to an enemy cell. The cell itself may be neutral, friendly, or enemy.



class GameMap:
    def __init__(self):
        self.initialize_game()

    def __iter__(self):
        # Allows direct iteration over all squares
        return itertools.chain.from_iterable(self.squares)
    
    def initialize_game(self):
        # This should only be called once, and at the beginning of the game
        self.my_id = int(get_string())
        map_size_string = get_string()
        production_map_string = get_string()
        self.phase = 0 # 0 = early, 1 = mid, 2 = late
        
        self.width, self.height = tuple(map(int, map_size_string.split()))
        self.frame = -1
        self.max_turns = 10 * ((self.width * self.height) ** 0.5)
                
        self.production_map = numpy.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()

        self.get_frame()
        
        self.starting_player_count = numpy.amax(self.owner_map) # Note, for range you'd need to increase the range by 1
        
        # Create the distance map
        self.distance_map = self.create_distance_map()        
        self.distance_map_no_decay = self.create_distance_map(1)
        
        self.get_configs()
        # Send the botname
        send_string(botname)
        
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
    
        self.owner_map = numpy.array(owners).reshape((self.height, self.width)).transpose()
        
        # This is then followed by WIDTH * HEIGHT integers, representing the strength values of the tiles in the map. 
        # It fills in the map in the same way owner values fill in the map.        
        assert len(split_string) == self.width * self.height
        str_list = list(map(int, split_string))
        
        self.strength_map = numpy.array(str_list).reshape((self.height, self.width)).transpose()
        
        # Create all squares for the GameMap
        self.squares = numpy.empty((self.width, self.height), dtype = numpy.object)
        #self.squares = [[None for y in range(self.height)] for x in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y] = Square(self, x, y, self.owner_map[x, y], self.strength_map[x, y], self.production_map[x, y])
        
        # Reset the move_map
        self.move_map = numpy.ones((self.width, self.height)) * -1  # Could possibly expand this in the future to consider enemy moves...
        
        self.frame += 1
    
    def get_configs(self):
        # To be expanded. A bit of a hack for now. (Who am i kidding, this will probably stay this way forever.
        early_game_squares_out_search_array = numpy.zeros((7, 5))
        # early_game_squares_out_search_array[map_size, player_count]
        # map_size: 20, 25, 30, 35, 40, 45, 50 = self.width / 5 - 4
        # player_count: 2, 3, 4, 5, 6 = self.starting_player_count - 2
        early_game_squares_out_search_array[0, 0] = 4
        early_game_squares_out_search_array[1, 0] = 5
        early_game_squares_out_search_array[2, 0] = 6
        early_game_squares_out_search_array[3, 0] = 6
        early_game_squares_out_search_array[4, 0] = 7
        early_game_squares_out_search_array[5, 0] = 8
        early_game_squares_out_search_array[6, 0] = 9
        early_game_squares_out_search_array[0, 1] = 4
        early_game_squares_out_search_array[1, 1] = 5
        early_game_squares_out_search_array[2, 1] = 6
        early_game_squares_out_search_array[3, 1] = 6
        early_game_squares_out_search_array[4, 1] = 7
        early_game_squares_out_search_array[5, 1] = 7
        early_game_squares_out_search_array[6, 1] = 8
        early_game_squares_out_search_array[0, 2] = 4
        early_game_squares_out_search_array[1, 2] = 5
        early_game_squares_out_search_array[2, 2] = 5
        early_game_squares_out_search_array[3, 2] = 5
        early_game_squares_out_search_array[4, 2] = 6
        early_game_squares_out_search_array[5, 2] = 7
        early_game_squares_out_search_array[6, 2] = 8
        early_game_squares_out_search_array[0, 3] = 3
        early_game_squares_out_search_array[1, 3] = 4
        early_game_squares_out_search_array[2, 3] = 4
        early_game_squares_out_search_array[3, 3] = 5
        early_game_squares_out_search_array[4, 3] = 6
        early_game_squares_out_search_array[5, 3] = 6
        early_game_squares_out_search_array[6, 3] = 7
        early_game_squares_out_search_array[0, 4] = 3
        early_game_squares_out_search_array[1, 4] = 4
        early_game_squares_out_search_array[2, 4] = 4
        early_game_squares_out_search_array[3, 4] = 5
        early_game_squares_out_search_array[4, 4] = 5
        early_game_squares_out_search_array[5, 4] = 6
        early_game_squares_out_search_array[6, 4] = 7
        
        map_size_index = min(6, max(0, int(self.width / 5) - 4))
        player_index = min(4, max(0, self.starting_player_count))
        self.early_game_squares_out_search = early_game_squares_out_search_array[map_size_index, player_index]
    
    def update(self):
        self.turns_left = self.max_turns - self.frame
        
        self.update_maps()
                
        if self.phase == 0 and numpy.sum(self.is_owned_map) > (10*(self.width * self.height)**.5) / ((self.starting_player_count**0.5) * 9):
            self.phase = 1
        if self.phase == 1 and numpy.sum(self.is_owned_map) < (self.width * self.height * 0.4):
            self.phase = 2
        


    def update_maps(self):
        
        # Create is_owner maps
        self.update_owner_maps()
        self.update_border_maps()

        # Create the list of border squares
        self.create_influence_production_map()
        self.update_influence_enemy_strength_maps()
        self.create_influence_prod_over_str_map()
        
        self.create_heuristic_map()
        self.update_recover_map()
        
        self.update_distance_maps()
    
    def update_owner_maps(self):
        # Creates a 3-d owner map from self.owner_map
        self.is_owner_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
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
        self.border_map = numpy.zeros((self.width, self.height))
        
        # Roll the territories we own around by 1 square in all directions
        self.border_map = spread_n(self.is_owner_map[self.my_id], 1)
        self.border_map = numpy.minimum(self.border_map, 1)
        
        # Take out our border
        # 1's means the cells that are bordering but not in our territory
        self.border_map -= self.is_owner_map[self.my_id]
        
        # Create the enemy border map
        # For now, we won't distinguish between enemies...
        self.enemy_border_map = numpy.zeros((self.width, self.height))
        
        # Do the same as we did for the border map
        self.enemy_border_map = spread_n(self.is_enemy_map * 1.0, 1)
        
        self.enemy_border_map = numpy.minimum(self.enemy_border_map, 1)
        self.enemy_border_map -= self.is_enemy_map
        
        self.inner_border_map = spread_n(self.border_map * 1.0, 1)
        self.inner_border_map = numpy.multiply(self.inner_border_map, self.is_owned_map) 
        
        self.combat_zone_map = numpy.multiply(self.border_map, numpy.multiply(self.is_neutral_map, self.strength_map == 0))

    def update_distance_maps(self):
        self.distance_from_owned = distance_from_owned(self.distance_map_no_decay, self.is_owned_map)
        self.distance_from_owned[numpy.nonzero(self.is_owned_map)] = 0 # Any territory we own has a distance of 0.
        
        self.distance_from_border = distance_from_owned(self.distance_map_no_decay, 1 - self.is_owned_map)
        self.distance_from_border[numpy.nonzero(1 - self.is_owned_map)] = 0
        
        self.distance_from_enemy = distance_from_owned(self.distance_map_no_decay, self.is_enemy_map)                          
        self.distance_from_enemy[numpy.nonzero(self.is_enemy_map)] = 0

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
        
    def rebase_map(self, map_a):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.width * self.height
        factor = (self.width * self.height) / numpy.sum(map_a)
        return numpy.multiply(map_a, factor)

    def create_influence_production_map(self):
        # Lots of tweaking to do...
        # Start with a basic production map     
        
        self.influence_production_map = numpy.zeros((self.width, self.height))
        
        # Take the base production map and alter it based on who controls it
        #modified_production_map = numpy.multiply(self.production_map, self.is_owner_map[self.my_id]) * production_self_factor + numpy.multiply(self.production_map, self.is_owner_map[0]) * production_neutral_factor + numpy.multiply(self.production_map, self.is_enemy_map) * production_enemy_factor

        self_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_owner_map[self.my_id]), production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_owner_map[0]), production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_enemy_map), production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
        
        
        self_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_owner_map[self.my_id]), late_game_production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_owner_map[0]), late_game_production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.production_map, self.is_enemy_map), late_game_production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.late_game_influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
        
        # Zero out areas we own 
        # Do we want to do this?
        #self.influence_production_map -= numpy.multiply(self.influence_production_map, self.is_owner_map[self.my_id])
        
    def update_influence_enemy_strength_maps(self):
        # Creates a list of the enemy strength projections.
        # Get all enemy strengths:
        enemy_strength_map = numpy.multiply(self.strength_map, self.is_enemy_map) * 1.0
        # It might be better to actually have 1 matrix referenced by [distance, x, y], but let's keep it this way for now.
        self.influence_enemy_strength_map = numpy.zeros((10, self.width, self.height))
        self.influence_enemy_territory_map = numpy.zeros((10, self.width, self.height))

        for x in range(10):
            # Note, we create a lot of these not necessarily because they're useful, but we can use it to see how far away we are from enemy territory.
            self.influence_enemy_strength_map[x] = spread_n(enemy_strength_map, x)
            self.influence_enemy_territory_map[x] = numpy.minimum(self.influence_enemy_strength_map[x], 1)
        
    def create_influence_prod_over_str_map(self):
        
        # Creates an influence map based off of production / strength. Very similar to the influence_production_map
        self.influence_prod_over_str_map = numpy.zeros((self.width, self.height))
        
        # Calculate the production / str maps.
        prod_str_map = numpy.divide(self.production_map, numpy.maximum(1, self.strength_map))
        scaled_prod_str_map = numpy.multiply(prod_str_map, self.is_owner_map[self.my_id]) * prod_over_str_self_factor + numpy.multiply(prod_str_map, self.is_owner_map[0]) * prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.is_enemy_map) * prod_over_str_enemy_factor
        late_game_scaled_prod_str_map = numpy.multiply(prod_str_map, self.is_owner_map[self.my_id]) * late_game_prod_over_str_self_factor + numpy.multiply(prod_str_map, self.is_owner_map[0]) * late_game_prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.is_enemy_map) * late_game_prod_over_str_enemy_factor
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_prod_over_str_map = spread_n(scaled_prod_str_map, production_decay, production_influence_max_distance)
        self.late_game_influence_prod_over_str_map = spread_n(late_game_scaled_prod_str_map, production_decay, production_influence_max_distance)
    
        self.influence_prod_over_str_map -= numpy.multiply(self.influence_prod_over_str_map, self.is_owner_map[self.my_id])
        
    def create_heuristic_map(self):
        self.heuristic_map = numpy.zeros((self.width, self.height))
        
        if self.phase <= 1:
        
            #self.heuristic_map += production_square_influence_factor * cell.production / max(cell.strength, 1)
            #cell_value += game_map.influence_production_map[cell.x, cell.y] * production_influence_factor
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_owner_map[0]) * prod_over_str_neutral_factor
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_owner_map[self.my_id]) * prod_over_str_self_factor
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_enemy_map) * prod_over_str_enemy_factor
            
            self.heuristic_map += numpy.multiply(self.production_map, self.is_owner_map[0]) * production_neutral_factor
            self.heuristic_map += numpy.multiply(self.production_map, self.is_owner_map[self.my_id]) * production_self_factor
            self.heuristic_map += numpy.multiply(self.production_map, self.is_enemy_map) * production_enemy_factor
        
            self.heuristic_map += numpy.multiply(self.strength_map, self.is_enemy_map) * enemy_strength_0_influence_factor
            self.heuristic_map += game_map.influence_enemy_strength_map[1] * enemy_strength_1_influence_factor
            self.heuristic_map += game_map.influence_enemy_strength_map[2] * enemy_strength_2_influence_factor
            self.heuristic_map += game_map.influence_enemy_strength_map[3] * enemy_strength_3_influence_factor
    
            self.heuristic_map += game_map.influence_enemy_territory_map[1] * enemy_territory_1
            self.heuristic_map += game_map.influence_enemy_territory_map[2] * enemy_territory_2
            self.heuristic_map += game_map.influence_enemy_territory_map[3] * enemy_territory_3
                
        else:
           
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_owner_map[0]) * late_game_prod_over_str_neutral_factor
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_owner_map[self.my_id]) * late_game_prod_over_str_self_factor
            self.heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), self.is_enemy_map) * late_game_prod_over_str_enemy_factor
            
            self.heuristic_map += numpy.multiply(self.production_map, self.is_owner_map[0]) * late_game_production_neutral_factor
            self.heuristic_map += numpy.multiply(self.production_map, self.is_owner_map[self.my_id]) * late_game_production_self_factor
            self.heuristic_map += numpy.multiply(self.production_map, self.is_enemy_map) * late_game_production_enemy_factor
            
            self.heuristic_map += numpy.multiply(numpy.multiply(self.strength_map, self.is_enemy_map), late_game_enemy_strength_0_influence_factor)
            self.heuristic_map += numpy.multiply(self.influence_enemy_strength_map[1], late_game_enemy_strength_1_influence_factor)
            self.heuristic_map += numpy.multiply(self.influence_enemy_strength_map[2], late_game_enemy_strength_2_influence_factor)
            self.heuristic_map += numpy.multiply(self.influence_enemy_strength_map[3], late_game_enemy_strength_3_influence_factor)
            self.heuristic_map += numpy.multiply(self.influence_enemy_territory_map[1], late_game_enemy_territory_1)
            self.heuristic_map += numpy.multiply(self.influence_enemy_territory_map[2], late_game_enemy_territory_2)
            self.heuristic_map += numpy.multiply(self.influence_enemy_territory_map[3], late_game_enemy_territory_3)  

    def update_recover_map(self):
        max_distance = 20
        self.recover_map = numpy.zeros((max_distance + 1, self.width, self.height))
        self.recover_map[0] = numpy.divide(self.strength_map, numpy.maximum(self.production_map, 0.01))
        self.recover_map[0] = numpy.multiply(self.recover_map[0], self.is_neutral_map)
        
        self.recover_map_enemy_smooth = numpy.zeros((max_distance + 1, self.width, self.height))
        # Smooth out enemy strength to reduce volatility in heuristic. If we are losing, will this devalue those cells too much when we should be attacking them instead?
        enemy_total_strength = numpy.sum(numpy.multiply(self.strength_map, self.is_enemy_map))
        enemy_total_squares = numpy.sum(self.is_enemy_map)
        enemy_average_strength = enemy_total_strength / enemy_total_squares
        
        self.recover_map_enemy_smooth[0] = numpy.divide(numpy.multiply(self.is_enemy_map, enemy_average_strength), numpy.maximum(self.production_map, 0.1))
        self.recover_map_enemy_smooth[0] += self.recover_map[0]
        
        self.recover_map[0] += (self.is_owned_map + self.is_enemy_map) * 999
        
        self.recover_map_spread = numpy.zeros((max_distance+1, self.width, self.height))
        
        for distance in range(1, max_distance + 1):
            dir_map = numpy.zeros((4, self.width, self.height))            
            dir_map[0] = roll_xy(self.recover_map[distance - 1], 0, 1)
            dir_map[1] = roll_xy(self.recover_map[distance - 1], 0, -1)
            dir_map[2] = roll_xy(self.recover_map[distance - 1], 1, 0)
            dir_map[3] = roll_xy(self.recover_map[distance - 1], -1, 0)
            
            self.recover_map[distance] = numpy.add(self.recover_map[distance - 1], numpy.amin(dir_map, 0))
            
            dir_map_smooth = numpy.zeros((4, self.width, self.height))
            dir_map_smooth[0] = roll_xy(self.recover_map_enemy_smooth[distance - 1], 0, 1)
            dir_map_smooth[1] = roll_xy(self.recover_map_enemy_smooth[distance - 1], 0, -1)
            dir_map_smooth[2] = roll_xy(self.recover_map_enemy_smooth[distance - 1], 1, 0)
            dir_map_smooth[3] = roll_xy(self.recover_map_enemy_smooth[distance - 1], -1, 0)
            
            self.recover_map_enemy_smooth[distance] = numpy.add(self.recover_map_enemy_smooth[distance - 1], numpy.amin(dir_map_smooth, 0))
        
        for d in range(2, max_distance + 1):
            self.recover_map[d] = self.recover_map[d] / d
            self.recover_map_enemy_smooth[d] = self.recover_map_enemy_smooth[d] / d      
            self.recover_map_spread = spread_n(self.recover_map[d], self.width // 5)  

        
    def get_distance(self, sq1, sq2):
        dx = abs(sq1.x - sq2.x)
        dy = abs(sq1.y - sq2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy
        
    def get_target(self, square, direction):
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
        return self.squares[(square.x + dx) % self.width][(square.y + dy) % self.height]

    def get_coord(self, sourcex, sourcey, dx, dy):
        return ((sourcex + dx) % self.width, (sourcey + dy) % self.height)
    
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
            if square.target != None:
                square.target.moving_here.remove(square)
            
        square.move = direction
        if direction != STILL:
            square.target = self.get_target(square, direction)
            square.target.moving_here.append(square)
    
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

    def get_best_moves(self):
        # Instead of each cell acting independently, look at the board as a whole and make squares move based on that.

        # Squares should always be moving towards a border. so get the list of border candidate squares
        all_targets = []
        production_squares = []
        for square in itertools.chain.from_iterable(self.squares):
            if self.border_map[square.x, square.y]:
                if self.influence_enemy_territory_map[3, square.x, square.y] == 0:
                    production_squares.append((square, self.recover_map[game_map.early_game_squares_out_search, square.x, square.y]))
                else: 
                    if square.owner == 0 and square.production == 1 and square.strength > 4:
                        continue
                    all_targets.append((square, heuristic(square)))
                
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        production_squares.sort(key = lambda x: x[1])
        all_targets.sort(key = lambda x: x[1], reverse = True)
        best_targets = all_targets[0:int(len(all_targets) * border_target_percentile)]

        if len(production_squares) > 0:
            threshold = production_squares[0][1] / early_game_value_threshold
        for border in production_squares:
            find_cell = False
            if border[1] <= threshold:
                find_cell = self.attack_cell(border[0], 4)
            if find_cell: 
                production_squares.remove(border)
        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.
        cells_out = 3
        for target in best_targets:
            success_attack = self.attack_cell(target[0], cells_out)
            if success_attack:
                best_targets.remove(target)

        # Now, there are some cells that haven't moved yet, but we might not want to move all of them. 
        cells_to_consider_moving = []
        for square in itertools.chain.from_iterable(self.squares):
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == self.my_id and (square.move == -1 or square.move == STILL):
                cells_to_consider_moving.append(square)

        # Simple logic for now:
        for square in cells_to_consider_moving:
            if square.is_border() == True:
                # Can we attack a bordering cell?
                targets = [n for n in square.neighbors() if (n.owner != self.my_id and n.strength < square.strength)]
                if len(targets) > 0:
                    targets.sort(key = lambda x: heuristic(x), reverse = True)
                    if heuristic(targets[0]) <= 0 or self.recover_map[5, targets[0].x, targets[0].y] > threshold:
                        target_map = numpy.multiply(self.recover_map_spread[5] + self.distance_map[square.x, square.y] * 2, self.border_map)
                        target_map += (self.is_owned_map + self.is_enemy_map) * 999
                        tx, ty = numpy.unravel_index(target_map.argmin(), (self.width, self.height))
                        square.move_to_target(self.squares[tx, ty], True)

                    else:
                        square.move_to_target(targets[0], False)
            elif square.strength > (square.production * buildup_multiplier):
                #self.go_to_border(square)
                #self.find_nearest_enemy_direction(square)
                #self.find_nearest_non_owned_border(square)
                #self.go_to_border(square)
                target_map = numpy.multiply(self.recover_map_spread + self.distance_map[square.x, square.y] * 2, self.border_map)
                target_map += (self.is_owned_map + self.is_enemy_map) * 999
                tx, ty = numpy.unravel_index(target_map.argmin(), (self.width, self.height))
                square.move_to_target(self.squares[tx, ty], True)
        
        # Any cells which are not moving now don't have a reason to move and can be used to prevent collisions.
       
    def attack_cell(self, target, max_cells_out = 1):
        # Will only attack the cell if sufficient strength
        # Otherwise, will attempt to move cells by cells_out so that it can gather enough strength.
        # Returns True if we have successfully found something to attack this
        # Returns False otherwise.
        
        # Only need to look at surrounding cells
        cells_out = 1
        while cells_out <= max_cells_out:
            if cells_out > 1 and target.owner != 0:
                return False
            
            available_squares = (self.move_map == -1) * 1
            distance_matrix = self.friendly_flood_fill(target, cells_out)
            distance_matrix[distance_matrix == -1] = 0
            
            #available_strength = numpy.sum(numpy.multiply(numpy.multiply(numpy.multiply(self.is_owner_map[self.my_id], self.strength_map), numpy.minimum(distance_matrix, 1)), available_squares))            
            available_strength = numpy.sum(numpy.multiply(numpy.multiply(self.strength_map, numpy.minimum(distance_matrix, 1)), available_squares))
            
            #logging.debug("avail str: " + str(available_strength))
            # Consider production if all cells stay still.
            distance_matrix = cells_out - distance_matrix
            distance_matrix[distance_matrix == cells_out] = 0
            available_production = numpy.sum(numpy.multiply(numpy.multiply(self.production_map, distance_matrix), available_squares))
            #logging.debug("avail prod:" + str(available_production))
    
            if available_strength + available_production > target.strength:
                # We have sufficient strength! Let's attack.
                # Get a list of all friendly neighbors
                attacking_cells = [x for x in target.neighbors(cells_out) if x.owner == self.my_id and x.move == -1]
                still_cells = []
                if cells_out > 1:
                    still_cells = [x for x in target.neighbors(cells_out - 1) if x.owner == self.my_id and x.move == -1]
                moving_cells = list(set(attacking_cells) - set(still_cells))
                
                # Ok, since we are doing this iteratively, we know that all cells in still_cells must stay still, otherwise an earlier cells_out would have worked
                for square in still_cells:
                    self.make_move(square, STILL)
                
                # How much remaining strength do we need?
                still_strength = numpy.sum(numpy.multiply(numpy.multiply(self.strength_map, numpy.minimum(distance_matrix, 1)), available_squares)) # Note this is the new distance map used for available_production
                needed_strength_from_movers = target.strength - available_production - still_strength
                
                if needed_strength_from_movers > 0:
                    # We don't necessarily want the highest strength piece to capture this. But, if we start with the smallest, we might be wasting moves/production.
                    # See if we need more than 1 piece to capture.
                    moving_cells.sort(key = lambda x: x.strength, reverse = True)
                    for square in moving_cells:
                        if cells_out == 1:
                            square.move_to_target(target, False)
                        else:
                            square.move_to_target(target, True)
                        needed_strength_from_movers -= square.strength
                        if needed_strength_from_movers < 0:
                            break
                        
                # Yay we're done.                        
                return True
            else:
                cells_out += 1
        return False

    def find_nearest_non_owned_border(self, square):
                
        current_distance = self.distance_from_border[square.x, square.y]
        for n in square.neighbors():
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_border[n.x, n.y] < current_distance:
                    success = square.move_to_target(n, True)
                    if success:
                        break            

    def find_nearest_non_npc_enemy_direction(self, square):
        dir_distance = []
        not_dir = []
        max_distance = max(game_map.width, game_map.height) / 2
        for d in (NORTH, EAST, SOUTH, WEST):
            distance = 0
            location = game_map.get_target(square, d)
            
            while (location.owner == self.my_id or (location.owner == 0 and location.strength == 0)) and distance < max_distance:
                distance += 1
                location = game_map.get_target(location, d)
            
            if location.owner == self.my_id:
                not_dir.append((d, distance, location))
            elif location.owner == 0 and location.strength > 0:
                not_dir.append((d, distance, location))
            else:
                dir_distance.append((d, distance, location))
            
        dir_distance.sort(key = lambda x: x[1])
        not_dir.sort(key = lambda x: x[1])
        
        success = False
        index = 0
        while not success and index < len(dir_distance):
            success = square.move_to_target(dir_distance[index][2], False)
            index += 1
        if not success:
            if len(not_dir) > 0:
                success = square.move_to_target(not_dir[0][2], False)                    
                    
    def find_nearest_enemy_border(self, square):
        current_distance = self.distance_from_enemy[square.x, square.y]
        for n in square.neighbors():
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_enemy[n.x, n.y] < current_distance:
                    success = square.move_to_target(n, True)
                    if success:
                        break
                    
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


    def get_best_moves_late_game(self):
        # Instead of each cell acting independently, look at the board as a whole and make squares move based on that.
        # Squares should always be moving towards a border. so get the list of border candidate squares
        
        all_targets = []
        for square in itertools.chain.from_iterable(self.squares):
            if self.border_map[square.x, square.y]:
                all_targets.append((square, self.heuristic_map[square.x, square.y]))
                
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        all_targets.sort(key = lambda x: x[1], reverse = True)
        best_targets = all_targets[0:int(len(all_targets) * border_target_percentile)]

        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.
        cells_out = 3
        for target in best_targets:
            success_attack = self.attack_cell(target[0], cells_out)
            if success_attack:
                best_targets.remove(target)

        
        # Now, there are some cells that haven't moved yet, but we might not want to move all of them. 
        cells_to_consider_moving = []
        for square in itertools.chain.from_iterable(self.squares):
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == self.my_id and (square.move == -1 or square.move == STILL):
                cells_to_consider_moving.append(square)

        # Simple logic for now:
        for square in cells_to_consider_moving:
            if square.is_border() == True:
                # Can we attack a bordering cell?
                targets = [n for n in square.neighbors() if (n.owner != self.my_id and n.strength < square.strength)]
                if len(targets) > 0:
                    targets.sort(key = lambda x: heuristic(x), reverse = True)
                    square.move_to_target(targets[0], False)
            elif square.strength > (square.production * buildup_multiplier):
                #self.go_to_border(square)
                self.find_nearest_non_npc_enemy_direction(square)
                #self.find_nearest_enemy_border(square)
                #self.go_to_border(square)
        
        # Any cells which are not moving now don't have a reason to move and can be used to prevent collisions.        
        
################
# Square class #        
################

class Square:
    def __init__(self, game_map, x, y, owner, strength, production):
        self.game_map = game_map
        self.x = x
        self.y = y
        self.owner = owner
        self.strength = strength
        self.production = production
        self.move = -1
        self.target = None
        self.moving_here = []
        self._is_border = None
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
                self._neighbors_1 = [self.game_map.squares[(self.x + dx) % self.game_map.width][(self.y + dy) % self.game_map.height] for dx, dy in combos if include_self or dx or dy]
                return self._neighbors_1
        else:
            combos = ((dx, dy) for dy in range(-n, n+1) for dx in range(-n, n+1) if abs(dx) + abs(dy) <= n)
        return (self.game_map.squares[(self.x + dx) % self.game_map.width][(self.y + dy) % self.game_map.height] for dx, dy in combos if include_self or dx or dy)
                
    def is_border(self):
        # looks at a square and sees if it's a border.
        # Looks at all neighbors and see if the owner != my_id
        # Have we done this calculation already? It shouldn't change within a frame
        # Is_border means that the square is owned by is AND there is a non-owned square next to it
        if self._is_border == None:
            if self.owner != self.game_map.my_id:
                self._is_border = False
            else:
                for n in self.neighbors():
                    if n.owner != self.game_map.my_id:
                        self._is_border = True
                        return True
                self._is_border = False
        return self._is_border
                
    def move_to_target(self, target, through_friendly):

        # Attempts to move to the designated target
        # Does so naively. Perhaps replace this with an A* or Dijkstra's algorithm at some point
    
        # Calculate cardinal direction distance to target.
        dist_w = (self.x - target.x) % self.game_map.width
        dist_e = (target.x - self.x) % self.game_map.width
        dist_n = (self.y - target.y) % self.game_map.height
        dist_s = (target.y - self.y) % self.game_map.height        
        
        if dist_w == 0 and dist_n == 0:
            return False

        w_neighbor = self.game_map.get_target(self, WEST)            
        e_neighbor = self.game_map.get_target(self, EAST)
        n_neighbor = self.game_map.get_target(self, NORTH)
        s_neighbor = self.game_map.get_target(self, SOUTH)
        
        ew_swap = False
        ns_swap = False
        
        if dist_w < dist_e:
            if through_friendly and w_neighbor.owner != self.game_map.my_id:
                if e_neighbor.owner == self.game_map.my_id:
                    ew_move = (EAST, e_neighbor)
                    ew_swap = True
                else:
                    ew_move = None
            else: 
                ew_move = (WEST, w_neighbor)
        elif dist_e < dist_w:
            if through_friendly and e_neighbor.owner != self.game_map.my_id:
                if w_neighbor.owner == self.game_map.my_id:
                    ew_move = (WEST, w_neighbor)
                    ew_swap = True
                else:
                    ew_move = None
            else: 
                ew_move = (EAST, e_neighbor)
        elif dist_w == 0:
            ew_move = None
        elif dist_w == dist_e:
            if through_friendly and (w_neighbor.owner != self.game_map.my_id or e_neighbor.owner != self.game_map.my_id):
                if w_neighbor.owner != self.game_map.my_id and e_neighbor.owner != self.game_map.my_id:
                    ew_move = None
                elif w_neighbor.owner == self.game_map.my_id and e_neighbor.owner != self.game_map.my_id:
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
            if through_friendly and s_neighbor.owner != self.game_map.my_id:
                if n_neighbor.owner == self.game_map.my_id:
                    ns_move = (NORTH, n_neighbor)
                    ns_swap = True
                else:
                    ns_move = None
            else: 
                ns_move = (SOUTH, s_neighbor)
        elif dist_n < dist_s:
            if through_friendly and n_neighbor.owner != self.game_map.my_id:
                if s_neighbor.owner == self.game_map.my_id:
                    ns_move = (SOUTH, s_neighbor)
                    ns_swap = True
                else:
                    ns_move = None
            else: 
                ns_move = (NORTH, n_neighbor)
        elif dist_s == 0:
            ns_move = None
        elif dist_s == dist_n:
            if through_friendly and (s_neighbor.owner != self.game_map.my_id or n_neighbor.owner != self.game_map.my_id):
                if s_neighbor.owner != self.game_map.my_id and n_neighbor.owner != self.game_map.my_id:
                    ns_move = None
                elif s_neighbor.owner == self.game_map.my_id and n_neighbor.owner != self.game_map.my_id:
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
        elif ns_swap == True and ew_swap == False:
            possible_moves.append(ew_move)
            possible_moves.append(ns_move)
        elif ns_swap == False and ew_swap == True:
            possible_moves.append(ns_move)
            possible_moves.append(ew_move)
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
        future_strength += possible_moves[0][1].strength if (possible_moves[0][1].owner == self.game_map.my_id and (possible_moves[0][1].move == -1 or possible_moves[0][1].move == STILL)) else 0
        if possible_moves[0][1].moving_here != []:
            future_strength += sum(x.strength for x in possible_moves[0][1].moving_here)
        
        if future_strength <= 255 + strength_buffer:
            # We're good, make the move
            self.game_map.make_move(self, possible_moves[0][0])
            return True
            
        # Can we test the second move if it exists?
        if len(possible_moves) > 1:
            future_strength = self.strength + possible_moves[1][1].strength if (possible_moves[1][1].owner == self.game_map.my_id and (possible_moves[1][1].move == -1 or possible_moves[1][1].move == STILL)) else 0
            if possible_moves[1][1].moving_here != []:
                future_strength += sum(x.strength for x in possible_moves[1][1].moving_here)
            
            if future_strength <= 255 + strength_buffer:
                # We're good, make the move
                self.game_map.make_move(self, possible_moves[1][0])
                return True            
        
        # Ok, so moving here will result in too much strength. What are our options?
        # Can we move the cell that we are moving to?
        if possible_moves[0][1].owner == self.game_map.my_id and (possible_moves[0][1].move == -1 or possible_moves[0][1] == STILL):
            if self.strength + sum(x.strength for x in possible_moves[0][1].moving_here) <= 255 + strength_buffer:
                # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                # This is dangerous, make sure to UNDO the fake move.
                self.game_map.make_move(self, possible_moves[0][0])
                success = possible_moves[0][1].move_to_target(target, False)
                if success:
                    return True
                else:
                    # UNDO THE MOVE
                    self.game_map.make_move(self, -1)
                # Is there anywhere else we can move this cell?
                if possible_moves[0][1].moving_here != []:
                    for secondary_target in possible_moves[0][1].moving_here:
                        # Simulate the move
                        self.game_map.make_move(self, possible_moves[0][0])
                        success = possible_moves[0][1].move_to_target(secondary_target.target, False)
                        if success:
                            return True
                        self.game_map.make_move(self, -1)
                # Ok, can we just move the destination to a different square?
                neighbor_targets = []
                for n in possible_moves[0][1].neighbors():
                    neighbor_strength = n.strength if n.owner == self.game_map.my_id else 0
                    neighbor_strength += sum(x.strength for x in n.moving_here)
                    neighbor_targets.append((n, neighbor_strength))
                # Try to move to the lowest strength target.
                neighbor_targets.sort(key = lambda x: x[1])
                # Attempt to move to the lowest strength neighbor
                for n_t in neighbor_targets:
                    if n_t[0].owner != self.game_map.my_id:
                        # We're attempting to attack a cell
                        if n_t[0].strength < possible_moves[0][1].strength + sum(x.strength for x in n_t[0].moving_here):
                            if possible_moves[0][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                self.game_map.make_move(self, possible_moves[0][0])
                                possible_moves[0][1].move_to_target(n_t[0], False)
                                return True
                    else:
                        future_n_strength = possible_moves[0][1].strength
                        future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                        future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                        if future_n_strength <= 255 + strength_buffer:
                            self.game_map.make_move(self, possible_moves[0][0])
                            possible_moves[0][1].move_to_target(n_t[0], True)
                            return True
                        else:
                            break
        # Ok, the cell we are moving to isn't the problem. WE are. Let's try the secondary direction
        if len(possible_moves) > 1:
            if possible_moves[1][1].owner == self.game_map.my_id and (possible_moves[1][1].move == -1 or possible_moves[1][1] == STILL):
                if self.strength + sum(x.strength for x in possible_moves[1][1].moving_here) <= 255 + strength_buffer:
                    # Ok, moving this cell away will be ok. let's try moving it to the same direction we are going to.
                    # This is dangerous, make sure to UNDO the fake move.
                    self.game_map.make_move(self, possible_moves[1][0])
                    success = possible_moves[1][1].move_to_target(target, False)
                    if success:
                        return True
                    else:
                        # UNDO THE MOVE
                        self.game_map.make_move(self, -1)
                    # Is there anywhere else we can move this cell?
                    if possible_moves[1][1].moving_here != []:
                        for secondary_target in possible_moves[0][1].moving_here:
                            # Simulate the move
                            self.game_map.make_move(self, possible_moves[1][0])
                            success = possible_moves[1][1].move_to_target(secondary_target.target, False)
                            if success:
                                return True
                            self.game_map.make_move(self, -1)
                    # Ok, can we just move the destination to a different square?
                    neighbor_targets = []
                    for n in possible_moves[1][1].neighbors():
                        neighbor_strength = n.strength if n.owner == self.game_map.my_id else 0
                        neighbor_strength += sum(x.strength for x in n.moving_here)
                        neighbor_targets.append((n, neighbor_strength))
                    # Try to move to the lowest strength target.
                    neighbor_targets.sort(key = lambda x: x[1])
                    # Attempt to move to the lowest strength neighbor
                    for n_t in neighbor_targets:
                        if n_t[0].owner != self.game_map.my_id:
                            # We're attempting to attack a cell
                            if n_t[0].strength < possible_moves[1][1].strength + sum(x.strength for x in n_t[0].moving_here):
                                if possible_moves[1][1].strength + sum(x.strength for x in n_t[0].moving_here) <= 255 + strength_buffer:
                                    self.game_map.make_move(self, possible_moves[1][0])
                                    possible_moves[1][1].move_to_target(n_t[0], False)
                                    return True
                        else:
                            future_n_strength = possible_moves[1][1].strength
                            future_n_strength += sum(x.strength for x in n_t[0].moving_here)
                            future_n_strength += n_t[0].strength if (n_t[0].move == -1 or n_t[0].move == STILL) else 0
                            if future_n_strength <= 255 + strength_buffer:
                                self.game_map.make_move(self, possible_moves[1][0])
                                possible_moves[1][1].move_to_target(n_t[0], True)
                                return True
                            else:
                                break
        # We can't do anything.
        return False
       
####################
# Helper Functions #
####################

def get_offset(direction):
    return ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
    
def distance_between(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dx > game_map.width / 2:
        dx = game_map.width - dx
    if dy > game_map.height / 2:
        dy = game_map.height - dy
    return dx + dy
    
def opposite_direction(direction):
    return (direction + 2) % 4 if direction != STILL else STILL

def roll_x(M, x):
    return numpy.roll(M, x, 0)

def roll_y(M, y):
    return numpy.roll(M, y, 1)

def roll_xy(M, x, y):
    return numpy.roll(numpy.roll(M, x, 0), y, 1)

def spread_n(M, n, decay = 1, include_self = True):
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
    
def spread(M, decay = 1, include_self = True):
    # For now to save time, we'll use game_map.distance_map and assume that we'll always be using the same falloff distances to calculate offsets.
    
    # Takes the matrix M and then for each point (x, y), calculate the product of the distance map and the decay factor.
    decay_map = numpy.exp(numpy.multiply(game_map.distance_map, -decay))
    
    spread_map = numpy.sum(numpy.multiply(decay_map, M), (2, 3))
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
    return numpy.apply_along_axis(numpy.min, 0, M[numpy.nonzero(mine)])
    

########################
# Core logic functions #    
########################

def heuristic(cell, source = None):

    # Currently, don't assign any value to moving into a friendly cell. This should be done through a different call.
    if cell.owner == game_map.my_id:
        return 0
    
    # If other cells are moving into this square, we don't want to duplicate effort. Especially if there are no enemy cells around
    other_cells_moving_into_cell = cell.moving_here
    cell_neighbors = cell.neighbors()

    bordered_by_hostile = False
    
    for c in cell_neighbors:
        if c.owner != 0 and c.owner != game_map.my_id:
            bordered_by_hostile = True
            
    if len(other_cells_moving_into_cell) > 0 and not bordered_by_hostile:
        # Someone else is capturing this neutral territory already.
        return 0
        
    # If this cell is neutral AND bordered by hostile, let's not attack it.
    #if cell.owner == 0 and bordered_by_hostile and cell.strength != 0: 
    #    return 0

    if cell.owner == 0 and not bordered_by_hostile and cell.strength / max(0.01, cell.production) > game_map.turns_left * 0.8:
        return 0

        
    cell_value = 0
    
    cell_value += production_square_influence_factor * cell.production / max(cell.strength, 1)
    cell_value += game_map.influence_production_map[cell.x, cell.y] * production_influence_factor
    cell_value += game_map.influence_prod_over_str_map[cell.x, cell.y] * prod_over_str_influence_factor
    cell_value += numpy.multiply(game_map.strength_map, game_map.is_enemy_map)[cell.x, cell.y] * enemy_strength_0_influence_factor
    cell_value += game_map.influence_enemy_strength_map[1, cell.x, cell.y] * enemy_strength_1_influence_factor
    cell_value += game_map.influence_enemy_strength_map[2, cell.x, cell.y] * enemy_strength_2_influence_factor
    cell_value += game_map.influence_enemy_strength_map[3, cell.x, cell.y] * enemy_strength_3_influence_factor
    cell_value += game_map.influence_enemy_territory_map[1, cell.x, cell.y] * enemy_territory_1
    cell_value += game_map.influence_enemy_territory_map[2, cell.x, cell.y] * enemy_territory_2
    cell_value += game_map.influence_enemy_territory_map[3, cell.x, cell.y] * enemy_territory_3

    return cell_value

def first_turns_heuristic():
    border_squares = []
    for square in game_map:
        if game_map.border_map[square.x, square.y]:
            border_squares.append((square, game_map.recover_map[game_map.early_game_squares_out_search, square.x, square.y]))
            if game_map.influence_enemy_territory_map[4, square.x, square.y] > 0:
                game_map.phase = 1
    border_squares.sort(key = lambda x: x[1])
    

    threshold = border_squares[0][1] / early_game_value_threshold

    for border in border_squares:
        find_cell = False
        if border[1] <= threshold:
            find_cell = game_map.attack_cell(border[0], 5)
        if find_cell: 
            border_squares.remove(border)
    
    cells_to_consider_moving = []
    for square in game_map:
        # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
        if square.owner == game_map.my_id and (square.move == -1):
            cells_to_consider_moving.append(square)
    
    for square in cells_to_consider_moving:
        if square.strength > (square.production * early_game_buildup_multiplier):
            #if not square.is_border():
            # Move to the highest valued cell
            #    square.move_to_target(border_squares[0][0], True)
            #else:
            if game_map.get_distance(square, border_squares[0][0]) > 2:
                square.move_to_target(border_squares[0][0], True)

#############
# Game Loop #
#############
def game_loop():
    
    game_map.get_frame()
    game_map.update()
    #game_map.create_production_influence_map()
    #logging.debug("\nFrame: " + str(game_map.frame))
    # Have each individual square decide on their own movement

    if game_map.phase == 0:
        #start = time.time()
        first_turns_heuristic()
        #end = time.time()
        #logging.debug("13.4.1 Frame: " + str(game_map.frame) + " : " + str(end - start))
    elif game_map.phase == 1:
        game_map.get_best_moves()
    else: # game_map.phase = 2
        game_map.get_best_moves_late_game()

    game_map.send_frame()




#####################################################################################################################
# Functions for communicating with the Halite game environment (formerly contained in separate module networking.py #
#####################################################################################################################

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


######################
# Game run-time code #
######################

logging.basicConfig(filename='logging.log',level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)

game_map = GameMap()


while True:
    game_loop()