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
#import time


#############
# Variables #
#############

botname = "shummie v10.2.5"

production_decay = 0.7
production_influence_max_distance = 8
buildup_multiplier = 7
early_game_buildup_multiplier = 9
early_game_value_threshold = 0.9
strength_buffer = 0
border_distance_decay_factor = 2
border_target_percentile = .5


production_self_factor = 0
production_neutral_factor = 1.25
production_enemy_factor = 1
production_influence_factor = .4 # Sample values are around 50-80
production_square_influence_factor = 30
prod_over_str_influence_factor = 50 # Sample values are around 0.5 - 1.0
prod_over_str_self_factor = 0
prod_over_str_neutral_factor = 1.25
prod_over_str_enemy_factor = 3
enemy_strength_0_influence_factor = 7
enemy_strength_1_influence_factor = 8 # Sample values around 4-20
enemy_strength_2_influence_factor = 3 # Sample values around 2-40
enemy_strength_3_influence_factor = 1 # Sample values around 2-50
enemy_territory_1 = 40
enemy_territory_2 = 25
enemy_territory_3 = 10

late_game_buildup_multiplier = 6
late_game_production_self_factor = -1
late_game_production_neutral_factor = 2
late_game_production_enemy_factor = 5
late_game_production_influence_factor = 9 # Sample values are around 50-80
late_game_production_square_influence_factor = 0
late_game_prod_over_str_influence_factor = 100 # Sample values are around 0.5 - 1.0
late_game_prod_over_str_self_factor = .4
late_game_prod_over_str_neutral_factor = 2
late_game_prod_over_str_enemy_factor = 2
late_game_enemy_strength_0_influence_factor = 7
late_game_enemy_strength_1_influence_factor = 9 # Sample values around 4-20
late_game_enemy_strength_2_influence_factor = 5 # Sample values around 2-40
late_game_enemy_strength_3_influence_factor = 2 # Sample values around 2-50
late_game_enemy_territory_1 = 20
late_game_enemy_territory_2 = 10
late_game_enemy_territory_3 = 5

        
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

    def initialize_game(self):
        # This should only be called once, and at the beginning of the game
        self.my_id = int(get_string())
        map_size_string = get_string()
        production_map_string = get_string()
        self.early_game = True
        
        self.width, self.height = tuple(map(int, map_size_string.split()))
        self.frame = 0
        
        self.production_map = numpy.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()

        self.get_frame()
        
        # Initialize all the maps that this stores
        
        self.projected_owner_map = numpy.ones((self.width, self.height)) * -1
        self.projected_strength_map = numpy.ones((self.width, self.height)) * -1

        self.starting_player_count = numpy.amax(self.owner_map) # Note, for range you'd need to increase the range by 1
        
        self.next_uncapped_strength_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
        # Create the distance map
        self.create_distance_map()        
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
        if self.frame > 1:
            self.next_uncapped_strength_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
        
        self.frame += 1
        
    def __iter__(self):
        # Allows direct iteration over all squares
        return itertools.chain.from_iterable(self.squares)
    
    def create_maps(self):
        
        # Create is_owner maps
        self.create_is_owner_map()
        self.create_border_map()

        # Create the list of border squares

        self.create_border_square_list()
        self.create_influence_production_map()
        self.create_influence_enemy_strength_map()
        self.create_influence_prod_over_str_map()
     
    def create_is_owner_map(self):
        # Creates a 3-d owner map from self.owner_map
        self.is_owner_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                self.is_owner_map[self.owner_map[x, y], x, y] = 1
    
    def create_border_map(self):
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
        self.is_enemy_map = sum(self.is_owner_map) - self.is_owner_map[0] - self.is_owner_map[self.my_id]
        
        # Do the same as we did for the border map
        self.enemy_border_map = spread_n(self.is_enemy_map, 1)
        
        self.enemy_border_map = numpy.minimum(self.enemy_border_map, 1)
        self.enemy_border_map -= self.is_enemy_map
        
    def create_border_square_list(self):
        self.border_square_list = []
        # Goes through all squares and puts them into the list of all borders for easy searching
        for square in itertools.chain.from_iterable(self.squares):
            if square.is_npc_border():
                self.border_square_list.append(square)

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
        
        self.distance_map = numpy.zeros((self.width, self.height, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                self.distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)

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
        
    def create_influence_enemy_strength_map(self):
        # Creates a list of the enemy strength projections.
        # Get all enemy strengths:
        enemy_strength_map = numpy.multiply(self.strength_map, self.is_enemy_map)
        # It might be better to actually have 1 matrix referenced by [distance, x, y], but let's keep it this way for now.
        self.influence_enemy_strength_map_1 = spread_n(enemy_strength_map, 1)
        self.influence_enemy_strength_map_2 = spread_n(enemy_strength_map, 2)
        self.influence_enemy_strength_map_3 = spread_n(enemy_strength_map, 3)
        self.influence_enemy_strength_map_8 = spread_n(enemy_strength_map, 8)
        
        self.influence_enemy_territory_map_1 = numpy.minimum(self.influence_enemy_strength_map_1, 1)
        self.influence_enemy_territory_map_2 = numpy.minimum(self.influence_enemy_strength_map_2, 1)
        self.influence_enemy_territory_map_3 = numpy.minimum(self.influence_enemy_strength_map_3, 1)
        self.influence_enemy_territory_map_8 = numpy.minimum(self.influence_enemy_strength_map_8, 1)
        
        
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
        
        
    def get_distance(self, sq1, sq2):
        dx = abs(sq1.x - sq2.x)
        dy = abs(sq1.y - sq2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy
        
    def get_target(self, square, direction):
        # This function might be unnecessary?
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
        return self.squares[(square.x + dx) % self.width][(square.y + dy) % self.height]

    def get_coord(self, sourcex, sourcey, dx, dy):
        return ((sourcex + dx) % self.width, (sourcey + dy) % self.height)
    
    def make_move(self, square, direction):
        # Queues up the move to be made.
        # First, store the move in the move_map for easy reference
        self.move_map[square.x, square.y] = direction
        # Update square to new direction
        square.make_move(direction)
    
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
        
    def calculate_uncapped_next_strength(self):
        # Given the move_map, calculate the uncapped strength in each cell.
        for x in range(self.width):
            for y in range(self.height):
                owner = self.owner_map[x, y]
                # 4. Add strength to pieces which choose to remain where they are.
                # Treat all cells that have a move value of -1 or 4 to be increasing in strength.
                # In practice, this is not true for enemy pieces, but for now, let's make this assumption
                if self.move_map[x, y] == 4 or self.move_map[x, y] == -1:
                    self.next_uncapped_strength_map[owner, x, y] += self.strength_map[x, y] + self.production_map[x, y] if owner > 0 else 0
                # 5. Simultaneously move (and combine if necessary) all player's pieces.
                else: 
                    direction = self.move_map[x, y]
                    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[int(direction)]
                    self.next_uncapped_strength_map[owner, (x + dx) % self.width, (y + dy) % self.height] += self.strength_map[x, y]
    
    def get_best_moves(self):
        # Instead of each cell acting independently, look at the board as a whole and make squares move based on that.

        
        # Squares should always be moving towards a border. so get the list of border candidate squares
        all_targets = []
        production_squares = []
        for square in itertools.chain.from_iterable(self.squares):
            if square.is_npc_border():
                if self.influence_enemy_territory_map_8[square.x, square.y] == 0:
                    production_squares.append((square, get_future_value(square, 5)))
                else: 
                    if square.owner == 0 and square.production == 1 and square.strength > 4:
                        continue
                    all_targets.append((square, heuristic(square)))
                
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        production_squares.sort(key = lambda x: x[1], reverse = True)
        all_targets.sort(key = lambda x: x[1], reverse = True)
        best_targets = all_targets[0:int(len(all_targets) * border_target_percentile)]

        if len(production_squares) > 0:
            threshold = production_squares[0][1] * early_game_value_threshold
        for border in production_squares:
            find_cell = False
            if border[1] >= threshold:
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
            if square.owner == self.my_id and (square.move == STILL or square.move == -1):
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
                self.find_nearest_enemy_direction(square)
                #self.go_to_border(square)
        
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

    def go_to_border2(self, square):
        
        targets = [x for x in square.neighbors()]
        targets.sort(key = lambda x: raw_heuristic(x), reverse = True)
        return square.move_to_target(targets[0], True)
            
            
    def go_to_border(self, square):
        # Going to do a simple search for the closest border then determine which of the 4 directions we should go
        
        #self.border_square_list.sort(key = lambda x: x.influence_production_npc() / (self.get_distance(square, x)**0.5), reverse = True)
        #self.border_square_list.sort(key = lambda x: x.influence_production_npc(), reverse = True)
        
        #self.border_square_list.sort(key = lambda x: self.influence_prod_over_str_map[x.x, x.y])
        self.border_square_list.sort(key = lambda x: heuristic(x) / self.get_distance(square, x)**border_distance_decay_factor, reverse = True)
        
        #if len(self.border_square_list) > 0:        
        return square.move_to_target(self.border_square_list[0], True)

    def find_nearest_non_npc_enemy_direction(self, square):
        dir_distance = []
        not_dir = []
        max_distance = min(game_map.width, game_map.height)
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
        
            
        
    def find_nearest_enemy_direction(self, square):

        max_distance = min(game_map.width, game_map.height) / 2
        
        dir_distance = []
        
        for d in (NORTH, EAST, SOUTH, WEST):
            distance = 0
            location = game_map.get_target(square, d)
            target_prod = location.production
            
            while (location.owner == self.my_id) and (distance < max_distance):
                distance += 1
                location = game_map.get_target(location, d)
            
            dir_distance.append((d, distance, target_prod, location))
        
        # For now, keep it simple. We can add in additional information such as, if there is a difference of distance 1, but production difference of 10, 
        # then we should try to go out of our way to avoid the high production square. But that's a future enhancement
        dir_distance.sort(key = lambda x: x[2]) # Sort by production
        dir_distance.sort(key = lambda x: x[1]) # Then sort by distance. Python's sorts are stable so production order is preserved.
        
        success = False
        index = 0
        while not success and index < 4:
            success = square.move_to_target(dir_distance[index][3], True)
            index += 1
        #self.make_move(square, dir_distance[0][0])
            
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

    def prevent_overstrength(self):
        # Tries to prevent wasting strength by having multiple cells move into the same square
        # Calculate the next turn's projected strengths based on moves so far.
        self.calculate_uncapped_next_strength()
        
        # Check the list of cells that will be capped
        cells_over = []
        for x in range(self.width):
            for y in range(self.height):
                if self.owner_map[x, y] == self.my_id: # We only care about our own cells
                    if self.next_uncapped_strength_map[self.my_id, x, y] > (255 + strength_buffer):
                        cells_over.append(self.squares[x, y])
        
        # cells_over contains a list of squares which will be over the strength cap
        cells_over_count = len(cells_over) # We'll be popping squares out so keep the initial count so we can return it later
        while len(cells_over) > 0:
            square = cells_over.pop(0)            
            
            # Case 1: There should never be a reason we are staying still and being too strong. In the event this happens... what?
            # Case 2: We are not moving, let's move this square into a square moving into us
            if (square.move == -1 or square.move == STILL):
                # Try to move into another square which is moving into us
                if len(square.moving_here) > 0:
                    square.move_to_target(random.choice(square.moving_here).target, False)
            else:
                # We are moving but the squares that are moving into here are going to collide.
                # See if we can reroute one of them perpendicular to where they are going, going the opposite direction is likely guaranteed to be counter productive
                if len(square.moving_here) > 1:
                    square_to_move = random.choice(square.moving_here)
                    option1dx, option1dy = get_offset((square_to_move.move + 1) % 4)
                    option2dx, option2dy = get_offset((square_to_move.move + 3) % 4)
                    
                    # Move to the square that would cause the smallest loss in strength
                    option1 = square_to_move.strength + self.next_uncapped_strength_map[self.my_id, (square_to_move.x + option1dx) % self.width, (square_to_move.y + option1dy) % self.height]
                    option2 = square_to_move.strength + self.next_uncapped_strength_map[self.my_id, (square_to_move.x + option2dx) % self.width, (square_to_move.y + option2dy) % self.height]
                    option0 = self.next_uncapped_strength_map[self.my_id, square.x, square.y]

                    if option1 == min(option1, option2, option0):
                        self.make_move(square_to_move, (square_to_move.move + 1) % 4)
                    elif option2 == min(option1, option2, option0):
                        self.make_move(square_to_move, (square_to_move.move + 3) % 4)
                    else:
                        # Do nothing
                        continue
            
        return cells_over_count

    def late_game_attack(self):
        # When we're in the late game, try to knock out the enemy's main production cells.
        # At this stage, we are ignoring neutral targets and focusing on attacking the enemy.
        
        late_game_heuristic_map = numpy.zeros((self.width, self.height))
        
        late_game_heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), late_game_production_square_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.late_game_influence_production_map, late_game_production_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.late_game_influence_prod_over_str_map, late_game_prod_over_str_influence_factor)
        late_game_heuristic_map += numpy.multiply(numpy.multiply(self.strength_map, self.is_enemy_map), late_game_enemy_strength_0_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_1, late_game_enemy_strength_1_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_2, late_game_enemy_strength_2_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_3, late_game_enemy_strength_3_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_1, late_game_enemy_territory_1)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_2, late_game_enemy_territory_2)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_3, late_game_enemy_territory_3)
        
        all_targets = []
        all_border_targets = []
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner != self.my_id: # do we want to target neutrals?
                all_targets.append(square)
            if square.is_npc_border():
                all_border_targets.append((square, late_game_heuristic_map[square.x, square.y]))
                
                
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        all_border_targets.sort(key = lambda x: x[1], reverse = True)
        best_border_targets = all_border_targets[0:int(len(all_targets) * border_target_percentile)]

        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.
        cells_out = 3
        for target in best_border_targets:
            success_attack = self.attack_cell(target[0], cells_out)
            if success_attack:
                best_border_targets.remove(target)

        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        

        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.. 
        cells_to_consider_moving = []
        for square in itertools.chain.from_iterable(self.squares):
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == self.my_id and (square.move == -1):
                cells_to_consider_moving.append(square)

        # Simple logic for now:
        for square in cells_to_consider_moving:
            if square.is_border():
                # Can we attack a bordering cell?
                targets = [n for n in square.neighbors() if (n.owner != self.my_id and n.strength < square.strength)]
                if len(targets) > 0:
                    targets.sort(key = lambda x: late_game_heuristic_map[square.x, square.y], reverse = True)
                    success = False
                    index = 0
                    while index < len(targets) and not success:
                        success = square.move_to_target(targets[index], False)                
                        index += 1
            elif square.strength > (square.production * late_game_buildup_multiplier): 
                 #cell_values = numpy.divide(late_game_heuristic_map, self.distance_map[square.x, square.y, :, :])
                 #tx, ty = numpy.unravel_index(cell_values.argmax(), cell_values.shape)
                 self.find_nearest_enemy_direction(square)
        
    def get_best_moves_late_game(self):
        # Instead of each cell acting independently, look at the board as a whole and make squares move based on that.
        late_game_heuristic_map = numpy.zeros((self.width, self.height))
        
        late_game_heuristic_map += numpy.multiply(numpy.divide(self.production_map, numpy.maximum(self.strength_map, 1)), late_game_production_square_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.late_game_influence_production_map, late_game_production_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.late_game_influence_prod_over_str_map, late_game_prod_over_str_influence_factor)
        late_game_heuristic_map += numpy.multiply(numpy.multiply(self.strength_map, self.is_enemy_map), late_game_enemy_strength_0_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_1, late_game_enemy_strength_1_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_2, late_game_enemy_strength_2_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_strength_map_3, late_game_enemy_strength_3_influence_factor)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_1, late_game_enemy_territory_1)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_2, late_game_enemy_territory_2)
        late_game_heuristic_map += numpy.multiply(self.influence_enemy_territory_map_3, late_game_enemy_territory_3)
        # Squares should always be moving towards a border. so get the list of border candidate squares
        
        all_targets = []
        for square in itertools.chain.from_iterable(self.squares):
            if square.is_npc_border():
                all_targets.append((square, late_game_heuristic_map[square.x, square.y]))
                
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
            if square.owner == self.my_id and (square.move == STILL or square.move == -1):
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
        self._is_npc_border = None
        

    def make_move(self, direction):
        # This should ONLY be called through the GameMap make_move function. Calling this function directly may screw things up
        # Update this square's move
        # Have we set this square's move already?
        dx, dy = get_offset(direction)
        
        if self.move != -1:
            # Yes, let's reset information
            self.target.moving_here.remove(self)
            
        self.move = direction
        self.target = self.game_map.get_target(self, direction)
        self.target.moving_here.append(self)
    
    def neighbors(self, n = 1, include_self = False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            combos = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
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
                
    def is_npc_border(self):
        # Looks at a square and sees if it's an NPC border square
        # Defined as a square which is owned by 0 and has a neighbor of my_id
        # Have we done this calculation already? It shouldn't change within a frame
        return game_map.border_map[self.x, self.y]

    def move_to_target(self, destination, through_friendly):
        # Calculate cardinal direction distance to target.
        dist_w = (self.x - destination.x) % self.game_map.width
        dist_e = (destination.x - self.x) % self.game_map.width
        dist_n = (self.y - destination.y) % self.game_map.height
        dist_s = (destination.y - self.y) % self.game_map.height

        if dist_w == 0 and dist_n == 0:
            return False
        
        possible_moves = []
 
        possible_moves.append((NORTH, self.game_map.owner_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height] == self.game_map.my_id, dist_n if dist_n > 0 else 999, self.game_map.production_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height]))
        possible_moves.append((SOUTH, self.game_map.owner_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height] == self.game_map.my_id, dist_s if dist_s > 0 else 999, self.game_map.production_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height]))
        possible_moves.append((EAST, self.game_map.owner_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_e if dist_e > 0 else 999, self.game_map.production_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))
        possible_moves.append((WEST, self.game_map.owner_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_w if dist_w > 0 else 999, self.game_map.production_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))

           
        # through friendly only
        if through_friendly:
            possible_moves = [x for x in possible_moves if x[1]]


        # Sort. Note sorts need to happen in reverse order of priority.
        random.shuffle(possible_moves) # Shuffle so we don't bias direction.
        possible_moves.sort(key = lambda x: x[3]) # Sort production, smaller is better
        possible_moves.sort(key = lambda x: x[2]) # Sort distance, smaller is better
       
         
        # Check to make sure we can actually go the direction we want without any strength clashing.
        if len(possible_moves) == 0:
            return False
        possible_target = self.game_map.get_target(self, possible_moves[0][0])
    
        
        # Can we safely move into this square?
        future_strength = self.strength + possible_target.strength if (possible_target.owner == self.game_map.my_id and (possible_target.move == -1 or possible_target.move == STILL)) else 0
        if possible_target.moving_here != None:
            future_strength += sum(x.strength for x in possible_target.moving_here)
        
        if future_strength <= 255 + strength_buffer:
            # We're ok, make the move.
            self.game_map.make_move(self, possible_moves[0][0])
            return True
        
        # Ok, so we can't go where we want. Is the next best option a possibility?
        if len(possible_moves) > 1 and (possible_moves[0][2] == possible_moves[1][2]) and (possible_moves[0][1] == possible_moves[1][1]):
            possible_target = self.game_map.get_target(self, possible_moves[1][0])
            future_strength = self.strength + possible_target.strength if (possible_target.owner == self.game_map.my_id and (possible_target.move == -1 or possible_target.move == STILL)) else 0
            if possible_target.moving_here != None:
                future_strength += sum(x.strength for x in possible_target.moving_here)
            if future_strength <= 255 + strength_buffer:
                # We're ok, make the move.
                self.game_map.make_move(self, possible_moves[1][0])
                return True
                
        # ok, so we know moving here as is will result in there being too much strength. Options?
        # Case 1: The cell we are moving to is staying still. Can we get it to move to our target and chain to our destination?
        # Case 1: If the cell we are moving is STILL and moves away we are ok.
        possible_target = self.game_map.get_target(self, possible_moves[0][0])
        if possible_target.owner == self.game_map.my_id and (possible_target.move == -1 or possible_target.move == STILL):
            if self.strength + sum(x.strength for x in possible_target.moving_here) <= 255 + strength_buffer:
                # Ok, let's try to move this cell to the same destination to chain.
                success = possible_target.move_to_target(destination, False)
                if success:
                    self.game_map.make_move(self, possible_moves[0][0])
                    return True
                # Is there anywhere else we can move this cell?
                if possible_target.moving_here != None:
                    for secondary_target in possible_target.moving_here:
                        success = possible_target.move_to_target(secondary_target.target, False)
                        if success:
                            self.game_map.make_move(self, possible_moves[0][0])
                            return True
                # Ok, Is there another square we can make a move into??
                neighbor_targets = []
                for n in self.neighbors():
                    # Move into the lowest strength cell
                    neighbor_strength = (n.strength if n.owner == self.game_map.my_id else 0) + sum(x.strength for x in n.moving_here)
                    neighbor_targets.append((n, neighbor_strength + possible_target.strength, n.owner == self.game_map.my_id))
                neighbor_targets.sort(key = lambda x: x[1])
                neighbor_targets.sort(key = lambda x: x[2], reverse = True)
                if neighbor_targets[0][1] < 255 + strength_buffer:
                    self.game_map.make_move(self, possible_moves[0][0])
                    possible_target.move_to_target(neighbor_targets[0][0], False)
                    return True
        # The cell we are moving to isn't the problem, we are the problem. If we move, we will have a strength collision. See if we can move a different direction
        # Try the other side
        if len(possible_moves) > 1 and (possible_moves[0][2] == possible_moves[1][2]) and (possible_moves[0][1] == possible_moves[1][1]):
            possible_target = self.game_map.get_target(self, possible_moves[1][0])
            if possible_target.owner == self.game_map.my_id and (possible_target.move == -1 or possible_target.move == STILL):
                if self.strength + sum(x.strength for x in possible_target.moving_here) <= 255 + strength_buffer:
                    # Ok, let's try to move this cell to the same destination to chain.
                    success = possible_target.move_to_target(destination, False)
                    if success:
                        self.game_map.make_move(self, possible_moves[1][0])
                        return True
                    # Is there anywhere else we can move this cell?
                    if possible_target.moving_here != None:
                        for secondary_target in possible_target.moving_here:
                            success = possible_target.move_to_target(secondary_target.target, False)
                            if success:
                                self.game_map.make_move(self, possible_moves[1][0])
                                return True
                    # Ok, Is there another square we can make a move into??
                    neighbor_targets = []
                    for n in self.neighbors():
                        # Move into the lowest strength cell
                        neighbor_strength = (n.strength if n.owner == self.game_map.my_id else 0) + sum(x.strength for x in n.moving_here)
                        neighbor_targets.append((n, neighbor_strength + possible_target.strength, n.owner == self.game_map.my_id))
                    neighbor_targets.sort(key = lambda x: x[1])
                    neighbor_targets.sort(key = lambda x: x[2], reverse = True)
                    if neighbor_targets[0][1] < 255 + strength_buffer:
                        self.game_map.make_move(self, possible_moves[1][0])
                        possible_target.move_to_target(neighbor_targets[0][0])
                        return True                
        return False
                
        
        
    def move_to_target_old(self, destination, through_friendly, can_reroute = True):
        if can_reroute == False:
            if self.move != -1: 
                return False
                
        dist_w = (self.x - destination.x) % self.game_map.width
        dist_e = (destination.x - self.x) % self.game_map.width
        dist_n = (self.y - destination.y) % self.game_map.height
        dist_s = (destination.y - self.y) % self.game_map.height

        if dist_w == 0 and dist_n == 0:
            return self.game_map.make_move(self, STILL)
        
        # Prioritize in the following order:
        # 1: Move through OWN territory
        # 2: Move CLOSER to the destination
        # 3: Move through LOWER production square
        possible_moves = []    

        possible_moves.append((NORTH, self.game_map.owner_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height] == self.game_map.my_id, dist_n if dist_n > 0 else 999, self.game_map.production_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height]))
        possible_moves.append((SOUTH, self.game_map.owner_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height] == self.game_map.my_id, dist_s if dist_s > 0 else 999, self.game_map.production_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height]))
        possible_moves.append((EAST, self.game_map.owner_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_e if dist_e > 0 else 999, self.game_map.production_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))
        possible_moves.append((WEST, self.game_map.owner_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_w if dist_w > 0 else 999, self.game_map.production_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))

        # Sort. Note sorts need to happen in reverse order of priority.
        random.shuffle(possible_moves) # Shuffle so we don't bias direction.
        possible_moves.sort(key = lambda x: x[3]) # Sort production, smaller is better
        possible_moves.sort(key = lambda x: x[2]) # Sort distance, smaller is better
        if through_friendly:
            possible_moves.sort(key = lambda x: x[1], reverse = True) # Sort owner, True = 1, False = 0
        #logging.debug(str(possible_moves))
        # The smallest move is the one we'll take.
        # TODO: Should we handle strength overage here??
        
        # Will moving into this square cause a conflict?
        possible_target = self.game_map.get_target(self, possible_moves[0][0])
        if sum(x.strength for x in possible_target.moving_here) + possible_target.strength if ((possible_target.move == -1 or possible_target.move == STILL) and possible_target.owner == self.game_map.my_id) else 0 <= 255 + strength_buffer:
            self.game_map.make_move(self, possible_moves[0][0])
            return True
        # Otherwise, we have a conflict. Can we go another direction?
        elif possible_moves[0][2] == possible_moves[1][2]:
            # Ok, moving to our 2nd choice is the same distance away. Let's try it.
            possible_target = self.game_map.get_target(self, possible_moves[1][0])
            if sum(x.strength for x in possible_target.moving_here) + possible_target.strength if ((possible_target.move == -1 or possible_target.move == STILL) and possible_target.owner == self.game_map.my_id) else 0 <= 255 + strength_buffer:
                # We're ok, make this move instead.
                self.game_map.make_move(self, possible_moves[1][0])
                return True
        # Ok, we can't move to either square without going further away from our target. If the other square is staying still can we swap them in?
        
        possible_target = self.game_map.get_target(self, possible_moves[0][0])
        if possible_target.move == -1 or possible_target.move == STILL: # If it's staying STILL, then it's likely doing so for a reason so don't mess with it
            # Can we tell them to go to our target?
            success = possible_target.move_to_target(destination, False)
            if success:
                self.game_map.make_move(self, possible_moves[0][0])
                return True
            else:
                # Check if we can move them into this square
                if possible_target.strength + sum(x.strength for x in self.moving_here) <= 255 + strength_buffer:
                    # Yes we can, swap!
                    self.game_map.make_move(possible_target, opposite_direction(possible_moves[0][0]))
                    self.game_map.make_move(self, possible_moves[0][0])
                    return True
                elif possible_moves[0][2] == possible_moves[1][2]: # Ok, was the 2nd move option a possibility?
                    possible_target = self.game_map.get_target(self, possible_moves[1][0])
                    if possible_target.strength + sum(x.strength for x in self.moving_here) <= 255 + strength_buffer:
                    # Yes we can, swap!
                        self.game_map.make_move(possible_target, opposite_direction(possible_moves[1][0]))
                        self.game_map.make_move(self, possible_moves[1][0]) 
                        return True
                
        
        # Ok... So, we can't move to another direction without moving further AND the other cell is moving. Just stay still then.
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
    if cell.owner == 0 and bordered_by_hostile and cell.strength != 0: 
        return 0
        
    cell_value = 0
    
    cell_value += production_square_influence_factor * cell.production / max(cell.strength, 1)
    cell_value += game_map.influence_production_map[cell.x, cell.y] * production_influence_factor
    cell_value += game_map.influence_prod_over_str_map[cell.x, cell.y] * prod_over_str_influence_factor
    cell_value += numpy.multiply(game_map.strength_map, game_map.is_enemy_map)[cell.x, cell.y] * enemy_strength_0_influence_factor
    cell_value += game_map.influence_enemy_strength_map_1[cell.x, cell.y] * enemy_strength_1_influence_factor
    cell_value += game_map.influence_enemy_strength_map_2[cell.x, cell.y] * enemy_strength_2_influence_factor
    cell_value += game_map.influence_enemy_strength_map_3[cell.x, cell.y] * enemy_strength_3_influence_factor
    cell_value += game_map.influence_enemy_territory_map_1[cell.x, cell.y] * enemy_territory_1
    cell_value += game_map.influence_enemy_territory_map_2[cell.x, cell.y] * enemy_territory_2
    cell_value += game_map.influence_enemy_territory_map_3[cell.x, cell.y] * enemy_territory_3

    return cell_value

def raw_heuristic(cell, source = None):

    # Returns the raw heuristic, not caring about any other factors
        
    cell_value = 0
    
    cell_value += production_square_influence_factor * cell.production / max(cell.strength, 1)
    cell_value += game_map.influence_production_map[cell.x, cell.y] * production_influence_factor
    cell_value += game_map.influence_prod_over_str_map[cell.x, cell.y] * prod_over_str_influence_factor
    cell_value += numpy.multiply(game_map.strength_map, game_map.is_enemy_map)[cell.x, cell.y] * enemy_strength_0_influence_factor
    cell_value += game_map.influence_enemy_strength_map_1[cell.x, cell.y] * enemy_strength_1_influence_factor
    cell_value += game_map.influence_enemy_strength_map_2[cell.x, cell.y] * enemy_strength_2_influence_factor
    cell_value += game_map.influence_enemy_strength_map_3[cell.x, cell.y] * enemy_strength_3_influence_factor
    cell_value += game_map.influence_enemy_territory_map_1[cell.x, cell.y] * enemy_territory_1
    cell_value += game_map.influence_enemy_territory_map_2[cell.x, cell.y] * enemy_territory_2
    cell_value += game_map.influence_enemy_territory_map_3[cell.x, cell.y] * enemy_territory_3

    return cell_value
    
def first_capture(start_cell):
    # Which bordering cell provides the most total strength
    for n in start_cell.neighbor():
        start_strength = start_cell.strength
        if start_strength > n.strength:
            # We can immediately capture a neighbor. 
            start_strength -= n.strength
            # How many turns to take over an adjacent cell?
        else:
            # How many turns will it take to capture this cell?
            turns_to_capture = math.ceil((n.strength - start_cell.strength) / start_cell.production)
            (10 - turns_to_capture) * math.p
    

def first_turns_heuristic3():
    border_squares = []
    for square in game_map:
        if square.is_npc_border():
            border_squares.append((square, get_future_value(square, 6)))
            if game_map.influence_enemy_territory_map_3[square.x, square.y] > 0:
                game_map.early_game = False
    border_squares.sort(key = lambda x: x[1], reverse = True)
    

    threshold = border_squares[0][1] * early_game_value_threshold

    for border in border_squares:
        find_cell = False
        if border[1] >= threshold:
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
                

def first_turns_heuristic4():
    border_squares = []
    for square in game_map:
        if square.is_npc_border():
            border_squares.append((square, get_future_value(square, 5)))
    border_squares.sort(key = lambda x: x[1], reverse = True)
    
    find_cell = True
    threshold = border_squares[0][1] * early_game_value_threshold
    while find_cell:
        find_cell = game_map.attack_cell(border_squares[0][0], 5)
        if find_cell:
            border_squares.pop(0)
            
    for border in border_squares:
        if border[1] >= threshold:
            find_cell = game_map.attack_cell(border[0], 5)
            
    
    cells_to_consider_moving = []
    for square in game_map:
        # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
        if square.owner == game_map.my_id and (square.move == -1):
            cells_to_consider_moving.append(square)
    
    for square in cells_to_consider_moving:
        if square.strength > square.production * early_game_buildup_multiplier:
            if not square.is_border():
            # Move to the highest valued cell
                square.move_to_target(border_squares[0][0], True)
            else:
                if game_map.get_distance(square, border_squares[0][0]) > 2:
                    square.move_to_target(border_squares[0][0], True)
                
    
            
    
        

def get_future_value(square, squares_out):
    # This is the initial function that calls the recursion
    return get_future_value_rec(square, 0, squares_out)

def get_future_value_rec(square, squares_out, max_dist):
    if squares_out == max_dist:
        if square.owner == game_map.my_id: 
            return 0
        else:
            return square.production / max(square.strength, 0.1)
    else:
        neighbor_vals = []
        this_square_val = 0
        if square.owner != game_map.my_id:
            this_square_val = (square.production / max(square.strength, 0.1))
        for n in square.neighbors():
            neighbor_vals.append(get_future_value_rec(n, squares_out + 1, max_dist))
        return this_square_val + max(neighbor_vals)        
    
def turns_to_capture(cell, range = 4):
    # How many turns does it take to capture the target cell?
    # How far out can we look?
    turn_max = 10
    # Get the distance map for this cell:
    turns = 1
    turns_to_capture = -1
    while turns < turn_max and turns_to_capture == -1:
        cells_out = min(turns, range)
        available_squares = (game_map.move_map == -1) * 1
        distance_matrix = game_map.friendly_flood_fill(cell, cells_out)
        distance_matrix[distance_matrix == -1] = 0

        owned_in_range = numpy.minimum(distance_matrix, 1)
            
        available_strength = numpy.sum(numpy.multiply(numpy.multiply(game_map.strength_map, numpy.minimum(distance_matrix, 1)), available_squares))

        distance_matrix = cells_out - distance_matrix
        distance_matrix[distance_matrix == cells_out] = 0
        available_production = numpy.sum(numpy.multiply(numpy.multiply(game_map.production_map, distance_matrix), available_squares))   
    
        if cells_out <= range:
            future_strength = available_strength + available_production
            if future_strength > cell.strength:
                return cells_out
        else:
            production_per_turn = numpy.sum(numpy.multiply(owned_in_range, game_map.production_map))
            future_strength = available_strength + available_production
            return int((cell.strength - future_strength) / production_per_turn)
            
            
    
    
    

#############
# Game Loop #
#############
def game_loop():
    
    game_map.get_frame()
    game_map.create_maps()
    #game_map.create_production_influence_map()
    #logging.debug("\nFrame: " + str(game_map.frame))
    # Have each individual square decide on their own movement
    #square_move_list = []
    #for square in game_map:
    #    if square.owner == game_map.my_id: 
    #        square_move_list.append(square)
    # Have smaller strength pieces move first. Mainly since otherwise especially for attacking, large pieces bounce back and forth when we want them to attack instead.
    #square_move_list.sort(key = lambda x: x.strength)   
    #percent_owned = len(square_move_list) / (game_map.width * game_map.height)
    
    #if game_map.frame % 25 == 0:
    #    logging.debug("\nFrame: " + str(game_map.frame))
    #    logging.debug("influence production map:")
    #    logging.debug(game_map.influence_production_map)
    #    logging.debug("influence production/str map:")
    #    logging.debug(game_map.influence_prod_over_str_map)
    #    logging.debug("enemy_str_1")
    #    logging.debug(game_map.influence_enemy_strength_map_1)
    #    logging.debug("enemy_str_2")
    #    logging.debug(game_map.influence_enemy_strength_map_2)
    #    logging.debug("enemy_str_3")
    #    logging.debug(game_map.influence_enemy_strength_map_3)
    #    logging.debug("enemy_str_controlled_1")
    #    logging.debug(game_map.influence_enemy_territory_map_1)
    #    logging.debug("enemy_str_controlled_2")
    #    logging.debug(game_map.influence_enemy_territory_map_2)
    #    logging.debug("enemy_str_controlled_3")
    #    logging.debug(game_map.influence_enemy_territory_map_3)
        
    if game_map.early_game and numpy.sum(game_map.is_owner_map[game_map.my_id]) < (10*(game_map.width * game_map.height)**.5) / ((game_map.starting_player_count**0.5) * 10):
        first_turns_heuristic3()
    elif numpy.sum(game_map.is_owner_map[0]) > (game_map.width * game_map.height) * 0.4:
        game_map.get_best_moves()
    else:
        #game_map.get_best_moves()
        #game_map.late_game_attack()
        game_map.get_best_moves_late_game()

    #for square in square_move_list:
    #    game_map.get_best_move(square)
    # Project the state of the board assuming for now that enemy pieces do not move    
    #game_map.create_projection()    
    # Do stuff

    #game_map.attack_border_multiple_pieces()
    #consolidate_strength()
    #if game_map.frame < 10:
    #    consolidate_strength(3)
    #elif game_map.frame < 20:
    #    consolidate_strength(2)
    #elif game_map.frame < 40:
    #game_map.consolidate_strength(1)
    
    #over_count = game_map.width * game_map.height

    #new_over_count = game_map.prevent_overstrength()

    #while new_over_count < over_count:
    #    over_count = new_over_count
    #    new_over_count = game_map.prevent_overstrength()

    
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