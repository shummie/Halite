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

botname = "shummie v15.1"

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
enemy_territory_0 = 50
enemy_territory_1 = 25
enemy_territory_2 = 10

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
late_game_enemy_territory_0 = 20
late_game_enemy_territory_1 = 10
late_game_enemy_territory_2 = -4

        
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
        self.phase = 0 # 0 = early, 1 = mid, 2 = late
        
        self.width, self.height = tuple(map(int, map_size_string.split()))
        
        self.frame = 0
        
        self.production_map = numpy.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()


        self.get_frame()        
        
        self.maps = Maps(self)
        # Initialize all the maps that this stores
        
        self.projected_owner_map = numpy.ones((self.width, self.height)) * -1
        self.projected_strength_map = numpy.ones((self.width, self.height)) * -1

        self.starting_player_count = numpy.amax(self.owner_map) # Note, for range you'd need to increase the range by 1
        
        self.next_uncapped_strength_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
        # Create the distance map
        
        self.get_configs()

        self.evaluators = [EarlyGameEvaluator(self, 0), MidGameEvaluator(self, 1), LateGameEvaluator(self, 2)]
        self.tactical_ais = [EarlyTacticalAI(self, 0), MidTacticalAI(self, 1), LateTacticalAI(self, 2)]
        self.strategy_ais = [EarlyStrategyAI(self, 0), MidStrategyAI(self, 1), LateStrategyAI(self, 2)]
        
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
    
    def get_configs(self):
        # To be expanded. A bit of a hack for now. (Who am i kidding, this will probably stay this way forever.
        early_game_squares_out_search_array = numpy.zeros((7, 5))
        # early_game_squares_out_search_array[map_size, player_count]
        # map_size: 20, 25, 30, 35, 40, 45, 50 = self.width / 5 - 4
        # player_count: 2, 3, 4, 5, 6 = self.starting_player_count - 2
        early_game_squares_out_search_array[0, 0] = 5
        early_game_squares_out_search_array[1, 0] = 5
        early_game_squares_out_search_array[2, 0] = 6
        early_game_squares_out_search_array[3, 0] = 6
        early_game_squares_out_search_array[4, 0] = 7
        early_game_squares_out_search_array[5, 0] = 8
        early_game_squares_out_search_array[6, 0] = 9
        early_game_squares_out_search_array[0, 1] = 5
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
        early_game_squares_out_search_array[0, 3] = 4
        early_game_squares_out_search_array[1, 3] = 4
        early_game_squares_out_search_array[2, 3] = 4
        early_game_squares_out_search_array[3, 3] = 5
        early_game_squares_out_search_array[4, 3] = 6
        early_game_squares_out_search_array[5, 3] = 6
        early_game_squares_out_search_array[6, 3] = 7
        early_game_squares_out_search_array[0, 4] = 4
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
        self.maps.update()
        for e in self.evaluators:
            e.update()
            
        if self.phase == 0 and numpy.sum(self.maps.is_owner_map[self.my_id]) > (10*(self.width * self.height)**.5) / ((self.starting_player_count**0.5) * 10):
            self.phase = 1
        elif self.phase == 1 and numpy.sum(self.maps.is_owner_map[0]) > (self.width * self.height) * 0.4:
            self.phase = 2
        
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

    def get_moves(self):
        # Make moves based on the Tactical AI first, then the strategy AI.
        # Also check for phase change here.
        self.tactical_ais[self.phase].get_moves()
        self.strategy_ais[self.phase].get_moves()
        
        
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
            available_strength = numpy.sum(numpy.multiply(numpy.multiply(self.maps.strength_map, numpy.minimum(distance_matrix, 1)), available_squares))
            
            #logging.debug("avail str: " + str(available_strength))
            # Consider production if all cells stay still.
            distance_matrix = cells_out - distance_matrix
            distance_matrix[distance_matrix == cells_out] = 0
            available_production = numpy.sum(numpy.multiply(numpy.multiply(self.maps.production_map, distance_matrix), available_squares))
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
                still_strength = numpy.sum(numpy.multiply(numpy.multiply(self.maps.strength_map, numpy.minimum(distance_matrix, 1)), available_squares)) # Note this is the new distance map used for available_production
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
        self._neighbors_1 = None
        
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
                
    def is_npc_border(self):
        # Looks at a square and sees if it's an NPC border square
        # Defined as a square which is owned by 0 and has a neighbor of my_id
        # Have we done this calculation already? It shouldn't change within a frame
        return self.game_map.maps.border_map[self.x, self.y]

    def move_to_target(self, destination, through_friendly):
        # Calculate cardinal direction distance to target.
        dist_w = (self.x - destination.x) % self.game_map.width
        dist_e = (destination.x - self.x) % self.game_map.width
        dist_n = (self.y - destination.y) % self.game_map.height
        dist_s = (destination.y - self.y) % self.game_map.height

        if dist_w == 0 and dist_n == 0:
            return False
        
        possible_moves = []
 
        possible_moves.append((NORTH, self.game_map.maps.owner_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height] == self.game_map.my_id, dist_n if dist_n > 0 else 999, self.game_map.maps.production_map[(self.x + 0) % self.game_map.width, (self.y - 1) % self.game_map.height]))
        possible_moves.append((SOUTH, self.game_map.maps.owner_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height] == self.game_map.my_id, dist_s if dist_s > 0 else 999, self.game_map.maps.production_map[(self.x + 0) % self.game_map.width, (self.y + 1) % self.game_map.height]))
        possible_moves.append((EAST, self.game_map.maps.owner_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_e if dist_e > 0 else 999, self.game_map.maps.production_map[(self.x + 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))
        possible_moves.append((WEST, self.game_map.maps.owner_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height] == self.game_map.my_id, dist_w if dist_w > 0 else 999, self.game_map.maps.production_map[(self.x - 1) % self.game_map.width, (self.y + 0) % self.game_map.height]))

           
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
                        possible_target.move_to_target(neighbor_targets[0][0], False)
                        return True                
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


##############
# Maps Class #
##############

# This class stores and calculates all the maps used by the bot

class Maps():
    def __init__(self, game_map):
        self.game_map = game_map
        # Width/height variables are fixed and here to avoid us having to type self.game_map.width over and over again.
        self.width = game_map.width
        self.height = game_map.height
        
        # production, owner, and strength maps are already created in the game_map object but keep them all centralized for consistency
        # These are all 2-d matrices of size width x height
        self.owner_map = numpy.copy(game_map.owner_map)
        self.production_map = numpy.copy(game_map.production_map)
        self.strength_map = numpy.copy(game_map.strength_map)
        
        # Create the distance maps. 1 with the specified decay factor, the other without in case we need it for any calcs that we don't want applied with a decay factor.
        self.distance_map = self.create_distance_map()
        self.distance_map_no_decay = self.create_distance_map(1)      

    def create_distance_map(self, decay = 1):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.
        
        # This is a very expensive function and should only be run during the initialization portion.
        
        # Create the base map for 0, 0
        zero_zero_map = numpy.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                dist_x = min(x, -x % self.width)
                dist_y = min(y, -y % self.width)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        zero_zero_map = zero_zero_map ** decay
        
        distance_map = numpy.zeros((self.width, self.height, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)
                
        return distance_map
        
    def update(self):
        # updates all maps.
        self.owner_map = numpy.copy(self.game_map.owner_map)
        self.strength_map = numpy.copy(self.game_map.strength_map)
        
        
        # Create is_owner maps
        self.update_is_owner_map()
        self.update_is_enemy_map()
        
        self.update_border_map()
        self.update_influence_enemy_maps()
        
    def update_is_owner_map(self):
        # Creates a 3-d owner map from self.owner_map
        # self.is_owner_map[# of players, width, height]
        # self.is_owner_map[player_id] returns a width x height matrix of 0's or 1's depending on whether or not that player owns that square
        # self.is_owner_map[0] returns 1 for all Neutral squares, 0 otherwise
        # self.is_owner_map[game_map.my_id] returns 1 for all player owned squares, 0 otherwise
        
        self.is_owner_map = numpy.zeros((self.game_map.starting_player_count + 1, self.width, self.height))
        # I can probably speed this up by doing something like self.is_owner_map[x] = (self.owner_map == x) or something like that but w/e.
        for x in range(self.width):
            for y in range(self.height):
                self.is_owner_map[self.owner_map[x, y], x, y] = 1
    
    def update_is_enemy_map(self):
        # Creates a 2-d map of enemy controlled territories

        # For now, we won't distinguish between enemies...
        self.is_enemy_map = numpy.ones((self.width, self.height)) - self.is_owner_map[0] - self.is_owner_map[self.game_map.my_id]
        
    def update_border_map(self):
        # The border maps are squares that are NEXT to the territory. 
        # self.border_map[x, y] = 1 if the square is NEXT to OUR territory but is NOT our territory (NEUTRAL)
        # self.enemy_border_map[x, y] = 1 if the square is NEXT to an ENEMY territory but is NEUTRAL territory      
        self.border_map = numpy.zeros((self.width, self.height))
        self.enemy_border_map = numpy.zeros((self.width, self.height))
        
        # Roll the territories we own around by 1 square in all directions
        self.border_map = spread_n(self.is_owner_map[self.game_map.my_id], 1)
        self.border_map = numpy.minimum(self.border_map, 1)
        # Take out our border
        # 1's means the cells that are bordering but not in our territory
        self.border_map -= self.is_owner_map[self.game_map.my_id]

        # Do the same as we did for the border map       
        self.enemy_border_map = spread_n(self.is_enemy_map, 1)
        self.enemy_border_map = numpy.minimum(self.enemy_border_map, 1)
        self.enemy_border_map -= self.is_enemy_map

    def update_influence_enemy_maps(self):
        # Creates a list of the enemy strength projections.
        # Get all enemy strengths:
        enemy_strength_map = numpy.multiply(self.strength_map, self.is_enemy_map)

        # It might be better to actually have 1 matrix referenced by [distance, x, y], but let's keep it this way for now.
        # self.influence_enemy_strength_map[n, x, y]
        # n is the # of squares an enemy's strength emanates out.
        self.influence_enemy_strength_map = numpy.zeros((10, self.width, self.height))
        self.influence_enemy_territory_map = numpy.zeros((10, self.width, self.height))
        
        # Hopefully performance here isn't an issue, but we may want to keep an eye on it.
        for x in range(10):
            # Note, we create a lot of these not necessarily because they're useful, but we can use it to see how far away we are from enemy territory.
            self.influence_enemy_strength_map[x] = spread_n(enemy_strength_map, x)
            self.influence_enemy_territory_map[x] = numpy.minimum(self.influence_enemy_strength_map[x], 1)
        

###################
# Evaluator Class #
###################

class Evaluator():
    def __init__(self, game_map, phase):
        self.game_map = game_map
        self.value_map = numpy.zeros((self.game_map.width, self.game_map.height))
        self.phase = phase
    
    def update(self):
        self.value_map = numpy.zeros((self.game_map.width, self.game_map.height))

class EarlyGameEvaluator(Evaluator):
    def __init__(self, game_map, phase, max_distance = 9):
        Evaluator.__init__(self, game_map, phase)
        self.max_distance = max_distance
        self.value_map = numpy.zeros((max_distance + 1, self.game_map.width, self.game_map.height))
            
    def update(self):
        self.value_map[0] = numpy.divide(self.game_map.maps.production_map, numpy.maximum(self.game_map.maps.strength_map, 0.3))
        self.value_map[0] = numpy.multiply(self.value_map[0], 1 - self.game_map.maps.is_owner_map[self.game_map.my_id])        
        distance = 1
        while distance <= self.max_distance:
            dir_map = numpy.zeros((4, self.game_map.width, self.game_map.height))
            dir_map[0] = roll_xy(self.value_map[distance - 1], 0, 1)
            dir_map[1] = roll_xy(self.value_map[distance - 1], 0, -1)
            dir_map[2] = roll_xy(self.value_map[distance - 1], 1, 0)
            dir_map[3] = roll_xy(self.value_map[distance - 1], -1, 0)
        
            self.value_map[distance] = numpy.add(self.value_map[distance - 1], numpy.amax(dir_map, 0))
            distance += 1

class MidGameEvaluator(Evaluator):
        
    def create_influence_prod_over_str_map(self):
        
        # Creates an influence map based off of production / strength. Very similar to the influence_production_map
        self.influence_prod_over_str_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Calculate the production / str maps.
        prod_str_map = numpy.divide(self.game_map.maps.production_map, numpy.maximum(1, self.game_map.maps.strength_map))
        scaled_prod_str_map = numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[self.game_map.my_id]) * prod_over_str_self_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[0]) * prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_enemy_map) * prod_over_str_enemy_factor
        late_game_scaled_prod_str_map = numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[self.game_map.my_id]) * late_game_prod_over_str_self_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[0]) * late_game_prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_enemy_map) * late_game_prod_over_str_enemy_factor
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_prod_over_str_map = spread_n(scaled_prod_str_map, production_decay, production_influence_max_distance)
        self.late_game_influence_prod_over_str_map = spread_n(late_game_scaled_prod_str_map, production_decay, production_influence_max_distance)


    def create_influence_production_map(self):
        
        self.influence_production_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Take the base production map and alter it based on who controls it
        #modified_production_map = numpy.multiply(self.production_map, self.is_owner_map[self.my_id]) * production_self_factor + numpy.multiply(self.production_map, self.is_owner_map[0]) * production_neutral_factor + numpy.multiply(self.production_map, self.is_enemy_map) * production_enemy_factor

        self_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[self.game_map.my_id]), production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[0]), production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_enemy_map), production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
        
        
        self_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[self.game_map.my_id]), late_game_production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[0]), late_game_production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_enemy_map), late_game_production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.late_game_influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
  
            
    def update(self):
        self.create_influence_prod_over_str_map()
        self.create_influence_production_map()
        
        self.value_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Initialize with the actual prod/str of each cell
        self.value_map += numpy.divide(self.game_map.maps.production_map, numpy.maximum(self.game_map.maps.strength_map, 1)) * production_square_influence_factor
        
        # Add the influence production map
        self.value_map += self.influence_production_map * production_influence_factor    
        self.value_map += self.influence_prod_over_str_map * prod_over_str_influence_factor
        
        self.value_map += self.game_map.maps.influence_enemy_strength_map[0] * enemy_strength_0_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[1] * enemy_strength_1_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[2] * enemy_strength_2_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[3] * enemy_strength_3_influence_factor
        
        self.value_map += self.game_map.maps.influence_enemy_territory_map[0] * enemy_territory_0
        self.value_map += self.game_map.maps.influence_enemy_territory_map[1] * enemy_territory_1
        self.value_map += self.game_map.maps.influence_enemy_territory_map[2] * enemy_territory_2        
        
        # The value of any cell we own is 0.
        self.value_map = numpy.multiply(self.value_map, 1 - self.game_map.maps.is_owner_map[self.game_map.my_id])
        
        # If this cell is neutral AND bordered by hostile, let's not attack it unless it's a recent combat zone (Strength == 0)
        self.value_map = numpy.multiply(self.value_map, 1 - numpy.multiply(self.game_map.maps.enemy_border_map, self.game_map.maps.strength_map > 0))
        
        #self.value_map = numpy.multiply(self.value_map, numpy.multiply(self.game_map.maps.strength_map != 0, 1 - numpy.multiply(self.game_map.maps.is_owner_map[0], self.game_map.maps.enemy_border_map)))
               
class LateGameEvaluator(Evaluator):
        
    def create_influence_prod_over_str_map(self):
        
        # Creates an influence map based off of production / strength. Very similar to the influence_production_map
        self.influence_prod_over_str_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Calculate the production / str maps.
        prod_str_map = numpy.divide(self.game_map.maps.production_map, numpy.maximum(1, self.game_map.maps.strength_map))
        scaled_prod_str_map = numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[self.game_map.my_id]) * prod_over_str_self_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[0]) * prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_enemy_map) * prod_over_str_enemy_factor
        late_game_scaled_prod_str_map = numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[self.game_map.my_id]) * late_game_prod_over_str_self_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_owner_map[0]) * late_game_prod_over_str_neutral_factor + numpy.multiply(prod_str_map, self.game_map.maps.is_enemy_map) * late_game_prod_over_str_enemy_factor
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_prod_over_str_map = spread_n(scaled_prod_str_map, production_decay, production_influence_max_distance)
        self.late_game_influence_prod_over_str_map = spread_n(late_game_scaled_prod_str_map, production_decay, production_influence_max_distance)


    def create_influence_production_map(self):
        
        self.influence_production_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Take the base production map and alter it based on who controls it
        #modified_production_map = numpy.multiply(self.production_map, self.is_owner_map[self.my_id]) * production_self_factor + numpy.multiply(self.production_map, self.is_owner_map[0]) * production_neutral_factor + numpy.multiply(self.production_map, self.is_enemy_map) * production_enemy_factor

        self_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[self.game_map.my_id]), production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[0]), production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_enemy_map), production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
        
        
        self_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[self.game_map.my_id]), late_game_production_self_factor)
        neutral_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_owner_map[0]), late_game_production_neutral_factor) 
        enemy_prod_map = numpy.multiply(numpy.multiply(self.game_map.maps.production_map, self.game_map.maps.is_enemy_map), late_game_production_enemy_factor)
        #modified_production_map = numpy.sum(numpy.sum(self_prod_map, neutral_prod_map), enemy_prod_map)
        modified_production_map = self_prod_map + neutral_prod_map + enemy_prod_map
        
        # Diffuse the production map so that high strength areas might be targeted.
        self.late_game_influence_production_map = spread_n(modified_production_map, production_decay, production_influence_max_distance)
  
            
    def update(self):
        self.create_influence_prod_over_str_map()
        self.create_influence_production_map()
        
        self.value_map = numpy.zeros((self.game_map.width, self.game_map.height))
        
        # Initialize with the actual prod/str of each cell
        self.value_map += numpy.divide(self.game_map.maps.production_map, numpy.maximum(self.game_map.maps.strength_map, 1)) * late_game_production_square_influence_factor
        
        # Add the influence production map
        self.value_map += self.influence_production_map * late_game_production_influence_factor    
        self.value_map += self.influence_prod_over_str_map * late_game_prod_over_str_influence_factor
        
        self.value_map += self.game_map.maps.influence_enemy_strength_map[0] * late_game_enemy_strength_0_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[1] * late_game_enemy_strength_1_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[2] * late_game_enemy_strength_2_influence_factor
        self.value_map += self.game_map.maps.influence_enemy_strength_map[3] * late_game_enemy_strength_3_influence_factor
        
        self.value_map += self.game_map.maps.influence_enemy_territory_map[0] * late_game_enemy_territory_0
        self.value_map += self.game_map.maps.influence_enemy_territory_map[1] * late_game_enemy_territory_1
        self.value_map += self.game_map.maps.influence_enemy_territory_map[2] * late_game_enemy_territory_2        
        
        # The value of any cell we own is 0.
        self.value_map = numpy.multiply(self.value_map, 1 - self.game_map.maps.is_owner_map[self.game_map.my_id])
        
        # If this cell is neutral AND bordered by hostile, let's not attack it.
        self.value_map = numpy.multiply(self.value_map, 1 - numpy.multiply(self.game_map.maps.enemy_border_map, self.game_map.maps.strength_map > 0))
        #self.value_map = numpy.multiply(self.value_map, numpy.multiply(self.game_map.maps.strength_map != 0, 1 - numpy.multiply(self.game_map.maps.is_owner_map[0], self.game_map.maps.enemy_border_map)))


            

###################
# TactialAI Class #    
###################

class TacticalAI():
    def __init__(self, game_map, phase):
        self.game_map = game_map
        self.phase = phase

class EarlyTacticalAI(TacticalAI):
    def get_moves(self):
        # Queues up a list of moves and tries to resolve them if they work.
        
        # Early game tactical AI only tries to capture cells on the border based on a heuristic map.
        
        # Get a list of border squares
        border_squares = []
        for square in self.game_map:
            #logging.debug(self.game_map.maps.border_map[])
            if self.game_map.maps.border_map[square.x, square.y]:
                border_squares.append((square, self.game_map.evaluators[self.phase].value_map[self.game_map.early_game_squares_out_search, square.x, square.y]))
                # For now, we will have the mid-game trigger occur here but we shoudl really have the game map evaluate this on its own.
                if self.game_map.maps.influence_enemy_territory_map[3, square.x, square.y] > 0:
                    self.game_map.phase = 1
        border_squares.sort(key = lambda x: x[1], reverse = True)

        threshold = border_squares[0][1] * early_game_value_threshold # Only consider squares that are at least as valuable as x% of the most valuable cell.        
    
        for border in border_squares:
            find_cell = False
            if border[1] >= threshold:
                find_cell = self.game_map.attack_cell(border[0], 5)
            
class MidTacticalAI(TacticalAI):
    
    def get_moves(self):
        
        # The mid-game AI tries to capture border cells based on the heuristic value.
        # The old version of this identified certain targets to utilize a separate heuristic. Can we call the Earlygame AI on these cells?
        
        # Squares should always be attacking a border. so get the list of border candidate squares
        all_targets = []
        production_targets = []
        for square in self.game_map:
            if self.game_map.maps.border_map[square.x, square.y] == 1 and self.game_map.evaluators[self.phase].value_map[square.x, square.y] > 0:
                if self.game_map.maps.influence_enemy_territory_map[6, square.x, square.y] == 0:
                    # We're far enough away, let's use the early game heuristic function instead
                    production_targets.append((square, self.game_map.evaluators[0].value_map[7, square.x, square.y]))
                else:
                    if (square.owner == 0 and square.production == 1 and square.strength > 40) or (square.production == 0):
                        # A hack to ignore the really low production squares.
                        continue
                    all_targets.append((square, self.game_map.evaluators[self.phase].value_map[square.x, square.y]))
    
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        production_targets.sort(key = lambda x: x[1], reverse = True)
        all_targets.sort(key = lambda x: x[1], reverse = True)
        best_targets = all_targets[0:int(len(all_targets) * border_target_percentile)]

        if len(production_targets) > 0:
            threshold = production_targets[0][1] * early_game_value_threshold
        for border in production_targets:
            find_cell = False
            if border[1] >= threshold:
                find_cell = self.game_map.attack_cell(border[0], 4)
            if find_cell: 
                production_targets.remove(border)
        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.
        cells_out = 3
        for target in best_targets:
            success_attack = self.game_map.attack_cell(target[0], cells_out)
            if success_attack:
                best_targets.remove(target)

class LateTacticalAI(TacticalAI):
    def get_moves(self):
        # The late game AI uses a different set of parameters to determine which border cells to take.        
        # Instead of each cell acting independently, look at the board as a whole and make squares move based on that.
        # Squares should always be moving towards a border. so get the list of border candidate squares
        
        all_targets = []
        for square in self.game_map:
            if self.game_map.maps.border_map[square.x, square.y] == 1:
                all_targets.append((square, self.game_map.evaluators[self.phase].value_map[square.x, square.y]))
                
        # Are all cells equally valuable?
        # Let's keep the top X% of cells. 
        all_targets.sort(key = lambda x: x[1], reverse = True)
        best_targets = all_targets[0:int(len(all_targets) * border_target_percentile)]

        # For each border cell, depending on either the state of the game or the border itself, different valuation algorithms should occur.
        
        # Ok now that we have a list of best targets, see if we can capture any of these immediately.
        cells_out = 3
        for target in best_targets:
            success_attack = self.game_map.attack_cell(target[0], cells_out)
            if success_attack:
                best_targets.remove(target)

        
                
##############
# StrategyAI #
##############

class StrategyAI():
    def __init__(self, game_map, phase):
        self.game_map = game_map
        self.phase = phase

class EarlyStrategyAI(StrategyAI):
    # The early strategy AI only cares about moving pieces towards the most valuable border cells.
    
    def get_moves(self):
                
        cells_to_consider_moving = []
        for square in self.game_map:
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == game_map.my_id and (square.move == -1):
                cells_to_consider_moving.append(square)
        
        # For now, we will move towards the highest value border square
        tx, ty = numpy.unravel_index(numpy.multiply(self.game_map.evaluators[self.phase].value_map[self.game_map.early_game_squares_out_search], self.game_map.maps.border_map).argmax(), (self.game_map.width, self.game_map.height))
        target = self.game_map.squares[tx, ty]

        for square in cells_to_consider_moving:
            if square.strength > (square.production * early_game_buildup_multiplier):
                if game_map.get_distance(square, target) > 2:
                    square.move_to_target(target, True)

class MidStrategyAI(StrategyAI):
    def get_moves(self):
        
        # Now, there are some cells that haven't moved yet, but we might not want to move all of them. 
        cells_to_consider_moving = []
        for square in self.game_map:
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == self.game_map.my_id and (square.move == -1):
                cells_to_consider_moving.append(square)

        # Simple logic for now:
        for square in cells_to_consider_moving:
            if square.is_border() == True:
                continue
                # If there is a square on the border that's not moving. There probably aren't good enough targets. Keep still.
                #targets = [n for n in square.neighbors() if (n.owner != self.game_map.my_id and n.strength < square.strength)]
                #if len(targets) > 0:
                #    targets.sort(key = lambda x: self.game_map.evaluators[self.phase].value_map[x.x, x.y], reverse = True)
                #    if self.game_map.evaluators[self.phase].value_map[targets[0].x, targets[0].y] > 0:
                #        square.move_to_target(targets[0], False)
            elif square.strength > (square.production * buildup_multiplier):
                #self.go_to_border(square)
                self.game_map.find_nearest_enemy_direction(square)
                #self.go_to_border(square)

class LateStrategyAI(StrategyAI):
    def get_moves(self):

        # Now, there are some cells that haven't moved yet, but we might not want to move all of them. 
        cells_to_consider_moving = []
        for square in self.game_map:
            # Do we risk undoing a multi-move capture if we move a piece that's "STILL"?
            if square.owner == self.game_map.my_id and (square.move == -1):
                cells_to_consider_moving.append(square)

        # Simple logic for now:
        for square in cells_to_consider_moving:
            if square.is_border() == True:
                # Can we attack a bordering cell?
                targets = [n for n in square.neighbors() if (n.owner != self.game_map.my_id and n.strength < square.strength)]
                if len(targets) > 0:
                    targets.sort(key = lambda x: self.game_map.evaluators[self.phase].value_map[x.x, x.y], reverse = True)
                    if self.game_map.evaluators[self.phase].value_map[targets[0].x, targets[0].y] > 0:
                        square.move_to_target(targets[0], False)
            elif square.strength > (square.production * buildup_multiplier):
                #self.go_to_border(square)
                self.game_map.find_nearest_non_npc_enemy_direction(square)
                #self.go_to_border(square)                                  
                 
########################
# Core logic functions #    
########################

        

#############
# Game Loop #
#############
def game_loop():
    
    game_map.get_frame()
    #logging.debug("\nFrame: " + str(game_map.frame))
    
    game_map.update()
    #percent_owned = len(square_move_list) / (game_map.width * game_map.height)
    game_map.get_moves()
   

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