# TODO LIST:
# -- Dijkstra's Algorithm for pathfinding?
# -- Look at more than 1 cell deep to see if we can capture territory earlier?
# -- How to identify which direction to focus growth? Looking at production map at beginning to see.
# -- Attack patterns? What's the best strategy for attacking / retreating / reinforcing?
# -- Varying production multiplier by distance to border?

# Version 1: Basic bot implementation - Modifications from random bot: To be added
# Version 2: 
# -- Moved hlt file to single file. 
# -- consolidate_strength: Completely rewritten. Searches all border tiles and then sees if we can consolidate. Ranked by production strength, then sees if we can consolidate.
# -- find_nearest_enemy_direction: Calculate ALL distances, and pick the shortest with lowest production if there is a tie. Otherwise, pick randomly.
# -- heuristic: look at not just the cell but adjacent cells to determine the value of capturing the cell.
# -- smallest strength cells move first. 
# Version 3: Uploaded incorrectly
# Version 4:
# -- move_to_target: old implementation might move into uncontrolled territory. Not good. New implementation moves only though adjacent owned territory, if possible. If multiple
#                    routes exist, then it takes the direction with the lowest production.
# -- consolidate_strength: Split into two subroutines. One which is the old multi-attacker into a cell, the other looks outwards to gather strength to attack a cell.
# --                       Idea: Can we expand multi-attacking into a cell to also look and see if we can capture a cell by moving units INTO adjacent cells??
# Version 5: Rewrote heuristic function. Tries not to overvalue expanding into cells bordering enemy territory too much.
# Version 6: ??
# Version 7: Rewrote move to border function. Now squares will try to move towards higher production cells instead of the nearest border.
# -- Complete code overhaul. Remove GameMap class, add Square class.

###########
# Imports #
###########
import math
from itertools import chain
import sys
import logging
import numpy
import random


#############
# Variables #
#############

botname = "shummie v7.6"
production_decay = 0.50
production_influence_max_distance = 5
buildup_multiplier = 5
strength_buffer = 0

        

        
        
#################
# GameMap Class #
#################

class GameMap:
    def __init__(self):
        
        self.initialize_game()

    def initialize_game(self):
        # This should only be called once, and at the beginning of the game
        self.my_id = int(get_string())
        map_size_string = get_string()
        production_map_string = get_string()
        
        self.width, self.height = tuple(map(int, map_size_string.split()))
        self.frame = 0
        
        self.production_map = numpy.array(list(map(int, production_map_string.split()))).reshape((self.height, self.width)).transpose()

        self.get_frame()
        
        # Initialize all the maps that this stores
        
        self.projected_owner_map = numpy.ones((self.width, self.height)) * -1
        self.projected_strength_map = numpy.ones((self.width, self.height)) * -1

        self.starting_player_count = numpy.amax(self.owner_map) # Note, for range you'd need to increase the range by 1
        
        self.next_uncapped_strength_map = numpy.zeros((self.starting_player_count + 1, self.width, self.height))
        
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
        return chain.from_iterable(self.squares)
        

        
    def is_npc_border(self, square):
        # Looks at a square and sees if it's an NPC border square
        # Defined as a square which is owned by 0 and has a neighbor of my_id
        if square.owner != 0: return False
        for n in self.neighbors(square):
            if n.owner == self.my_id:
                return True
        return False
        
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
        for sq in chain.from_iterable(self.squares):
            if sq.owner == self.my_id:
                if sq.move == -1 and sq.strength > 0:
                    # In the event we didn't actually assign a move, make sure it's moving
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

    def create_production_influence_map(self):
        # Lots of tweaking to do...
        # Start with a basic prod/strength evaluation for npc cells
        for x in range(self.width):
            for y in range(self.height):                                
                prod_value = self.production_map[x, y]
                if self.owner_map[y, x] == 0:
                    str_value = max(1, self.strength_map[x, y])
                else:
                    # If we want to do something differently with strengths in enemy territories, we can alter it here.
                    str_value = max(1, self.strength_map[x, y])  # This will cause cells to avoid border cells...

                value = prod_value / str_value
                combos = ((dx, dy) for dy in range(-production_influence_max_distance, production_influence_max_distance+1) for dx in range(-production_influence_max_distance, production_influence_max_distance+1) if abs(dx) + abs(dy) <= production_influence_max_distance)
                for c in combos:
                    distance = abs(c[0]) + abs(c[1])
                    decay_factor = math.exp(-production_decay * distance)
                    self.influence_production_map[self.owner_map[x, y], (x + c[0]) % self.width, (y + c[1]) % self.height] += value * decay_factor

    def get_best_move(self, square):
        # For a given square, find the "best" move we can
        border = False
        
        targets = []
        for d in (NORTH, EAST, SOUTH, WEST):
            target = game_map.get_target(square, d)
            if target.owner != self.my_id:
                border = True
                val = heuristic(target, square)
                targets.append((target, val))

        targets.sort(key = lambda x: x[1], reverse = True) # Sorts targets from high to low
        
        # We have a list of all adjacent cells. If targets is not None, let's see what we can do
        if len(targets) > 0:
            # Go through the list and see if we can attack one
            for t in targets:
                if t[0].strength < square.strength:
                    return square.move_to_target(t[0], False)
                    
        # if we don't have enough strength to make it worth moving yet, stay still
        if square.strength < (square.production * buildup_multiplier):
            # Don't actually set a cell to STILL unless we want it to stay still
            return True
        # If we aren't at a border, move towards the closest one
        elif not border:
            return self.go_to_border(square)
        # Else, we're at a border, don't have the strength to attack another cell, and have less than the buildup multipler. Let other functions handle movement
        else:
            return True
    
    def go_to_border(self, square):
        # Going to do a simple search for the closest border then determine which of the 4 directions we should go
        target = (None, 0)
        max_distance = max(game_map.width, game_map.height) / 2
        for d in (NORTH, EAST, SOUTH, WEST):
            distance = 1
            location = self.get_target(square, d)
            while location.owner == self.my_id and distance < max_distance:
                location = self.get_target(location, d)
            border_value = location.influence_production_npc()
            scaled_value = border_value / distance
            if scaled_value > target[1]:
                target = (location, scaled_value)
        if target[0] != None:
            square.move_to_target(target[0], True)
        else:
            # If all cardinal directions are owned, is it possible to actually not move?
            # Move randomly then?
            self.make_move(square, random.choice(range(4)))
'''            
    def prevent_overstrength(self):
        # Tries to prevent wasting strength by having multiple cells move into the same square
        # Calculate the next turn's projected strengths based on moves so far.
        self.calculate_uncapped_next_strength()
        
        # Check the list of cells that will be capped
        cells_over = []
        for x in range(self.width):
            for y in range(self.height):
                if self.owner_map[x, y] == self.my_id: # We only care about our own cells
                    if game_map.next_uncapped_strength_map[self.my_id, x, y] > (255 + strength_buffer):
                        cells_over.append(self.squares[x, y])
        
        # cells_over contains a list of squares which will be over the strength cap
        cells_over_count = len(cells_over) # We'll be popping squares out so keep the initial count so we can return it later
        while len(cells_over) > 0:
            square = cells_over.pop(0)            
            
            # Case 1: If No squares are moving INTO this cell AND we are not moving, then we're going over due to production.
            # Let's move this cell closer to the border so we can do something with it.
            if len(square.moving_here) == 0 and (square.move == -1 or square.move == STILL):
                # If we are capped, we should move to the lowest valued cell:
                if square.strength >= 222:
                    # See if we can swap with a cell
                    cell_neighbors = [x for x in square.neighbors() if x.owner == self.my_id]
                    cell_neighbors.sort(key = lambda x: x.strength)
                    square.move_to_target(cell_neighbors[0])
                else:
                    self.go_to_border(square)
            # Case 2: Squares are moving INTO this cell and we are not moving. Move to the square with the least future strength that we own.
            elif len(square.moving_here) > 0 and (square.move == -1 or square.move == STILL):
                # Will moving out of the square fix the problem?
                if self.next_uncapped_strength_map[self.my_id, x, y] - square.strength - square.production < (255 + strength_buffer):
                    # Yes it will. Let's move to the square with the lowest future strength
                    cell_neighbors = [x for x in square.neighbors() if x.owner == self.my_id]
                    cell_neighbors.sort(key = lambda x: x.strength)
                    if len(cell_neighbors) > 0:
                        square.move_to_target(cell_neighbors[0], True)
                else:
                    # Moving out won't solve the problem. Since the problem is solved iteratively, let's try changing one of the incoming cells to STILL
                    cell_to_change = random.choice(square.moving_here)
                    self.make_move(cell_to_change, STILL)
            else:
                # We're already moving but the cell is being overpopulated. Change a random cell to STILL
                # TODO: There is a better way to do this but this'll do for now.
                cell_to_change = random.choice(square.moving_here)
                self.make_move(cell_to_change, STILL)
        
        return cells_over_count
'''
            
    def prevent_overstrength(self):
        # Tries to prevent wasting strength by having multiple cells move into the same square
        # Calculate the next turn's projected strengths based on moves so far.
        self.calculate_uncapped_next_strength()
        
        # Check the list of cells that will be capped
        cells_over = []
        for x in range(self.width):
            for y in range(self.height):
                if self.owner_map[x, y] == self.my_id: # We only care about our own cells
                    if game_map.next_uncapped_strength_map[self.my_id, x, y] > (255 + strength_buffer):
                        cells_over.append(self.squares[x, y])
        
        # cells_over contains a list of squares which will be over the strength cap
        cells_over_count = len(cells_over) # We'll be popping squares out so keep the initial count so we can return it later
        while len(cells_over) > 0:
            square = cells_over.pop(0)            
            
            # Case 1: If No squares are moving INTO this cell AND we are not moving, then we're going over due to production.
            # Let's move this cell closer to the border so we can do something with it.
            # Case 2: Squares are moving INTO this cell and we are not moving. Move to the square with the least future strength that we own.
            if (square.move == -1 or square.move == STILL):
                # Will moving out of the square fix the problem?
                if self.next_uncapped_strength_map[self.my_id, x, y] - square.strength - square.production < (255 + strength_buffer):
                    # Yes it will. Let's move to the square with the lowest future strength
                    directions = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
                    for d in directions:
                        cell_neighbors = [x for x in square.neighbors() if x.owner == self.my_id]
                        cell_neighbors.sort(key = lambda x: self.next_uncapped_strength_map[x.x, x.y])
                    if len(cell_neighbors) > 0:
                        square.move_to_target(cell_neighbors[0], True)
                else:
                    # Moving out won't solve the problem. Since the problem is solved iteratively, let's try changing one of the incoming cells to STILL
                    cell_to_change = random.choice(square.moving_here)
                    self.make_move(cell_to_change, STILL)
            else:
                # We're already moving but the cell is being overpopulated. Change a random cell to STILL
                # TODO: There is a better way to do this but this'll do for now.
                cell_to_change = random.choice(square.moving_here)
                self.make_move(cell_to_change, STILL)
        
        return cells_over_count

        
    def attack_border_multiple_pieces(self):
        # Looks to see if there are any border cells which can be attacked right now by multiple pieces at the same time.
        # Looks only at cells whose move value is -1 and are bordering a neighboring cell.
        border_squares = []
        for square in chain.from_iterable(self.squares):
            if square.is_npc_border():
                border_squares.append((square, heuristic(square)))
        
        border_squares.sort(key = lambda x: x[1], reverse = True)
        
        for border_square in border_squares:
            # For each border square, starting with the most valuable, attempt to capture it.
            friendly_neighbors = [x for x in border_square[0].neighbors() if x.owner == self.my_id]
            available_strength = 0
            # TODO: There's a more pythonic way to do this instead of the loop below. 
            for f in friendly_neighbors:
                if f.move == -1:
                    available_strength += f.strength
            
            if available_strength > border_square[0].strength:
                attacking_strength = 0
                for f in friendly_neighbors:
                    if f.move == -1 and attacking_strength <= border_square[0].strength:
                        attacking_strength += f.strength
                        f.move_to_target(border_square[0], False)
    
    def consolidate_strength(self, cells_out = 1):
        # Looks at border cells and sees if there is an opportunity to look N neighbors out to consolidate strength to capture a territory.
        border_squares = []
        for square in chain.from_iterable(self.squares):
            if square.is_npc_border():
                border_squares.append((square, heuristic(square)))
                
        border_squares.sort(key = lambda x: x[1], reverse = True) # Sorts by all border cells which will not be taken next turn by the heuristic above.        

        distance = 1
        while distance <= cells_out:
            self.consolidate_n_out(border_squares, distance)
            distance += 1
    
    def consolidate_n_out(self, border_squares_list, cells_out):
        # For each border_square, we want to look at each friendly neighbor and see if we can take over this square in cells_out turns from now.
        
        for border_square_tuple in border_squares_list:
            border_square = border_square_tuple[0]
            # Get a list of all friendly neighbors to this square: These are the TARGET squares to move to.
            friendly_neighbors = [x for x in border_square.neighbors() if x.owner == self.my_id]
            
            for f in friendly_neighbors:
                # How much strength do we need and can we get it cells_out away?
                needed_strength = border_square.strength + 1
                
                moving_cells = False
                
                # Check friendly neighboring cells.
                for distance_out in range(1, cells_out + 1):
                    neighbor_strength = 0
                    # Are we currently moving? If not, we can add this to the strength
                    if f.move == -1:
                        # While we can check if f.move == STILL, it's likely that it's STILL for a reason and we don't want to cause conflicts.
                        neighbor_strength += (f.strength + (f.production * distance_out))
                    f_neighbors = [x for x in f.neighbors(distance_out) if x.owner == self.my_id]
                    # This returns a list of ALL neighbors between 1 and distance_out inclusive.
                    f_neighbors_minus_one = []
                    if distance_out > 1:
                        f_neighbors_minus_one = [x for x in f.neighbors(distance_out - 1) if x.owner == self.my_id]
                    f_neighbors_at_cells_out = list(set(f_neighbors) - set(f_neighbors_minus_one))
                    # Ok, now we have a list of all cells AT distance_out and all cells LESS than distance_out
                    # Why is this necessary? We only want to MOVE cells at distance_out and let all squares LESS than distance_out produce
                    
                    # Ok, first, check needed strength for all squares LESS than distance_out
                    for f_n in f_neighbors_minus_one:
                        if f_n.move == -1:
                            neighbor_strength += f_n.strength + f_n.production * self.get_distance(f_n, f)
                    # Now, check if moving neighbors will produce enough strength.
                    needed_strength_at_cells_out = needed_strength - neighbor_strength
                    for f_n in f_neighbors_at_cells_out:
                        if f_n.move == -1:
                            neighbor_strength += f_n.strength
                    # Do we have enough strength?
                    if neighbor_strength > needed_strength:
                        # Yes! Let's move the outside squares towards f_n.
                        f_neighbors_at_cells_out.sort(key = lambda x: x.strength, reverse = True)
                        for f_n in f_neighbors_at_cells_out:
                            if f_n.move == -1 and needed_strength_at_cells_out > 0:
                                f_n.move_to_target(f, True) 
                                # There may be edge cases where we can't actually move to a square, or that it takes more turns than expected. Might need to make a new function that looks at distance through friendly squares
                                needed_strength_at_cells_out -= f_n.strength
                        moving_cells = True
                        if f.move == -1:
                            self.make_move(f, STILL)
                        # Stop looking any further out
                        break
                    
                if moving_cells:
                    # We've found something to attack this border square eventually, let's move to the next.
                    break
        
        
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
        self._influence_production_npc = None
        

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
        if self._is_border == None:
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
        if self._is_npc_border == None:
            if self.owner != 0:
                self._is_npc_border = False
                return False
            for n in self.neighbors():
                if n.owner == self.game_map.my_id:
                    self._is_npc_border = True
                    return True
            self._is_npc_border = False
            return False
        return self._is_npc_border
    
    def move_to_target(self, destination, through_friendly):
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

        possible_moves.append((NORTH, self.game_map.owner_map[(self.y - 1) % self.game_map.height, (self.x + 0) % self.game_map.width] == self.game_map.my_id, dist_n if dist_n > 0 else 999, self.game_map.production_map[(self.y - 1) % self.game_map.height, (self.x + 0) % self.game_map.width]))
        possible_moves.append((SOUTH, self.game_map.owner_map[(self.y + 1) % self.game_map.height, (self.x + 0) % self.game_map.width] == self.game_map.my_id, dist_s if dist_s > 0 else 999, self.game_map.production_map[(self.y + 1) % self.game_map.height, (self.x + 0) % self.game_map.width]))
        possible_moves.append((EAST, self.game_map.owner_map[(self.y + 0) % self.game_map.height, (self.x + 1) % self.game_map.width] == self.game_map.my_id, dist_e if dist_e > 0 else 999, self.game_map.production_map[(self.y + 0) % self.game_map.height, (self.x + 1) % self.game_map.width]))
        possible_moves.append((WEST, self.game_map.owner_map[(self.y + 0) % self.game_map.height, (self.x - 1) % self.game_map.width] == self.game_map.my_id, dist_w if dist_w > 0 else 999, self.game_map.production_map[(self.y + 0) % self.game_map.height, (self.x - 1) % self.game_map.width]))

        # Sort. Note sorts need to happen in reverse order of priority.
        random.shuffle(possible_moves) # Shuffle so we don't bias direction.
        possible_moves.sort(key = lambda x: x[3]) # Sort production, smaller is better
        possible_moves.sort(key = lambda x: x[2]) # Sort distance, smaller is better
        if through_friendly:
            possible_moves.sort(key = lambda x: x[1], reverse = True) # Sort owner, True = 1, False = 0
        #logging.debug(str(possible_moves))
        # The smallest move is the one we'll take.
        self.game_map.make_move(self, possible_moves[0][0])        
    
    def influence_production_npc(self):
        # So that we don't have to calculate the entire map every tick, and to prevent recalcs, calculate and store into the square so we can reference it whenever we want
        # Lots of tweaking to do.
        if self._influence_production_npc == None:
            self._influence_production_npc = 0
            if self.owner == 0:
                # I think for any purpose we would use here, if we own the territory, we don't actually care about this value
                return self._influence_production_npc
                
            neighbors = self.neighbors(production_influence_max_distance, True)
            
            for n in neighbors:
                distance = self.game_map.get_distance(self, n)
                prod_n = n.production
                if n.owner == 0:
                    str_value = max(1, n.strength)
                elif n.owner == self.game_map.my_id:
                    # Do not assign any influence for cells we own
                    prod_n = 0
                    str_value = 1 # This doesn't matter i think since value will just equal 0
                else:
                    # If we want to do something differently with strengths in enemy terrotiry, we can alter it here.
                    str_value = max(1, n.strength)
                
                decay_factor = math.exp(-production_decay * distance)
                value = prod_n / str_value
                
                self._influence_production_npc += value * decay_factor
                
        return self._influence_production_npc
        
        
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
        if c.owner != 0:
            bordered_by_hostile = True
            
    if len(other_cells_moving_into_cell) > 0 and not bordered_by_hostile:
        # Someone else is capturing this neutral territory already.
        return 0
        
    # Calculate how much attack damage we would do by moving into here (assumes everyone else stays still)
    total_damage = 0
    
    # Calculate the strength of other cells moving into here
    total_attack_strength = 0
    for c in other_cells_moving_into_cell:
        if c.owner == game_map.my_id:
            total_attack_strength += c.strength
    
    directions = [NORTH, EAST, SOUTH, WEST]
    for d in directions:
        target = game_map.get_target(cell, d)
        if target.owner != 0 and target.owner != game_map.my_id:
            total_damage += min(max(target.strength - total_attack_strength, 0), source.strength if source != None else 999)

    value = 0          
    neighbor_values = []
    if cell.owner == 0:
        #value = max(1, cell.strength) / cell.production # Number of turns to recover. LOWER is better.
        production_value = cell.production / max(cell.strength, 1)
        for c in cell_neighbors:
            if c.owner == 0:
                neighbor_values.append(c.production / max(c.strength, 1))
    value = production_value + 0.1 * sum(neighbor_values)
    
    # This should be changed, but we'll keep it at this for now:
        
    return value + total_damage # Total damage is going to totally overpower value...




#############
# Game Loop #
#############
def game_loop():
    
    game_map.get_frame()
    #game_map.create_production_influence_map()
    #logging.debug("Frame: " + str(game_map.frame) + "\n")
    # Have each individual square decide on their own movement
    square_move_list = []
    for square in game_map:
        if square.owner == game_map.my_id: 
            square_move_list.append(square)
    # Have smaller strength pieces move first. Mainly since otherwise especially for attacking, large pieces bounce back and forth when we want them to attack instead.
    square_move_list.sort(key = lambda x: x.strength)   

    #percent_owned = len(square_move_list) / (game_map.width * game_map.height)

    for square in square_move_list:
        game_map.get_best_move(square)
    # Project the state of the board assuming for now that enemy pieces do not move    
    #game_map.create_projection()    
    # Do stuff

    game_map.attack_border_multiple_pieces()
    #consolidate_strength()
    #if game_map.frame < 10:
    #    consolidate_strength(3)
    #elif game_map.frame < 20:
    #    consolidate_strength(2)
    #elif game_map.frame < 40:
    game_map.consolidate_strength(1)
    
    over_count = game_map.width * game_map.height
    
    new_over_count = game_map.prevent_overstrength()
    
    while new_over_count < over_count:
        over_count = new_over_count
        new_over_count = game_map.prevent_overstrength()
    
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
    
    
    
    
    
    
    
    