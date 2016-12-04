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
# Version 3:
# -- move_to_target: old implementation might move into uncontrolled territory. Not good. New implementation moves only though adjacent owned territory, if possible. If multiple
#                    routes exist, then it takes the direction with the lowest production.


import math
from collections import namedtuple
from itertools import chain, zip_longest, count
import sys
import logging
import numpy
import random


def heuristic(cell):
    if cell.owner == 0:
        if len(game_map.get_all_moving_to(cell)) > 0: # If someone is moving here, then we probably don't need to since they should have enough strength.
            return 0
        else:
            # Get a list of all neighbors.
            cell_npc_neighbors = [x for x in game_map.neighbors(cell) if x.owner == 0]
            value = cell.production / (cell.strength if cell.strength > 0 else 1)
            n_values = 0
            n_cells = 0
            for n in cell_npc_neighbors:
                n_values += n.production / (n.strength if n.strength > 0 else 1)
                n_cells += 1
            # TODO: Altering this cell below is probably a major key in advancing some ranks.
            value += 0.1 * n_values    # Give partial weight to neighboring cells. Will then attempt to capture cells that have neighbors over cells that don't...
            return value
    else:
        total_damage = 0
        directions = [NORTH, EAST, SOUTH, WEST]
        # random.shuffle(directions)        
        for d in directions:
            target = game_map.get_target(cell, d)
            if target.owner != 0 and target.owner != my_id:
                total_damage += target.strength
        return total_damage
        

def find_nearest_enemy_direction(square):

    direction = NORTH
    max_distance = min(game_map.width, game_map.height) / 2
    
    dir_distance = []
    
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        location = game_map.get_target(square, d)
        target_prod = location.production
        
        while (location.owner == my_id) and (distance < max_distance):
            distance += 1
            location = game_map.get_target(location, d)
        
        dir_distance.append((d, distance, target_prod))
    
    # For now, keep it simple. We can add in additional information such as, if there is a difference of distance 1, but production difference of 10, 
    # then we should try to go out of our way to avoid the high production square. But that's a future enhancement
    dir_distance.sort(key = lambda x: x[2]) # Sort by production
    dir_distance.sort(key = lambda x: x[1]) # Then sort by distance. Python's sorts are stable so production order is preserved.
    
    return dir_distance[0][0]
    
    
def get_move(square, buildup_multiplier = 5):
    #buildup_multiplier = 9

    border = False    
    targets = []
    
    # We don't consider STILL. Are there situations where staying still would result in MORE strength? Would this require simulating enemy movements?
    for d in (NORTH, EAST, SOUTH, WEST):
        target = game_map.get_target(square, d)
        if target.owner != my_id:
            border = True
            val = heuristic(target)
            targets.append((target, val))
    
    targets.sort(key = lambda x: x[1], reverse = True)  # Sorts the targets from high to low based on the heuristic
    
    # We have a list of values for all adjacent cells. If targets is not none, let's see what we can do.
    if len(targets) > 0:
        # Go through the list and see if we can attack one.
        for t in targets:
            if t[0].strength < square.strength:
                return game_map.move_to_target(square, t[0], False)
                
    # If we don't have enough strength to make it worth moving yet, stay still
    #if square.strength < max(15, (square.production * buildup_multiplier)):
    if square.strength < (square.production * buildup_multiplier):
        game_map.make_move(square, STILL)
    # If we aren't at a border, move towards the closest one
    elif not border:
        game_map.make_move(square, find_nearest_enemy_direction(square))
    # Else, we're at a border, don't have the strength to attack anyone adjacent, and have less than the buildup multiplier
    # Can we combine forces with an adjacent cell to capture another cell?
    else: 
        game_map.make_move(square, STILL)
    

def prevent_overstrength():
    # Calculate the next turn's projected strengths
    game_map.calculate_uncapped_next_strength()
    
    # Allow going over by the strength_buffer to avoid excess movement.
    strength_buffer = 0
    
    # Check the list of cells which will be capped:
    cells_over = []
    directions = ((0, 1), (1, 0), (0, 1), (-1, 0)) # N, E, S, W
    for y in range(game_map.height):
        for x in range(game_map.width):
            if game_map.owner_map[y, x] == my_id:    # We only care about our cells.
                if game_map.next_uncapped_strength_map[y, x, my_id] > (255 + strength_buffer):
                    cells_over.append((x, y))
    
    cells_over_count = len(cells_over)
    
    # cells_over contains a list of all cells that are over:
    while len(cells_over) > 0:
        x, y = cells_over.pop(0)        
        moving_into = []
        # Get a list of all squares that will be moving INTO this cell next turn.
        for direction in (NORTH, EAST, SOUTH, WEST):
            dx, dy = directions[direction]
            if game_map.owner_map[(y + dy) % game_map.height, (x + dx) % game_map.width] == my_id: # only care about our cells moving into this square
                if game_map.move_map[(y + dy) % game_map.height, (x + dx) % game_map.width] == opposite_direction(direction):
                    moving_into.append(((x + dx) % game_map.width, (y + dy) % game_map.height))
        # Case 1: NO squares are moving into this cell AND we are not moving -- Going over due to overproduction. 
        if len(moving_into) == 0 and game_map.move_map[y, x] == STILL:
            # Move into the cell (even if it's staying still) that has the least future strength.
            # Check projected strength for next turn
            cell_strengths = [(STILL, game_map.production_map[y, x])]
            for direction in (NORTH, EAST, SOUTH, WEST):
                dx, dy = directions[direction]
                cell_strengths.append((direction, game_map.next_uncapped_strength_map[y, x, my_id]))
            cell_strengths.sort(key = lambda tup: tup[1])            
            game_map.move_map[y, x] = cell_strengths[0][0]
        # Case 2: Squares are moving into this cell AND we are not moving - Move to the square with the least future strength?
        elif len(moving_into) > 0 and game_map.move_map[y, x] == STILL:
            # Will moving out of this square fix the problem?
            if game_map.next_uncapped_strength_map[y, x, my_id] - game_map.strength_map[y, x] - game_map.production_map[y, x] < (255 + strength_buffer):
                # Yes it will, so let's move out into the lowest future strength square.
                cell_strengths = []
                for direction in (NORTH, EAST, SOUTH, WEST):
                    dx, dy = directions[direction]
                    cell_strengths.append((direction, game_map.next_uncapped_strength_map[(y + dy) % game_map.height, (x + dx) % game_map.width, my_id]))
                cell_strengths.sort(key = lambda tup: tup[1])
                game_map.move_map[y, x] = cell_strengths[0][0]
            else:    # Moving out won't solve the problem. Will changing one of the incoming cells solve it?
                # Let's try changing a random piece to stay STILL instead.
                cell_to_change = random.choice(moving_into)
                game_map.move_map[cell_to_change[1], cell_to_change[0]] = STILL
        else:
            # We're moving out but still being overpopulated. Change a random cell
            cell_to_change = random.choice(moving_into)
            game_map.move_map[cell_to_change[1], cell_to_change[0]] = STILL
    
    return (cells_over_count)

def consolidate_strength():
    # Looks at border cells and sees if there is an opportunity to look N neighbors out to consolidate strength to capture a territory.
    border_squares = []
    
    def border_value(cell):
        # This is the heuristic that will be used to determine how valuable this border cell is and whether or not we should try to capture it.
        # 1: Check if it is going to be captured next turn.
        if len(game_map.get_all_moving_to(cell)) > 0:
            #logging.debug(str(game_map.get_all_moving_to(cell)))
            return 0
        elif cell.strength == 0:
            return cell.strength
        else:
            return cell.production / cell.strength
    
    
    for square in game_map:
        if game_map.is_npc_border(square):
            #logging.debug(str(square) + ":" + str(border_value(square)))
            if border_value(square) > 0:

                border_squares.append((square, border_value(square)))
    
    border_squares.sort(key = lambda x: x[1], reverse = True) # Sorts by all border cells which will not be taken next turn by the heuristic above.
    #logging.debug(str(border_squares))
    for border_square in border_squares:
        # For each border square, starting with the most valuable, let's see if we can capture it.
        friendly_neighbors = [x for x in game_map.neighbors(border_square[0]) if x.owner == my_id]
        #logging.debug("border_square: " + str(border_square))
        #logging.debug("friendly neigbors")
        available_strength = 0
        for f in friendly_neighbors:            
            #logging.debug(str(f) + "move_map: " + str(game_map.move_map[f.y, f.x]))
            if game_map.move_map[f.y, f.x] == STILL:
                #logging.debug(str(f))
                available_strength += f.strength
        
        # Case 1: This border square is bordered by enough squares that if multiple squares attacked it, it would be captured.
        # available_strength = sum(x.strength for x in friendly_neighbors if game_map.move_map[x.y, x.x] == STILL)
        if available_strength > border_square[0].strength:
            #logging.debug("Test1")
            attacking_strength = 0
            for f in friendly_neighbors:
                if (game_map.move_map[f.y, f.x] == STILL) and (attacking_strength < border_square[0].strength):
                    attacking_strength += f.strength
                    game_map.move_to_target(f, border_square[0], False)
        
        # Case 2: Ok, the surrounding squares aren't enough. Can we gather strength in an adjacent friendly square 1 space away, so that this can be captured next turn?
        else:
            #logging.debug("Test2")
            # Look at each friendly square.
            for f in friendly_neighbors:                
                # For EACH friendly neighbor, see if ITS neighbors are STILL and if so, can we move neighbors to attack this square the FOLLOWING turn?
                #logging.debug("test2: " + str(f))
                needed_strength = border_square[0].strength
                #logging.debug("test2 needed_strength:" + str(needed_strength))

                if game_map.move_map[f.y, f.x] == STILL:
                    needed_strength = needed_strength - f.strength - f.production    # We'll be producing this turn so we can reduce needed strength by production
                #logging.debug("test2 needed_strength2:" + str(needed_strength))
                # Do we have enough strengths in the neighbors?
                #f_neighbors = [x for x in game_map.neighbors(f) if x.owner == my_id]
                #f_neighbors = [x for x in f_neighbors if game_map.move_map[x.y, x.x] == STILL]
                f_neighbors = [x for x in game_map.neighbors(f) if x.owner == my_id]
                neighbor_strength = 0
                for f_n in f_neighbors:
                    #logging.debug("f_n: " + str(f_n) + "move map: " + str(game_map.move_map[f_n.y, f_n.x]))
                    if game_map.move_map[f_n.y, f_n.x] == STILL:
                        neighbor_strength += f_n.strength
                #logging.debug("neighbor_str: " + str(neighbor_strength))
                if neighbor_strength > needed_strength:
                    for f_n in f_neighbors:
                        if game_map.move_map[f_n.y, f_n.x] == STILL and (needed_strength > 0):
                            game_map.move_to_target(f_n, f, True)
                            needed_strength -= f_n.strength 


logging.basicConfig(filename='logging.log',level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)

ATTACK = 0
STOP_ATTACK = 1

Square = namedtuple("Square", "x y owner strength production")
Move = namedtuple("Move", "square direction")

def opposite_direction(direction):
    return (direction + 2) % 4 if direction != STILL else STILL

def grouper(iterable, n, fillvalue = None):
    # Collect data into fixed-length chunks or blocks
    # grouper("ABCDEFG", "3", "x") --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue = fillvalue)


    

class GameMap:

    def __init__(self, size_string, production_string, player_tag, map_string = None):
        self.width, self.height = tuple(map(int, size_string.split()))
        self.production = tuple(tuple(map(int, substring)) for substring in grouper(production_string.split(), self.width))
        self.contents = None
        self.starting_player_count = 0
        self.my_id = player_tag
        self.frame = 0
        
        self.owner_map = numpy.ones((self.height, self.width)) * -1
        self.strength_map = numpy.ones((self.height, self.width)) * -1
        self.production_map = numpy.ones((self.height, self.width)) * -1
        self.move_map = numpy.ones((self.height, self.width)) * -1
        
        self.projected_owner_map = numpy.ones((self.height, self.width)) * -1
        self.projected_strength_map = numpy.ones((self.height, self.width)) * -1
                
        self.get_frame(map_string)
        self.starting_player_count = len(set(square.owner for square in self)) - 1
        
    def get_frame(self, map_string = None):
        # Updates the map information form the latest frame provided by the game environment
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
        
        # This is then followed by WIDTH * HEIGHT integers, representing the strength values of the tiles in the map. 
        # It fills in the map in the same way owner values fill in the map.
        assert len(split_string) == self.width * self.height
                                          
        self.contents = [[Square(x, y, owner, strength, production)
            for x, owner, strength, production in zip(count(), owner_row, strength_row, production_row)]
            for y, owner_row, strength_row, production_row in zip(count(), grouper(owners, self.width), grouper(map(int,split_string), self.width), self.production)]
            
        # update the array maps
        for cell in chain.from_iterable(self.contents):
            self.owner_map[cell.y,cell.x] = cell.owner
            self.strength_map[cell.y,cell.x] = cell.strength
            self.production_map[cell.y,cell.x] = cell.production

        self.move_map = numpy.ones((self.height, self.width)) * -1
        
        #if self.starting_player_count == 0: self.starting_player_count = len(set(square.owner for square in self)) - 1
        #self.create_projection()
        self.frame += 1
    
    def __iter__(self):
        # Allows direct iteration over all squares in the GameMap instance
        return chain.from_iterable(self.contents)
        
    def neighbors(self, square, n = 1, include_self = False):
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            combos = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
        else:
            combos = ((dx, dy) for dy in range(-n, n+1) for dx in range(-n, n+1) if abs(dx) + abs(dy) <= n)
        return (self.contents[(square.y + dy) % self.height][(square.x + dx) % self.width] for dx, dy in combos if include_self or dx or dy)
        
    def inBounds(self, l):
        return l.x >= 0 and l.x < self.width and l.y >= 0 and l.y < self.height

    def is_border(self, square):
        # Looks at a square and sees if it's a border
        # look at all neighbors and see if the owner is != my_id
        for n in self.neighbors(square):
            if n.owner != self.my_id:
                return True
        return False        
       
    def is_npc_border(self, square):
        # Looks at a square and sees if it's an NPC border square
        # Defined as a square which is owned by 0 and has a neighbor of my_id
        if square.owner != 0: return False
        for n in self.neighbors(square):
            if n.owner == self.my_id:
                return True
        return False
            
    
    def get_distance(self, l1, l2):
        dx = abs(l1.x - l2.x)
        dy = abs(l1.y - l2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy

    def get_target(self, square, direction):
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
        return self.contents[(square.y + dy) % self.height][(square.x + dx) % self.width]
    
    def get_offset(self, direction):
        return ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]
    
    def get_coord(self, sx, sy, dx, dy):
        return ((sx + dx) % self.width, (sy + dy) % self.height)

    def create_projection(self):
        # V3 Note: I think the below is broken but right now nothing really uses this. Break this up into different pieces as well.
        # This will need to undergo a LOT of tweaking as we make this more accurate. 
        # For now, a basic implementation should hopefully provide a simplified view of a future state given the expected moves
        #temp_map = [[[0 for k in range(self.starting_player_count + 1)] for i in range(self.width)] for j in range(self.height)]
        temp_map = numpy.zeros((self.height, self.width, self.starting_player_count + 1))
        for x in range(self.width):
            for y in range(self.height):
                owner = self.owner_map[y,x]
                temp_map[y,x,owner] = self.strength_map[y,x]
                
                # 4. Add strength to pieces which choose to remain where they are.
                # Treat all cells that have a move value of -1 or 4 to be increasing in strength.
                # In practice, this is not true for enemy pieces, but for now, let's make this assumption                
                if self.move_map[y,x] == 4 or self.move_map[y,x] == -1:
                    temp_map[y,x,owner] += self.production_map[y,x] if owner > 0 else 0
                # 5. Simultaneously move (and combine if necessary) all player's pieces. The capping of strengths to 255 occurs here.
                else: 
                    for direction in range(0, 4):
                        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
                        temp_map[(y + dy) % self.height,(x + dx) % self.width,owner] += min(temp_map[(y + dy) % self.height,(x + dx) % self.width,owner] + self.strength_map[y][x], 255)
                        temp_map[y,x,owner] -= self.strength_map[y,x]
                    
        # 6. Simultaneously damage (and remove if damage equals or exceeds strength) all player's pieces. All pieces will output damage equivalent to their strength when starting this phase, and the damage will apply to all coinciding or adjacent enemy squares.                    
        #projected_power_map = [[[0 for k in range(self.starting_player_count + 1)] for i in range(self.width)] for j in range(self.height)]
        projected_power_map = numpy.zeros((self.height, self.width, self.starting_player_count + 1))
        #total_power_map = [[0 for i in range(self.width)] for j in range(self.height)]
        total_power_map = numpy.zeros((self.height, self.width))
        for x in range(self.width):
            for y in range(self.height):
                # Calculate the influence (aka projected power but I already used that name)
                combos = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
                for owner in range(self.starting_player_count + 1):
                    if owner > 0:
                        for (dx, dy) in combos:                    
                            projected_power_map[(y + dy) % self.height,(x + dx) % self.width,owner] += temp_map[y,x,owner]
                            total_power_map[(y + dy) % self.height,(x + dx) % self.width] += temp_map[y,x,owner]
                    else:
                        projected_power_map[y,x,owner] += temp_map[y,x,owner]
                        # Neutral territory decreases the strength of all units attacking it by its strength (is this accurate?)
                        # Code this up later.
                        
                # Now that we have the all projected power from all owners in a cell, we can calculate the winner
                # For each owner that actually is wanting to occupy this cell (or currently occupies it), calculate if the power is greater than the projected power of all other members.
                cell_owner = 0
                cell_strength = 0
                for owner in range(0, self.starting_player_count + 1):
                    # Special cases galore for neutral owners: TODO
                    if temp_map[y,x,owner] > 0:
                        attacking_power = total_power_map[y,x] - projected_power_map[y,x][owner]
                        if attacking_power < temp_map[y,x,owner]:
                        # Enemy attacking is not enough to dislodge the owner
                            cell_owner = owner
                            cell_strength = temp_map[y,x,owner] - attacking_power
                            break
                self.projected_owner_map[y,x] = cell_owner
                self.projected_strength_map[y,x] = cell_strength
    
    def calculate_uncapped_next_strength(self):
        # Given the move_map, calculate the uncapped strength in each cell.
        self.next_uncapped_strength_map = numpy.zeros((self.height, self.width, self.starting_player_count + 1))
        for x in range(self.width):
            for y in range(self.height):
                owner = self.owner_map[y,x]
                self.next_uncapped_strength_map[y,x,owner] = self.strength_map[y,x]
                
                # 4. Add strength to pieces which choose to remain where they are.
                # Treat all cells that have a move value of -1 or 4 to be increasing in strength.
                # In practice, this is not true for enemy pieces, but for now, let's make this assumption                
                if self.move_map[y,x] == 4 or self.move_map[y,x] == -1:
                    self.next_uncapped_strength_map[y,x,owner] += self.production_map[y,x] if owner > 0 else 0
                # 5. Simultaneously move (and combine if necessary) all player's pieces.
                else: 
                    direction = self.move_map[y,x]
                    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[int(direction)]
                    self.next_uncapped_strength_map[(y + dy) % self.height,(x + dx) % self.width,owner] += self.next_uncapped_strength_map[(y + dy) % self.height,(x + dx) % self.width,owner] + self.strength_map[y][x]
                    self.next_uncapped_strength_map[y,x,owner] -= self.strength_map[y,x]
        # At this point we have calculated for each player where the uncapped strengths are.
        
    def move_to_target(self, source, destination, through_friendly):
        # Source & Destinations are actual squares.
        
        # TODO: Return STILL if we shouldn't move. though then this shouldn't be called...
        
        dist_w = (source.x - destination.x) % self.width
        dist_e = (destination.x - source.x) % self.width
        dist_n = (source.y - destination.y) % self.height
        dist_s = (destination.y - source.y) % self.height
        
        # Prioritize in the following order:
        # 1: Move through OWN territory
        # 2: Move CLOSER to the destination
        # 3: Move through LOWER production square
        possible_moves = []
        possible_moves.append((NORTH, game_map.owner_map[(source.y - 1) % self.height, (source.x + 0) % self.width] == game_map.my_id, dist_n if dist_n > 0 else 999, game_map.production_map[(source.y - 1) % self.height, (source.x + 0) % self.width]))
        possible_moves.append((SOUTH, game_map.owner_map[(source.y + 1) % self.height, (source.x + 0) % self.width] == game_map.my_id, dist_s if dist_s > 0 else 999, game_map.production_map[(source.y + 1) % self.height, (source.x + 0) % self.width]))
        possible_moves.append((EAST, game_map.owner_map[(source.y + 0) % self.height, (source.x + 1) % self.width] == game_map.my_id, dist_e if dist_e > 0 else 999, game_map.production_map[(source.y + 0) % self.height, (source.x + 1) % self.width]))
        possible_moves.append((WEST, game_map.owner_map[(source.y + 0) % self.height, (source.x - 1) % self.width] == game_map.my_id, dist_w if dist_w > 0 else 999, game_map.production_map[(source.y + 0) % self.height, (source.x - 1) % self.width]))
        
        # Sort. Note sorts need to happen in reverse order of priority.
        random.shuffle(possible_moves) # Shuffle so we don't bias direction.
        possible_moves.sort(key = lambda x: x[3]) # Sort production, smaller is better
        possible_moves.sort(key = lambda x: x[2]) # Sort distance, smaller is better
        if through_friendly:
            possible_moves.sort(key = lambda x: x[1], reverse = True) # Sort owner, True = 1, False = 0
        #logging.debug(str(possible_moves))
        # The smallest move is the one we'll take.
        self.make_move(source, possible_moves[0][0])
        
        
        
    
    def get_all_moving_to(self, target):
        # Returns a list of all squares that are queued to move INTO the target square.
        combos = ((0, -1), (1, 0), (0, 1), (-1, 0))
        square_list = []
        for direction in range(0, 4):
            dx, dy = combos[direction]
            direction2 = self.move_map[(target.y + dy) % self.height, (target.x + dx) % self.width]
            if (direction2 + 2) % 4 == direction and direction2 != STILL and direction2 != -1:
                square_list.append(self.contents[(target.y + dy) % self.height][(target.x + dx) % self.width])
        return square_list        
    
    def make_move(self, square, direction):
        # Simulates a move, NOT simultaneous.
        # Keep this simple for now.
        self.move_map[square.y,square.x] = direction
        # Let's see if we can update the map...
        #self.create_projection()
        
    def get_moves(self):
        # Goes through self.move_map and creates a move object
        # TODO: This is redundant. We can just get all the moves from move_map and create it that way.
        move_list = []
        
        for sq in chain.from_iterable(self.contents):
            if sq.owner == self.my_id:
                move_list.append(Move(sq, self.move_map[sq.y,sq.x]))
        return (move_list)
        
    
                

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
    to_be_sent += '\n'
    sys.stdout.write(to_be_sent)
    sys.stdout.flush()

def get_string():
    return sys.stdin.readline().rstrip('\n')

def get_init():
    player_id = int(get_string())
    m = GameMap(get_string(), get_string(), player_id)  
    return (player_id, m)

def send_init(name):
    send_string(name)

def send_frame(moves):
    send_string(' '.join(str(move.square.x) + ' ' + str(move.square.y) + ' ' + str(translate_cardinal(move.direction)) for move in moves))

##################
# Main Game Loop #
##################

my_id, game_map = get_init()
send_init("shummie v3.3")


while True:
    game_map.get_frame()
    #logging.debug("Frame: " + str(game_map.frame) + "\n")
    # Have each individual square decide on their own movement
    square_move_list = []
    for square in game_map:
        if square.owner == game_map.my_id: 
            square_move_list.append(square)
    # Have smaller strength pieces move first. Mainly since otherwise especially for attacking, large pieces bounce back and forth when we want them to attack instead.
    square_move_list.sort(key = lambda x: x.strength)   

    percent_owned = len(square_move_list) / (game_map.width * game_map.height)

    for square in square_move_list:
        get_move(square)
    # Project the state of the board assuming for now that enemy pieces do not move    
    game_map.create_projection()    
    # Do stuff
    consolidate_strength()
    over_count = game_map.width * game_map.height
    
    new_over_count = prevent_overstrength()
    
    while new_over_count < over_count:
        over_count = new_over_count
        new_over_count = prevent_overstrength()
    
    moves = game_map.get_moves()
    
    send_frame(moves)
    
