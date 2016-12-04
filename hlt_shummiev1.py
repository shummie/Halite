import random
import math
import copy
from collections import namedtuple
from itertools import chain, zip_longest, count
import sys
import logging
import numpy

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
        self.myID = player_tag
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
            combos = ((0, 1), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
        else:
            combos = ((dx, dy) for dy in range(-n, n+1) for dx in range(-n, n+1) if abs(dx) + abs(dy) <= n)
        return (self.contents[(square.y + dy) % self.height][(square.x + dx) % self.width] for dx, dy in combos if include_self or dx or dy)
        
    def inBounds(self, l):
        return l.x >= 0 and l.x < self.width and l.y >= 0 and l.y < self.height

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
                combos = ((0, -11), (1, 0), (0, 1), (-1, 0), (0, 0)) # N, E, S, W, STILL
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
        
    def move_to_target(self, source, destination):
        # Source & Destinations are actual squares.
        delta_x = destination.x - source.x
        delta_y = destination.y - source.y
        
        # We'll move towards the X axis first
        if abs(delta_x) > 0:
            # if delta_x is < 0, then that means we need to move WEST. However, since the map wraps then it might be quicker to go EAST instead
            if delta_x < 0:
                if abs(delta_x) < self.width / 2:
                    self.make_move(source, WEST)
                else:
                    self.make_move(source, EAST)
            else:
                if abs(delta_x) < self.width / 2:
                    self.make_move(source, EAST)
                else:
                    self.make_move(source, WEST)
        elif abs(delta_y) > 0:
            if delta_y < 0:
                if abs(delta_y) < self.height / 2:
                    self.make_move(source, NORTH)
                else:
                    self.make_move(source, SOUTH)
            else:
                if abs(delta_y) < self.height / 2:
                    self.make_move(source, SOUTH)
                else:
                    self.make_move(source, NORTH)
    
    def get_all_moving_to(self, target):
        # Returns a list of all squares that are queued to move INTO the target square.
        combos = ((0, 1), (1, 0), (0, 1), (-1, 0))
        square_list = []
        for direction in range(0, 4):
            dx, dy = combos[direction]
            direction2 = self.move_map[(target.y + dy) % self.height, (target.x + dx) % self.width]
            if (direction2 + 2) % 4 == direction:
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
        move_list = []
        
        for sq in chain.from_iterable(self.contents):
            if sq.owner == self.myID:
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

def send_string(toBeSent):
    toBeSent += '\n'
    sys.stdout.write(toBeSent)
    sys.stdout.flush()

def get_string():
    return sys.stdin.readline().rstrip('\n')

def get_init():
    playerTag = int(get_string())
    m = GameMap(get_string(), get_string(), playerTag)  
    return (playerTag, m)

def send_init(name):
    send_string(name)

def send_frame(moves):
    send_string(' '.join(str(move.square.x) + ' ' + str(move.square.y) + ' ' + str(translate_cardinal(move.direction)) for move in moves))