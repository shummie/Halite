# ==============================================================================
# Imports
# ==============================================================================
import functools
import cProfile
import itertools
import logging
import math
import numpy as np
import random
import scipy.sparse
import sys
import time
from timeit import default_timer as timer
import copy

# ==============================================================================
# Variables
# ==============================================================================
botname = "shummie v63-5-1"
strength_buffer = 0
print_maps = False
profile = False
MAX_TURN_TIME = 1.25


def print_map(npmap, name):
    directory = "Maps/"
    if print_maps:
        np.savetxt(directory + name + str(game.frame) + ".txt", npmap)

# ==============================================================================
# Game Class
# ==============================================================================


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

        self.starting_player_count = np.amax(self.owner_map)  # Note, for range you'd need to increase the range by 1

        # Create the distance map
        self.create_one_time_maps()

        self.max_turns = 10 * ((self.width * self.height) ** 0.5)

        self.set_configs()

        # Send the botname
        send_string(botname)

    def __iter__(self):
        # Allows direct iteration over all squares
        return itertools.chain.from_iterable(self.squares)

    def get_frame(self, map_string=None):
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
        self.moving_into_map = np.zeros((self.width, self.height))

        self.frame += 1

    def send_frame(self):
        # Goes through each square and get the list of moves.
        move_list = []
        for sq in itertools.chain.from_iterable(self.squares):
            if sq.owner == self.my_id:
                if sq.strength == 0:  # Squares with 0 strength shouldn't move.
                    sq.move = 4
                if sq.move == -1:
                    # In the event we didn't actually assign a move, make sure it's coded to STILL
                    sq.move = 4
                move_list.append(sq)

        send_string(' '.join(str(square.x) + ' ' + str(square.y) + ' ' + str(translate_cardinal(square.move)) for square in move_list))

    def create_squares_list(self):
        self.squares = np.empty((self.width, self.height), dtype=np.object)
        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y] = Square(self, x, y, self.production_map[x, y])

        for x in range(self.width):
            for y in range(self.height):
                self.squares[x, y].after_init_update()

    def create_one_time_maps(self):
        # self.distance_map = self.create_distance_map()
        self.distance_map_no_decay = self.create_distance_map(1)

        self.production_map_01 = np.maximum(self.production_map, 0.1)
        self.production_map_1 = np.maximum(self.production_map, 1)

        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)

        self.create_dijkstra_maps()

        self.create_parity_maps()

    def create_parity_maps(self):
        self.even = np.zeros((self.width, self.height))
        self.odd = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                if (x + y) % 2 == 0:
                    self.even[x, y] = 1
                else:
                    self.odd[x, y] = 1

    def create_dijkstra_maps(self):
        h, w = self.height, self.width

        def get_cost_recov(cellnum):
            x = cellnum // h
            y = cellnum % h
            return self.strength_map_1[x, y] / (self.production_map_01[x, y] ** 1.1)

        dij_recov_costs = scipy.sparse.dok_matrix((w * h, w * h))

        for x in range(w):
            for y in range(h):
                coord = x * h + y

                dij_recov_costs[coord, ((x + 1) % w) * h + ((y + 0) % h)] = get_cost_recov(((x + 1) % w) * h + ((y + 0) % h))
                dij_recov_costs[coord, ((x - 1) % w) * h + ((y + 0) % h)] = get_cost_recov(((x - 1) % w) * h + ((y + 0) % h))
                dij_recov_costs[coord, ((x + 0) % w) * h + ((y + 1) % h)] = get_cost_recov(((x + 0) % w) * h + ((y + 1) % h))
                dij_recov_costs[coord, ((x + 0) % w) * h + ((y - 1) % h)] = get_cost_recov(((x + 0) % w) * h + ((y - 1) % h))

        self.dij_recov_cost, self.dij_recov_route = scipy.sparse.csgraph.dijkstra(dij_recov_costs, return_predecessors=True)

        self.dij_recov_distance_map = np.zeros((w, h, w, h))
        self.dij_recov_route_map = np.zeros((w, h, w, h), dtype=int)

        for x in range(self.width):
            for y in range(self.height):
                self.dij_recov_distance_map[x, y, :, :] = self.dij_recov_cost[x * h + y].reshape((w, h))
                self.dij_recov_route_map[x, y, :, :] = self.dij_recov_route[x * h + y].reshape((w, h))

    def create_distance_map(self, falloff=1):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.
        # Create the base map for 0, 0
        zero_zero_map = np.zeros((self.width, self.height), dtype=int)

        for x in range(self.width):
            for y in range(self.height):
                dist_x = min(x, -x % self.width)
                dist_y = min(y, -y % self.width)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        if falloff != 1:
            zero_zero_map = zero_zero_map ** falloff

        distance_map = np.zeros((self.width, self.height, self.width, self.height), dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)

        return distance_map

    def set_configs(self):
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 4), 9)
        self.pre_combat_threshold = -3
        self.combat_radius = 5
        self.production_cells_out = 12
        self.phase = 0

    def update_configs(self):
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 5), 5)
        # self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 4), 7)
        self.buildup_multiplier = self.buildup_multiplier - (self.distance_from_border ** 0.4)
        # self.combat_radius = int(min(max(5, self.percent_owned * self.width / 2), self.width // 2))
        self.combat_radius = 8

        if self.percent_owned > 0.6:
            self.buildup_multiplier -= 1
            self.pre_combat_threshold = 0
            self.combat_radius = 10

        elif self.my_production_sum / self.next_highest_production_sum > 1.1:
            self.buildup_multiplier += 1

        if np.sum(self.even * self.is_owned_map * self.strength_map) > np.sum(self.odd * self.is_owned_map * self.strength_map):
            self.parity = 0
        else:
            self.parity = 1


    def update(self):
        # start = time.time()
        self.update_maps()
        # end = time.time()
        # logging.debug("update_maps Frame: " + str(game.frame) + " : " + str(end - start))
        self.update_stats()
        self.update_configs()

    def update_maps(self):
        print_map(self.strength_map, "strength_map")
        print_map(self.production_map, "production_map")

        self.update_calc_maps()
        self.update_owner_maps()

        self.update_border_maps()
        # start = time.time()
        self.update_enemy_maps()
        # end = time.time()
        # logging.debug("update_enemymaps Frame: " + str(game.frame) + " : " + str(end - start))
        # start = time.time()
        self.update_value_production_map()
        # end = time.time()
        # logging.debug("update_valuemaps Frame: " + str(game.frame) + " : " + str(end - start))

        self.update_controlled_influence_production_maps()

    def update_calc_maps(self):
        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)

    def update_owner_maps(self):
        self.is_owned_map = np.zeros((self.width, self.height), dtype=int)
        self.is_neutral_map = np.zeros((self.width, self.height), dtype=int)
        self.is_enemy_map = np.zeros((self.width, self.height), dtype=int)

        self.is_owned_map[np.where(self.owner_map == self.my_id)] = 1
        self.is_neutral_map[np.where(self.owner_map == 0)] = 1
        self.is_enemy_map = 1 - self.is_owned_map - self.is_neutral_map

    def update_border_maps(self):
        self.border_map = np.zeros((self.width, self.height), dtype=int)
        self.combat_zone_map = np.zeros((self.width, self.height), dtype=int)

        self.border_map += self.is_owned_map
        self.border_map += roll_xy(self.is_owned_map, 0, 1)
        self.border_map += roll_xy(self.is_owned_map, 0, -1)
        self.border_map += roll_xy(self.is_owned_map, 1, 0)
        self.border_map += roll_xy(self.is_owned_map, -1, 0)

        self.border_map = np.minimum(self.border_map, 1)
        self.border_map -= self.is_owned_map

        border_squares_indices = np.transpose(np.nonzero(self.border_map))
        border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        self.distance_from_border = self.friendly_flood_fill_multiple_sources(border_squares, max(self.width, self.height))

        self.combat_zone_map = self.border_map * (self.strength_map == 0)

        if self.starting_player_count > 1 and np.sum(self.combat_zone_map) >= 1:  # Breaks in single player mode otherwise.
            combat_squares_indices = np.transpose(np.nonzero(self.combat_zone_map))
            combat_squares = [self.squares[c[0], c[1]] for c in combat_squares_indices]
            self.distance_from_combat_zone = self.friendly_flood_fill_multiple_sources(combat_squares, max(self.width, self.height))
            self.distance_from_combat_zone[self.distance_from_combat_zone == -1] = 9999
            # print_map(self.distance_from_combat_zone, "distance_from_combat_zone")
        else:
            self.distance_from_combat_zone = np.ones((self.width, self.height)) * 999

    def update_enemy_maps(self):
        self.enemy_strength_map = np.zeros((5, self.width, self.height))
        self.enemy_strength_map[0] = self.strength_map * self.is_enemy_map

        for x in range(len(self.enemy_strength_map)):
            self.enemy_strength_map[x] = spread_n(self.enemy_strength_map[0], x)
            print_map(self.enemy_strength_map[x], "enemy_str_" + str(x) + "_")

        self.own_strength_map = np.zeros((5, self.width, self.height))
        self.own_strength_map[0] = self.strength_map * self.is_owned_map

        for x in range(len(self.own_strength_map)):
            self.own_strength_map[x] = spread_n(self.own_strength_map[0], x)

    def update_value_production_map(self):

        self.base_value_map = np.divide(self.production_map_01, self.strength_map_1) * (self.is_neutral_map - self.combat_zone_map)
        # Each neutral cell gets assigned to the closest border non-combat cell
        global_targets_indices = np.transpose(np.nonzero(self.is_neutral_map - self.combat_zone_map))
        global_targets = [self.squares[c[0], c[1]] for c in global_targets_indices]
        self.global_border_map = np.zeros((self.width, self.height))

        gb_map = self.dij_recov_distance_map * (self.border_map - self.combat_zone_map)
        gb_map[gb_map == 0] = 9999

        for g in global_targets:
            if self.base_value_map[g.x, g.y] > 0.02:
                # Find the closest border square that routes to g
                gb_map = self.dij_recov_distance_map[g.x, g.y] * (self.border_map - self.combat_zone_map)
                gb_map[gb_map == 0] = 9999
                tx, ty = np.unravel_index(gb_map.argmin(), (self.width, self.height))
                self.global_border_map[tx, ty] += self.base_value_map[g.x, g.y] / self.dij_recov_distance_map[g.x, g.y, tx, ty]

        self.value_production_map = 1 / np.maximum(self.base_value_map + self.global_border_map * 1, 0.001)

        self.value_production_map *= (self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] == 0)
        self.value_production_map[self.value_production_map == 0] = 9999
        turns_left = self.max_turns - self.frame
        recover_threshold = turns_left * 0.6
        self.value_production_map[self.value_production_map > recover_threshold] == 9999

        avg_recov_threshold = 2
        # avg_map_recovery = np.sum(self.strength_map * (self.border_map - self.combat_zone_map)) / np.sum(self.production_map * (self.border_map - self.combat_zone_map))
        avg_map_recovery = np.sum(self.strength_map * self.border_map) / np.sum(self.production_map * self.border_map)
        self.value_production_map[self.value_production_map > (avg_recov_threshold * avg_map_recovery)] = 9999

    def update_controlled_influence_production_maps(self):
        max_distance = 6
        self.controlled_production_influence_map = np.zeros((max_distance + 1, self.width, self.height))
        self.controlled_production_influence_map[0] = self.production_map * (self.is_enemy_map + self.is_owned_map)
        for distance in range(1, max_distance + 1):
            self.controlled_production_influence_map[distance] = spread_n(self.controlled_production_influence_map[distance - 1], 1)
            self.controlled_production_influence_map[distance] = rebase_map(self.controlled_production_influence_map[distance - 1], False)

    def get_moves(self):
        # This is the main logic controlling code.
        # Find super high production cells
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_pre_combat_production()
        # 1 - Find combat zone cells and attack them.
        # start = time.time()
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_attack()
        # end = time.time()
        # logging.debug("get_move_attack Frame: " + str(game.frame) + " : " + str(end - start))
        # self.get_moves_prepare_strength()
        # 2 - Find production zone cells and attack them
        # start = time.time()
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_production()
        # end = time.time()
        # logging.debug("get production moves Frame: " + str(game.frame) + " : " + str(end - start))
        # 3 - Move all other unassigned cells.
        # start = time.time()
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_other()
        # end = time.time()
        # logging.debug("get other moves Frame: " + str(game.frame) + " : " + str(end - start))

    def get_pre_combat_production(self):
        # In the event we are trying to fight in a very high production zone, reroute some attacking power to expand in this area.
        potential_targets_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices if (1 / self.base_value_map[c[0], c[1]] < self.pre_combat_threshold)]
        if len(potential_targets) == 0:
            return

        potential_targets.sort(key=lambda sq: 1 / self.base_value_map[sq.x, sq.y])

        best_target_value = 1 / self.base_value_map[potential_targets[0].x, potential_targets[0].y]
        # anything with X of the best_value target should be considered. Let's set this to 4 right now.
        while len(potential_targets) > 0 and 1 / self.base_value_map[potential_targets[0].x, potential_targets[0].y] <= (best_target_value + 1):
            target = potential_targets.pop(0)
            self.attack_cell(target, 2)

    def get_moves_attack(self):
        # Attempts to attack all border cells that are in combat
        combat_zone_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]

        combat_zone_squares.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
        combat_zone_squares.sort(key=lambda x: self.enemy_strength_map[1, x.x, x.y], reverse=True)

        # TODO: Should sort by amount of overkill damage possible.
        for square in combat_zone_squares:
            self.attack_cell(square, 1)

        self.get_moves_breakthrough()
        # Get a list of all squares within 5 spaces of a combat zone.
        # TODO: This causes bounciness, i should probably do a floodfill of all combat zone squares instead?
        combat_distance_matrix = self.friendly_flood_fill_multiple_sources(combat_zone_squares, self.combat_radius)
        # combat_distance_matrix[combat_distance_matrix == -1] = 0
        # combat_distance_matrix[combat_distance_matrix == 1] = 0
        combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]
        combat_squares.sort(key=lambda x: x.strength, reverse=True)
        combat_squares.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
        combat_squares.sort(key=lambda x: self.enemy_strength_map[1, x.x, x.y], reverse=True)


        print_map(combat_distance_matrix, "combat_distance_matrix_")

        for square in combat_squares:
            if (square.strength > 0) and (combat_distance_matrix[square.x, square.y] == 1) and (square.move == -1 or square.move == STILL):
                targets = []
                alt_targets = []
                for n in square.neighbors:
                    if n.owner == 0 and n.strength == 0:
                        targets.append(n)
                    elif n.owner == self.my_id:
                        alt_targets.append(n)
                targets.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
                alt_targets.sort(key=lambda x: x.strength)
                success = False
                for t in targets:
                    success = self.move_square_to_target_simple(square, t, False)
                    if success:
                        break
                if not success:
                    for t in targets:
                        success = self.move_square_to_target_simple(square, t, True)
                        if success:
                            break
            elif (square.strength > (square.production * (self.buildup_multiplier[square.x, square.y] + self.distance_from_combat_zone[square.x, square.y]))) and ((square.x + square.y) % 2 == self.parity) and square.move == -1 and square.moving_here == []:
                self.move_towards_map_old(square, combat_distance_matrix)

            else:
                if combat_distance_matrix[square.x, square.y] > 1:
                    self.make_move(square, STILL, None)

    def find_nearest_non_owned_border(self, square):
        current_distance = self.distance_from_border[square.x, square.y]
        # Todo, minor optimization by moving to lower production square if possible.
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
                elif distance_map[n.x, n.y] < current_distance:
                    queue.append(n)
        random.shuffle(targets)
        target = targets.pop(0)
        # success = self.move_square_to_target(square, target, True)
#        while len(targets) > 0:
#            target = targets.pop(0)
#            success = self.move_square_to_target(square, target, True)
#            if success:
#                return

    def move_towards_map_old(self, square, distance_map, through_friendly=True):
        current_distance = distance_map[square.x, square.y]
        possible_moves = []
        for n in square.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if distance_map[n.x, n.y] <= current_distance - 1:
                    possible_moves.append(n)
        if len(possible_moves) > 0:
            random.shuffle(possible_moves)
            possible_moves.sort(key=lambda sq: self.enemy_strength_map[4, sq.x, sq.y], reverse=True)
            possible_moves.sort(key=lambda sq: sq.production)
            self.move_square_to_target(square, possible_moves[0], True)

    def get_moves_prepare_strength(self):
        # Attempts to build up strength prior to an immediate engagement, only if we aren't already in combat
        border_prepare_indices = np.transpose(np.nonzero(self.border_map * self.enemy_strength_map[1] > 0))
        enemy_border_squares = [self.squares[c[0], c[1]] for c in border_prepare_indices]

        if len(enemy_border_squares) > 0:
            combat_distance_matrix = self.friendly_flood_fill_multiple_sources(enemy_border_squares, 5)
            combat_distance_matrix[combat_distance_matrix == -1] = 0
            combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]

            for square in combat_squares:
                if (self.distance_from_border[square.x, square.y] > 3) and (square.strength > square.production * self.buildup_multiplier[square.x, square.y] + 5) and ((square.x + square.y) % 2 == self.parity) and square.move == -1 and square.moving_here == []:
                    self.move_towards_map_old(square, combat_distance_matrix)
                elif (square.strength >= 240) and (self.own_strength_map[2, square.x, square.y] >= 750) and (combat_distance_matrix[square.x, square.y] == 1):
                    # Attack
                    targets = []
                    for n in square.neighbors:
                        if combat_distance_matrix[n.x, n.y] == 0:
                            targets.append(n)
                    targets.sort(key=lambda n: self.enemy_strength_map[1, n.x, n.y], reverse=True)
                    self.move_square_to_target_simple(square, targets[0], False)
                elif square.move == -1:
                    self.make_move(square, STILL, None)

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
        potential_targets.sort(key=lambda x: x[0].strength)
        potential_targets.sort(key=lambda x: x[1] + (x[2] * 1))

        # Keep only the top 80ile?
        percentile = 0.85
        cutoff = int(len(potential_targets) * percentile)
        potential_targets = potential_targets[:cutoff]
        remove_targets = potential_targets[cutoff:]
        for t in remove_targets:
            self.value_production_map[t[0].x, t[0].y] = 9999

        while len(potential_targets) > 0:
            if (timer() - game.start) > MAX_TURN_TIME:
                return
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
        # logging.debug(str(self.own_strength_map[4]))
        for square in potential_squares:
            if self.own_strength_map[4, square.x, square.y] > 750 and (self.own_strength_map[4, square.x, square.y] > 1.5 * self.enemy_strength_map[4, square.x, square.y]):
                self.attack_cell(square, 1)

    def get_moves_other(self):
        idle_squares_indices = np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))
        idle_squares = [self.squares[c[0], c[1]] for c in idle_squares_indices]

        if len(idle_squares) == 0:
            return

        # Move squares closer to the border first.
        idle_squares.sort(key=lambda sq: sq.strength, reverse = True)
        idle_squares.sort(key=lambda sq: self.distance_from_border[sq.x, sq.y])

        for square in idle_squares:
            if (timer() - game.start) > MAX_TURN_TIME:
                return
            if square.strength > square.production * self.buildup_multiplier[square.x, square.y] and square.move == -1 and square.moving_here == []:
                if self.percent_owned > 0.75:
                    self.find_nearest_non_owned_border(square)
                else:
                    if np.sum(self.is_owned_map) > 120:
                        d_map = self.distance_map_no_decay[square.x, square.y]
                    else:
                        d_map = self.flood_fill_to_border([square])
                        d_map[d_map == -1] = 9999
                    # Move towards the closest border
                    value_map = (self.value_production_map + d_map[square.x, square.y] * 1) * self.border_map
                    value_map[np.nonzero(self.combat_zone_map)] = 0
                    value_map += d_map[square.x, square.y] * 0.66 * self.combat_zone_map
                    value_map -= self.controlled_production_influence_map[5, square.x, square.y] * 5 * self.combat_zone_map
                    value_map[value_map == 0] = 9999
                    tx, ty = np.unravel_index(value_map.argmin(), (self.width, self.height))
                    target = self.squares[tx, ty]

                    # We're targeting either a combat square, or a production square. Don't move towards close production squares.
                    if self.distance_between(square, target) < 6 and self.distance_from_combat_zone[square.x, square.y] < 7:
                        if (square.x + square.y) % 2 != self.parity:
                            continue

                    if (self.enemy_strength_map[3, square.x, square.y] > 0) and (((square.x + square.y) % 2) != (self.parity)):
                        self.make_move(square, STILL, None)
                    elif self.combat_zone_map[tx, ty]:
                        if max(self.width, self.height) > 44 and self.distance_between(square, target) > 10:
                            self.find_nearest_non_owned_border(square)
                        elif self.distance_between(square, target) > 14:
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

    def attack_cell(self, target, max_cells_out, min_cells_out=1):
        # Attempts to coordinate attack to a specific cell.
        cells_out = min_cells_out

        while cells_out <= max_cells_out:
            # If we're trying to attack a combat zone cell, this isn't the function to do it. cancel.
            if cells_out > 1 and self.combat_zone_map[target.x, target.y]:
                return False

            if target.strength == 0:  # or target.production >= 5:  # or self.phase == 0:
                free_squares = self.is_owned_map * (self.move_map == -1)
            else:
                free_squares = self.is_owned_map * (self.move_map == -1) * (self.strength_map >= self.buildup_multiplier * self.production_map) * (self.moving_into_map == 0)
            target_distance_matrix = self.friendly_flood_fill(target, cells_out)
            target_distance_matrix[target_distance_matrix == -1] = 0
            target_distance_matrix = target_distance_matrix * free_squares
            available_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix, 1))

            target_distance_matrix_production = cells_out - target_distance_matrix
            target_distance_matrix_production[target_distance_matrix_production == cells_out] = 0  # Cells furthest out would be moving so no production
            target_distance_matrix_production = target_distance_matrix_production * free_squares
            available_production = np.sum(self.production_map * target_distance_matrix_production)

            if available_strength + available_production > target.strength + 0:
                attacking_cells_indices = np.transpose(np.nonzero(target_distance_matrix > 0))
                attacking_cells = [self.squares[c[0], c[1]] for c in attacking_cells_indices]

                still_cells = []
                if cells_out > 1:
                    still_cells_indices = np.transpose(np.nonzero(target_distance_matrix_production > 0))
                    still_cells = [self.squares[c[0], c[1]] for c in still_cells_indices]
                moving_cells = list(set(attacking_cells) - set(still_cells))

                for square in still_cells:
                    self.make_move(square, STILL, None)

                still_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix_production, 1))
                needed_strength_from_movers = target.strength - available_production - still_strength + 1

                if needed_strength_from_movers > 0:
                    # Handle movement here
                    moving_cells.sort(key=lambda x: x.strength, reverse=True)
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

        if direction == -1:  # Reset the square move
            if square.target is not None:
                square.target.moving_here.remove(square)
                self.moving_into_map[square.target.x, square.target.y] -= 1
                square.target = None
                square.far_target = None
            square.move = -1
            square.far_target = None
            return

        if square.move != -1:
            if square.target is not None:
                square.target.moving_here.remove(square)
                self.moving_into_map[square.target.x, square.target.y] -= 1
                square.target = None
            square.far_target = None

        square.move = direction
        if direction != STILL:
            square.target = square.neighbors[direction]
            square.target.moving_here.append(square)
            self.moving_into_map[square.target.x, square.target.y] += 1
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
        path_choices.sort(key=lambda x: x[1].production)

        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength  # + target.production
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
                        self.make_move(source, direction, destination)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        n_neighbors.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for n_d in n_directions:
                            n = target.neighbors[n_d]
                            if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength  # + n.production
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

        if ns_move is None and ew_move is None:
            return False

        path_choices = []
        if ns_move is None:
            path_choices.append(ew_move)
        elif ew_move is None:
            path_choices.append(ns_move)
        elif ns_swap is True and ew_swap is False:
            path_choices.append(ew_move)
            path_choices.append(ns_move)
        elif ns_swap is False and ew_swap is True:
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
                    future_strength = target.strength  # + target.production
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
                    future_strength = target.strength  # + target.production
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
                        self.make_move(source, direction, destination)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        n_neighbors.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for n_d in n_directions:
                            n = target.neighbors[n_d]
                            if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength  # + n.production
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

    def flood_fill_to_border(self, sources):
        q = sources
        distance_matrix = np.ones((self.width, self.height)) * -1
        if len(sources) == 0:
            return distance_matrix

        for sq in sources:
            distance_matrix[sq.x, sq.y] = 0

        while len(q) > 0:
            c = q.pop(0)
            c_dist = distance_matrix[c.x, c.y]
            if c.owner == self.my_id:
                for n in c.neighbors:
                    if distance_matrix[n.x, n.y] == -1 or distance_matrix[n.x, n.y] > (c_dist + 1):
                        distance_matrix[n.x, n.y] = c_dist + 1
                        q.append(n)

        return distance_matrix

    def flood_fill_until_target(self, source, destination, friendly_only):
        # Does a BFS flood fill to find shortest distance from source to target.
        # Starts the fill AT destination and then stops once we hit the target.
        q = [destination]
        distance_matrix = np.ones((self.width, self.height), dtype=int) * -1
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
        distance_matrix = np.ones((self.width, self.height), dtype=int) * -1
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

    def friendly_flood_fill_multiple_sources(self, sources, max_distance=999):
        # Returns a np.array((self.width, self.height)) that contains the distance to the target by traversing through friendly owned cells only.
        # q is a queue(list) of items (cell, distance). sources is a list that contains the source cells.
        q = sources
        distance_matrix = np.ones((self.width, self.height), dtype=int) * -1
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

    def friendly_flood_fill_multiple_sources_cells_out(self, sources, max_distance=999):
        # Returns a np.array((self.width, self.height)) that contains the distance to the target by traversing through friendly owned cells only.
        # q is a queue(list) of items (cell, distance). sources is a list that contains the source cells.
        q = sources
        distance_matrix = np.ones((self.width, self.height), dtype=int) * -1
        for source in q:
            distance_matrix[source.x, source.y] = 0

        while len(q) > 0:
            current = q.pop(0)
            current_distance = distance_matrix[current.x, current.y]
            for neighbor in current.neighbors:
                if (distance_matrix[neighbor.x, neighbor.y] == -1 or distance_matrix[neighbor.x, neighbor.y] > (current_distance + 1)) and neighbor.owner == self.my_id:
                    distance_matrix[neighbor.x, neighbor.y] = current_distance + 1
                    if current_distance < max_distance - 1:
                        current_distance += 1
                        q.append(neighbor)

        return distance_matrix

    def non_friendly_flood_fill_multiple_sources(self, sources, max_distance=999):
        # Returns a np.array((self.width, self.height)) that contains the distance to the target by traversing through non owned cells only.
        # q is a queue(list) of items (cell, distance). sources is a list that contains the source cells.
        q = sources
        distance_matrix = np.ones((self.width, self.height), dtype=int) * -1
        for source in q:
            distance_matrix[source.x, source.y] = 0

        while len(q) > 0:
            current = q.pop(0)
            current_distance = distance_matrix[current.x, current.y]
            for neighbor in current.neighbors:
                if (distance_matrix[neighbor.x, neighbor.y] == -1 or distance_matrix[neighbor.x, neighbor.y] > (current_distance + 1)) and neighbor.owner != self.my_id:
                    distance_matrix[neighbor.x, neighbor.y] = current_distance + 1
                    if current_distance < max_distance - 1:
                        q.append(neighbor)

        return distance_matrix

    def last_resort_strength_check(self):
        # Calculates the projected strength map and identifies squares that are violating it.
        # Ignore strength overloads due to production for now
        # Validate moves
        projected_strength_map = np.zeros((self.width, self.height), dtype=int)
        # We only care about our moves.
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.my_id:
                if square.move == -1 or square.move == STILL:
                    projected_strength_map[square.x, square.y] += square.strength  # + square.production
                else:
                    dx, dy = get_offset(square.move)
                    projected_strength_map[(square.x + dx) % self.width, (square.y + dy) % self.height] += square.strength

        # Get a list of squares that are over the cap
        violation_indices = np.transpose(np.nonzero((projected_strength_map > 255 + strength_buffer)))
        violation_squares = [self.squares[c[0], c[1]] for c in violation_indices]
        violation_count = len(violation_squares)

        violation_squares.sort(key=lambda sq: sq.strength, reverse=True)
        violation_squares.sort(key=lambda sq: self.distance_from_combat_zone[sq.x, sq.y])

        for square in violation_squares:
            if square.owner == self.my_id and (square.move == -1 or square.move == STILL):
                # We can try to move this square to an neighbor.
                possible_paths = []
                for d in range(0, 4):
                    # Move to the lowest strength neighbor. this might cause a collision but we'll resolve it with multiple iterations
                    n = square.neighbors[d]
                    if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                        possible_paths.append((d, n, projected_strength_map[n.x, n.y]))
                    else:
                        # Try attacking a bordering cell
                        if (square.strength > (2 * n.strength)) and (n.production > 1):
                            possible_paths.append((d, n, n.strength))

                possible_paths.sort(key=lambda x: x[2])
                possible_paths.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                # Force a move there
                self.make_move(square, d, n)
            else:
                # We aren't the problem. one of the squares that's moving here is going to collide with us.
                # How do we resolve this?
                options_list = []
                for n in square.neighbors:
                    if n.owner == self.my_id:
                        options_list.append((n, projected_strength_map[n.x, n.y]))
                options_list.sort(key=lambda x: x[1])
                # Let's try having the smallest one stay still instead
                for opt in options_list:
                    self.make_move(opt[0], STILL, None)
                # self.make_move(options_list[0][0], STILL, None)

        return violation_count

    def stop_swaps(self):
        # Check if two squares are swapping places for no reason.
        for x in range(self.width):
            for y in range(self.height):
                if self.is_owned_map[x, y]:
                    s = self.squares[x, y]
                    if s.target is not None:
                        if s.target in s.moving_here:
                            if abs(s.strength - s.target.strength) < 165:
                                self.make_move(s.target, STILL, None)
                                self.make_move(s, STILL, None)

    def check_parity(self):
        indices = np.transpose(np.nonzero((self.is_owned_map * self.enemy_strength_map[3])))
        squares = [self.squares[c[0], c[1]] for c in indices]
        squares.sort(key=lambda sq: sq.strength, reverse=True)

        for s in squares:
            if (self.enemy_strength_map[2, s.x, s.y] == 0) and (s.x + s.y) % 2 != self.parity and (s.move != STILL and s.move != -1):
                self.make_move(s, STILL, None)
                future_strength = s.strength + sum(x.strength for x in s.moving_here)
                if future_strength > 255 + strength_buffer:
                    s.moving_here.sort(key=lambda x: x.strength)
                    while future_strength > 255:
                        future_strength -= s.moving_here[0].strength
                        self.make_move(s.moving_here[0], STILL, None)
            elif (self.enemy_strength_map[2, s.x, s.y] > 0) and (s.move == STILL or s.move == -1):
                # Try to capture a neutral cell
                neutral_targets = []
                friendly_targets = []
                near_combat = False
                for t in s.neighbors:
                    if t.owner == self.my_id:
                        friendly_targets.append(t)
                    else:
                        if t.strength == 0:
                            near_combat = True
                        neutral_targets.append(t)
                friendly_targets.sort(key=lambda x: sum(y.strength for y in x.moving_here))
                success = False
                if near_combat:
                    for t in friendly_targets:
                        future_strength = sum(x.strength for x in t.moving_here) + t.strength if (t.move == STILL or t.move == -1) else 0
                        if future_strength + s.strength <= 255 + strength_buffer:
                            success = self.move_square_to_target_simple(s, t, True)
                            if success:
                                break
                    if not success:
                        neutral_targets.sort(key=lambda x: sum(y.strength for y in x.moving_here))
                        for t in neutral_targets:
                            future_strength = sum(x.strength for x in t.moving_here)
                            if future_strength + s.strength <= 255 + strength_buffer:
                                success = self.move_square_to_target_simple(s, t, False)
                                if success:
                                    break

    def update_stats(self):
        # Updates various stats used for tracking
        self.turns_left = self.max_turns - self.frame
        self.percent_owned = np.sum(self.is_owned_map) / (self.width * self.height)
        self.production_values = [0]
        for i in range(1, self.starting_player_count + 1):
            self.production_values.append(np.sum(self.production_map * (self.owner_map == i)))
        self.my_production_sum = self.production_values[self.my_id]
        temp_production_sum = copy.copy(self.production_values)
        temp_production_sum.pop(self.my_id)
        temp_production_sum.pop(0)
        self.next_highest_production_sum = max(temp_production_sum)

# ==============================================================================
# Square class
# ==============================================================================


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
        self.neighbors = [self.north, self.east, self.south, self.west]  # doesn't include self

    def get_neighbors(self, n=1, include_self=False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            if not include_self:
                return self.neighbors

        combos = ((dx, dy) for dy in range(-n, n + 1) for dx in range(-n, n + 1) if abs(dx) + abs(dy) <= n)
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


def opposite_direction(direction):
    return (direction + 2) % 4 if direction != STILL else STILL


def roll_x(M, x):
    return np.roll(M, x, 0)


def roll_y(M, y):
    return np.roll(M, y, 1)


def roll_xy(M, x, y):
    return np.roll(np.roll(M, x, 0), y, 1)


def spread_n(M, n, decay=0, include_self=True):
    # Takes a matrix M, and then creates an influence map by offsetting by N in every direction.
    # Decay function is currently of the form exp(-decay * distance)
    if include_self is True:
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


def spread(M, decay=0, include_self=True):
    # For now to save time, we'll use game_map.distance_map and assume that we'll always be using the same falloff distances to calculate offsets.

    # Takes the matrix M and then for each point (x, y), calculate the product of the distance map and the decay factor.
    decay_map = np.exp(np.multiply(game.distance_map, -decay))

    spread_map = np.sum(np.multiply(decay_map, M), (2, 3))
    return spread_map


def get_all_d_away(d):
    combos = []
    for x in range(0, d + 1):
        x_vals = list(set([x, -x]))
        y_vals = list(set([d - x, -(d - x)]))
        combos.extend(list(itertools.product(x_vals, y_vals)))
    return list(set(combos))


def distance_from_owned(M, mine):
    # Returns the minimum distance to get to any point if already at all points in xys using 4D array M
    return np.apply_along_axis(np.min, 0, M[np.nonzero(mine)])


def rebase_map(map_a, total=True):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.width * self.height
    # If Total = False, rebases to the # of non-zero squares
    if total:
        size = functools.reduce(lambda x, y: x * y, map_a.shape)
    else:
        size = np.sum(map_a != 0)
    factor = size / np.sum(map_a)
    return np.multiply(map_a, factor)

# ==============================================================================
# Functions for communicating with the Halite game environment (formerly contained in separate module networking.py
# ==============================================================================


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

# ==============================================================================
# Game Loop
# ==============================================================================


def game_loop():
    game.start = timer()

    game.update()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    game.get_moves()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    game.stop_swaps()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        collision_check = game.last_resort_strength_check()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    game.check_parity()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        collision_check = game.last_resort_strength_check()

# #####################
# Game run-time code #
# #####################


logging.basicConfig(filename='logging.log', level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)
directions = [NORTH, EAST, SOUTH, WEST, STILL]

if (profile):
    pr = cProfile.Profile()
    pr.enable()

game = Game()

while True:

    game.get_frame()
    # logging.debug("Frame: " + str(game.frame))
    game_loop()
    game.send_frame()

    if profile and game.frame == 199:
        pr.disable()
        pr.dump_stats("test.prof")
