# ==============================================================================
# Imports
# ==============================================================================
import functools
from functools import wraps
import cProfile
import itertools
import logging
import math
import numpy as np
import scipy.sparse
import sys
import time
from timeit import default_timer as timer
import copy

# ==============================================================================
# Variables
# ==============================================================================
botname = "shummie v69"
print_maps = False
print_times = False
profile = False
MAX_TURN_TIME = 1.35


def print_map(npmap, name):
    directory = "Maps/"
    if print_maps:
        np.savetxt(directory + name + str(game.frame - 1) + ".txt", npmap)


def timethis(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        if print_times:
            logging.debug("Frame: {0} {1} secs to run func: {2}".format(args[0].frame, end - start, f.__name__))
        return result
    return wrap

# ==============================================================================
# Game Class
# ==============================================================================


class Game:

    def __init__(self):
        self.my_id = int(get_string())
        map_size_string = get_string()
        self.w, self.h = tuple(map(int, map_size_string.split()))

        production_map_string = get_string()
        self.production_map = np.array(list(map(int, production_map_string.split()))).reshape((self.h, self.w)).transpose()

        self.create_squares_list()
        self.frame = -1
        self.max_turns = 10 * ((self.w * self.h) ** 0.5)

        self.get_frame()

        self.starting_player_count = np.amax(self.owner_map)  # Note, for range you'd need to increase the range by 1

        self.create_one_time_maps()

        self.set_configs()

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
        while len(owners) < self.w * self.h:
            counter = int(split_string.pop(0))
            owner = int(split_string.pop(0))
            owners.extend([owner] * counter)
        assert len(owners) == self.w * self.h

        self.owner_map = np.array(owners).reshape((self.h, self.w)).transpose()

        # This is then followed by WIDTH * HEIGHT integers, representing the strength values of the tiles in the map.
        # It fills in the map in the same way owner values fill in the map.
        assert len(split_string) == self.w * self.h
        str_list = list(map(int, split_string))

        self.strength_map = np.array(str_list).reshape((self.h, self.w)).transpose()

        # Update all squares
        for x in range(self.w):
            for y in range(self.h):
                self.squares[x, y].update(self.owner_map[x, y], self.strength_map[x, y])

        # Reset the move_map
        self.move_map = np.ones((self.w, self.h)) * -1  # Could possibly expand this in the future to consider enemy moves...
        self.moving_into_map = np.zeros((self.w, self.h))

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
        self.squares = np.empty((self.w, self.h), dtype=np.object)
        for x in range(self.w):
            for y in range(self.h):
                self.squares[x, y] = Square(self, x, y, self.production_map[x, y])

        for x in range(self.w):
            for y in range(self.h):
                self.squares[x, y].after_init_update()

    def set_configs(self):
        self.str_cap = 255
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 4), 9)
        self.pre_combat_threshold = -3
        self.combat_radius = 5
        self.production_cells_out = 12
        self.phase = 0

    def update_configs(self):
        self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 5), 5)
        # self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 4), 7)
        self.buildup_multiplier = self.buildup_multiplier - (self.distance_from_border ** 0.4)
        # self.combat_radius = int(min(max(5, self.percent_owned * self.w / 2), self.w // 2))
        self.combat_radius = 6

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

        if self.percent_owned > 0.10:
            self.buildup_multiplier += 4

        self.buildup_multiplier = np.minimum(self.buildup_multiplier, 230 / self.production_map_1)

    def create_one_time_maps(self):
        self.distance_map_no_decay = self.create_distance_map(1)

        self.production_map_01 = np.maximum(self.production_map, 0.1)
        self.production_map_1 = np.maximum(self.production_map, 1)

        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)

        self.create_dijkstra_maps()
        self.create_parity_maps()

    @timethis
    def create_distance_map(self, falloff=1):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.
        # Create the base map for 0, 0
        zero_zero_map = np.zeros((self.w, self.h), dtype=int)

        for x in range(self.w):
            for y in range(self.h):
                dist_x = min(x, -x % self.w)
                dist_y = min(y, -y % self.w)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        if falloff != 1:
            zero_zero_map = zero_zero_map ** falloff

        distance_map = np.zeros((self.w, self.h, self.w, self.h), dtype=int)
        for x in range(self.w):
            for y in range(self.h):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)

        return distance_map

    @timethis
    def create_dijkstra_maps(self):
        h, w = self.h, self.w
        cost_recov_map = self.strength_map_1 / self.production_map_01

        def get_cost_recov(cellnum):
            x = cellnum // h
            y = cellnum % h
            return cost_recov_map[x, y]

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

        for x in range(self.w):
            for y in range(self.h):
                self.dij_recov_distance_map[x, y, :, :] = self.dij_recov_cost[x * h + y].reshape((w, h))
                self.dij_recov_route_map[x, y, :, :] = self.dij_recov_route[x * h + y].reshape((w, h))

    def create_parity_maps(self):
        self.even = np.zeros((self.w, self.h))

        for x in range(self.w):
            for y in range(self.h):
                self.even[x, y] = 0 if (x + y) % 2 else 1
        self.odd = 1 - self.even

    def update(self):

        self.update_maps()
        self.update_stats()
        self.update_configs()

    @timethis
    def update_maps(self):
        print_map(self.strength_map, "strength_map")
        print_map(self.production_map, "production_map")

        self.update_calc_maps()
        self.update_owner_maps()
        self.update_border_maps()
        self.update_enemy_maps()
        self.update_neutral_map()
        self.update_controlled_influence_production_maps()

        self.update_value_production_map()

    def update_calc_maps(self):
        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)

    def update_owner_maps(self):
        self.is_owned_map = np.zeros((self.w, self.h), dtype=int)
        self.is_neutral_map = np.zeros((self.w, self.h), dtype=int)
        self.is_enemy_map = np.zeros((self.w, self.h), dtype=int)

        self.is_owned_map[np.where(self.owner_map == self.my_id)] = 1
        self.is_neutral_map[np.where(self.owner_map == 0)] = 1
        self.is_enemy_map = 1 - self.is_owned_map - self.is_neutral_map

    @timethis
    def update_border_maps(self):
        self.border_map = np.zeros((self.w, self.h), dtype=int)
        self.combat_zone_map = np.zeros((self.w, self.h), dtype=int)

        self.border_map += self.is_owned_map
        self.border_map += roll_xy(self.is_owned_map, 0, 1)
        self.border_map += roll_xy(self.is_owned_map, 0, -1)
        self.border_map += roll_xy(self.is_owned_map, 1, 0)
        self.border_map += roll_xy(self.is_owned_map, -1, 0)

        self.border_map = np.minimum(self.border_map, 1)
        self.border_map -= self.is_owned_map

        border_squares_indices = np.transpose(np.nonzero(self.border_map))
        border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        self.distance_from_border = self.flood_fill(border_squares, max(self.w, self.h), True)

        self.combat_zone_map = self.border_map * (self.strength_map == 0)

        if self.starting_player_count > 1 and np.sum(self.combat_zone_map) >= 1:  # Breaks in single player mode otherwise.
            combat_squares_indices = np.transpose(np.nonzero(self.combat_zone_map))
            combat_squares = [self.squares[c[0], c[1]] for c in combat_squares_indices]
            self.distance_from_combat_zone = self.flood_fill(combat_squares, max(self.w, self.h), True)
            self.distance_from_combat_zone[self.distance_from_combat_zone == -1] = 9999
            print_map(self.distance_from_combat_zone, "distance_from_combat_zone")
        else:
            self.distance_from_combat_zone = np.ones((self.w, self.h)) * 999

    @timethis
    def update_enemy_maps(self):
        self.enemy_strength_map = np.zeros((5, self.w, self.h))
        self.enemy_strength_map[0] = self.strength_map * self.is_enemy_map
        self.enemy_strength_map[0] += self.is_enemy_map * 0.001

        for x in range(len(self.enemy_strength_map)):
            self.enemy_strength_map[x] = spread_n(self.enemy_strength_map[0], x)
            print_map(self.enemy_strength_map[x], "enemy_str_" + str(x) + "_")

        self.own_strength_map = np.zeros((8, self.w, self.h))
        self.own_strength_map[0] = self.strength_map * self.is_owned_map

        for x in range(len(self.own_strength_map)):
            self.own_strength_map[x] = spread_n(self.own_strength_map[0], x)

        e_square_index = np.transpose(np.nonzero(self.is_enemy_map))
        enemy_sqs = [self.squares[c[0], c[1]] for c in e_square_index]
        self.distance_from_enemy = self.flood_fill_enemy_map(enemy_sqs)
        print_map(self.distance_from_enemy, "distance_from_enemy_")

    @timethis
    def update_neutral_map(self):
        self.neutral_map = np.maximum(self.border_map - (self.enemy_strength_map[1] > 0), 0)
        n_s_i = np.transpose(np.nonzero(self.neutral_map))
        neutral_squares = [self.squares[c[0], c[1]] for c in n_s_i]
        self.distance_from_neutral = self.flood_fill(neutral_squares, max(self.w, self.h), True)

    @timethis
    def update_controlled_influence_production_maps(self):
        max_distance = 6
        self.controlled_production_influence_map = np.zeros((max_distance + 1, self.w, self.h))
        self.controlled_production_influence_map[0] = self.production_map * (self.is_enemy_map + self.is_owned_map)
        for distance in range(1, max_distance + 1):
            self.controlled_production_influence_map[distance] = spread_n(self.controlled_production_influence_map[distance - 1], 1)
            self.controlled_production_influence_map[distance] = rebase_map(self.controlled_production_influence_map[distance - 1], False)

    @timethis
    def update_value_production_map(self):

        if self.frame == 1:
            # "gini coefficient" calc
            gini_values = ((self.production_map / self.strength_map_1) * self.is_neutral_map).flatten()
            gini_values = gini_values[np.where(gini_values > 0)]
            gini_values = np.cumsum(sorted(gini_values))
            gini = (len(gini_values) * gini_values[-1] - 2 * np.trapz(gini_values) + gini_values[0]) / len(gini_values) / gini_values[-1]
            logging.debug("Frame:" + str(self.frame) + " Gini: " + str(gini))
            if gini > 0.5:
                self.g_mult = 0.65
            else:
                self.g_mult = 0.25

        self.base_value_map = np.divide(self.production_map_01, self.strength_map_1) * (self.is_neutral_map - self.combat_zone_map)
        # Each neutral cell gets assigned to the closest border non-combat cell
        global_targets_indices = np.transpose(np.nonzero(self.is_neutral_map - self.combat_zone_map))
        global_targets = [self.squares[c[0], c[1]] for c in global_targets_indices]
        self.global_border_map = np.zeros((self.w, self.h))

        gb_map = self.dij_recov_distance_map * (self.border_map - self.combat_zone_map)
        gb_map[gb_map == 0] = 9999

        for g in global_targets:
            if self.base_value_map[g.x, g.y] > 0.02:
                # Find the closest border square that routes to g
                gb_map = self.dij_recov_distance_map[g.x, g.y] * (self.border_map - self.combat_zone_map)
                gb_map[gb_map == 0] = 9999
                tx, ty = np.unravel_index(gb_map.argmin(), (self.w, self.h))
                self.global_border_map[tx, ty] += self.base_value_map[g.x, g.y] / self.dij_recov_distance_map[g.x, g.y, tx, ty]

        self.value_production_map = 1 / np.maximum(self.base_value_map + self.global_border_map * self.g_mult, 0.001)

        self.value_production_map *= (self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] == 0)
        self.value_production_map[self.value_production_map == 0] = 9999
        turns_left = self.max_turns - self.frame
        recover_threshold = turns_left * 0.6
        self.value_production_map[self.value_production_map > recover_threshold] == 9999

        avg_recov_threshold = 2
        # avg_map_recovery = np.sum(self.strength_map * (self.border_map - self.combat_zone_map)) / np.sum(self.production_map * (self.border_map - self.combat_zone_map))
        avg_map_recovery = np.sum(self.strength_map * self.border_map) / np.sum(self.production_map * self.border_map)
        self.value_production_map[self.value_production_map > (avg_recov_threshold * avg_map_recovery)] = 9999

    def update_stats(self):
        # Updates various stats used for tracking
        self.turns_left = self.max_turns - self.frame
        self.percent_owned = np.sum(self.is_owned_map) / (self.w * self.h)
        self.production_values = [0]
        for i in range(1, self.starting_player_count + 1):
            self.production_values.append(np.sum(self.production_map * (self.owner_map == i)))
        self.my_production_sum = self.production_values[self.my_id]
        temp_production_sum = copy.copy(self.production_values)
        temp_production_sum.pop(self.my_id)
        temp_production_sum.pop(0)
        if len(temp_production_sum) > 0:
            self.next_highest_production_sum = max(temp_production_sum)
        else:
            self.next_highest_production_sum = 9999

        if np.sum(self.is_owned_map * self.enemy_strength_map[4]) > 0:
            self.near_enemy = True
        else:
            self.near_enemy = False

        # Detect who we are currently in combat with
        self.in_combat_with = []
        combat_zone_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]
        for sq in combat_zone_squares:
            if self.own_strength_map[1, sq.x, sq.y] > 0:
                for n in sq.neighbors:
                    if n.owner != 0 and n.owner != self.my_id:
                        self.in_combat_with.append(n.owner)

        self.in_combat_with = list(set(self.in_combat_with))

    def update_focus_territory(self):
        self.production_cells_out = 10
        self.combat_radius = min(self.turns_left, self.combat_radius)
        self.value_production_map = np.copy(self.strength_map * self.is_neutral_map)

    @timethis
    def get_moves(self):
        if self.turns_left <= 10:
            self.update_focus_territory()
        # This is the main logic controlling code.
        # Find super high production cells
        # self.get_pre_combat_production()
        # 1 - Find combat zone cells and attack them.
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_attack()
        # self.get_moves_prepare_strength()
        # 2 - Find production zone cells and attack them
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_production()
        # 3 - Move all other unassigned cells.
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        self.get_moves_other()

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

    @timethis
    def get_moves_attack(self):
        # Consider attack patterns
        self.safe_to_move = np.ones((self.w, self.h))

        self.get_attack_patterns()

        # Attempts to attack all border cells that are in combat
        combat_zone_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]

        combat_zone_squares.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
        combat_zone_squares.sort(key=lambda x: self.enemy_strength_map[1, x.x, x.y], reverse=True)

        # TODO: Should sort by amount of overkill damage possible.
        # for square in combat_zone_squares:
        #     if len(square.moving_here) == 0:
        #         self.attack_cell(square, 1)

        self.get_moves_breakthrough()
        # Get a list of all squares within x spaces of a combat zone.
        # TODO: This causes bounciness, i should probably do a floodfill of all combat zone squares instead?
        combat_distance_matrix = self.flood_fill(combat_zone_squares, self.combat_radius, True)
        # combat_distance_matrix[combat_distance_matrix == -1] = 0
        # combat_distance_matrix[combat_distance_matrix == 1] = 0
        combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]
        combat_squares = [s for s in combat_squares if s.owner == self.my_id]

        combat_squares.sort(key=lambda x: x.strength, reverse=True)
        combat_squares.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
        # combat_squares.sort(key=lambda x: x.strength - self.enemy_strength_map[2, x.x, x.y], reverse=True)
        combat_squares.sort(key=lambda x: self.distance_from_combat_zone[x.x, x.y])
        combat_squares.sort(key=lambda x: self.distance_from_enemy[x.x, x.y])

        print_map(combat_distance_matrix, "combat_distance_matrix_")
        self.safe_to_move = np.ones((self.w, self.h))
        for square in combat_squares:
            if (combat_distance_matrix[square.x, square.y] == 1):
                if (square.strength > square.production) and (square.move == -1):
                    if square.is_isolated():
                        # Check diagonals & Plus for isolated.
                        diagonals = [(-1, -1), (1, 1), (-1, 1), (1, -1), (0, 2), (-2, 0), (2, 0), (0, -2)]
                        should_still = False
                        enemy_count = 0
                        for n in square.get_neighbors(2):
                            if n.owner != 0 and n.owner != self.my_id and n.strength > n.production * 2:
                                enemy_count += 1
                        if enemy_count > 1:
                            should_still = True
                            neighbor_combat_squares = 0
                            for n in square.neighbors:
                                if self.combat_zone_map[n.x, n.y] and self.enemy_strength_map[1, n.x, n.y] >= 1:
                                    neighbor_combat_squares += 1
                            if neighbor_combat_squares <= 1:
                                should_still = False
                            if self.enemy_strength_map[2, square.x, square.y] < square.strength:
                                should_still = False
                            for (dx, dy) in diagonals:
                                dsq = self.squares[(square.x + dx) % self.w, (square.y + dy) % self.h]
                                if dsq.owner == self.my_id and dsq.is_isolated() and (dsq.move == 4 or dsq.move == -1):
                                    should_still = False
                                    break
                        if should_still and self.safe_to_move[square.x, square.y]:
                            self.make_move(square, STILL)
                            if self.distance_from_enemy[square.x, square.y] <= 3 and square.strength > 10:
                                for n in square.get_neighbors(2):
                                    if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                        self.safe_to_move[n.x, n.y] = 0
                            continue
                    targets = []
                    alt_targets = []
                    inside_target = []
                    for n in square.neighbors:
                        if self.enemy_strength_map[2, n.x, n.y] == 0 or self.safe_to_move[n.x, n.y]:
                            if n.owner == 0 and n.strength == 0:
                                targets.append(n)
                            elif n.owner == self.my_id:
                                if combat_distance_matrix[n.x, n.y] < combat_distance_matrix[square.x, square.y]:
                                    alt_targets.append(n)
                                else:
                                    inside_target.append(n)
                    targets.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)
                    targets.sort(key=lambda x: self.distance_from_enemy[x.x, x.y])
                    alt_targets.sort(key=lambda x: x.strength)
                    inside_target.sort(key=lambda x: x.strength)
                    success = False
                    for t in targets:
                        success = self.move_square_to_target_simple(square, t, False)
                        if success:
                            if self.distance_from_enemy[t.x, t.y] <= 3 and square.strength > 10:
                                for n in t.get_neighbors(2):
                                    if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                        self.safe_to_move[n.x, n.y] = 0
                            break
                    if not success:
                        for t in alt_targets:
                            success = self.move_square_to_target_simple(square, t, True)
                            if success:
                                if self.distance_from_enemy[t.x, t.y] <= 3 and square.strength > 10:
                                    for n in t.get_neighbors(2):
                                        if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                            self.safe_to_move[n.x, n.y] = 0
                                break
                    if not success:
                        if self.safe_to_move[square.x, square.y]:
                            success = True
                            self.make_move(square, STILL)
                            if self.distance_from_enemy[square.x, square.y] <= 3 and square.strength > 10:
                                for n in square.get_neighbors(2):
                                    if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                        self.safe_to_move[n.x, n.y] = 0
                    if not success:
                        for t in inside_target:
                            success = self.move_square_to_target_simple(square, t, True)
                            if success:
                                if self.distance_from_enemy[t.x, t.y] <= 3 and square.strength > 10:
                                    for n in t.get_neighbors(2):
                                        if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                            self.safe_to_move[n.x, n.y] = 0
                                break
            # elif (combat_distance_matrix[square.x, square.y] == 2):
            #     self.make_move(square, STILL)
            # elif ((square.strength > (square.production * (self.buildup_multiplier[square.x, square.y] + self.distance_from_combat_zone[square.x, square.y]))) or square.strength > 250) and (square.parity == self.parity) and square.move == -1 and square.moving_here == []:
            # elif ((square.strength > (square.production * (self.buildup_multiplier[square.x, square.y] + 2))) or square.strength > 250) and (square.parity == self.parity) and square.move == -1 and square.moving_here == []:
            elif ((square.strength > (square.production * (self.buildup_multiplier[square.x, square.y] + 2))) or square.strength > 250) and square.move == -1 and square.moving_here == [] and combat_distance_matrix[square.x, square.y] > 1:
                # elif ((square.strength > (square.production * (self.buildup_multiplier[square.x, square.y] + 2))) or square.strength > 250) and square.move == -1 and square.moving_here == []:
                # elif square.move == -1 and square.moving_here == [] and square.strength > self.buildup_multiplier[square.x, square.y] + 7:
                current_distance = combat_distance_matrix[square.x, square.y]
                possible_moves = []
                for n in square.neighbors:
                    if self.is_owned_map[n.x, n.y]:
                        if combat_distance_matrix[n.x, n.y] == current_distance - 1 and self.safe_to_move[n.x, n.y]:
                            possible_moves.append(n)
                if len(possible_moves) > 0:
                    possible_moves.sort(key=lambda sq: sq.production)
                    possible_moves.sort(key=lambda sq: self.enemy_strength_map[4, sq.x, sq.y], reverse=True)
                    self.move_square_to_target(square, possible_moves[0], True)
                    if self.distance_from_enemy[n.x, n.y] <= 3 and square.strength > 10:
                        for n in possible_moves[0].get_neighbors(2):
                            if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                self.safe_to_move[n.x, n.y] = 0
                else:
                    self.make_move(square, STILL)
                    if self.distance_from_enemy[square.x, square.y] <= 3 and square.strength > 10:
                        for n in square.get_neighbors(2):
                            if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                self.safe_to_move[n.x, n.y] = 0
            else:
                if combat_distance_matrix[square.x, square.y] > 1:
                    self.make_move(square, STILL)
                    if self.distance_from_enemy[square.x, square.y] <= 3 and square.strength > 10:
                        for n in square.get_neighbors(2):
                            if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                self.safe_to_move[n.x, n.y] = 0

    @timethis
    def get_moves_prepare_strength(self):
        # Attempts to build up strength prior to an immediate engagement, only if we aren't already in combat
        border_prepare_indices = np.transpose(np.nonzero(self.border_map * self.enemy_strength_map[1] > 0))
        enemy_border_squares = [self.squares[c[0], c[1]] for c in border_prepare_indices]

        if len(enemy_border_squares) > 0:
            combat_distance_matrix = self.flood_fill(enemy_border_squares, 5, True)
            combat_distance_matrix[combat_distance_matrix == -1] = 0
            combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]

            for square in combat_squares:
                if (self.distance_from_combat_zone[square.x, square.y] > 3) and (square.strength > square.production * self.buildup_multiplier[square.x, square.y] + 5) and (square.parity == self.parity) and square.move == -1 and square.moving_here == []:
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
                    self.make_move(square, STILL)

    @timethis
    def get_moves_production(self):
        # Tries to find the best cells to attack from a production standpoint.
        # Does not try to attack cells that are in combat zones.
        # potential_targets_indices = np.transpose(np.nonzero((self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] == 0)))
        potential_targets_indices = np.transpose(np.nonzero((self.value_production_map < 8000)))
        potential_targets_one = [self.squares[c[0], c[1]] for c in potential_targets_indices]

        potential_targets_one.sort(key=lambda x: self.value_production_map[x.x, x.y])
        percentile = 0.85
        cutoff = int(len(potential_targets_one) * percentile)
        potential_targets_one = potential_targets_one[:cutoff]

        potential_targets = []
        for c in potential_targets_one:
            target = self.squares[c.x, c.y]
            value = self.value_production_map[c.x, c.y]
            cells_out = 1
            while cells_out <= self.production_cells_out:
                potential_targets.append((target, value, cells_out))
                cells_out += 1

        if len(potential_targets) == 0:
            return
        potential_targets.sort(key=lambda x: x[0].strength)
        potential_targets.sort(key=lambda x: x[1] + (x[2] * 1))

        # Keep only the top x% ile?
        percentile = 1
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

    @timethis
    def get_moves_breakthrough(self):
        # Determine if we should bust through and try to open up additional lanes of attack into enemy territory
        # Best to have a separate lane. so we should evaluate squares that are not next to already open channels.
        # We are only looking at squares which are next to the enemy already.
        if len(self.in_combat_with) == 0:
            enemy_targets = [1, 2, 3, 4, 5, 6]
            enemy_targets.remove(self.my_id)
            enemy_targets = []
        else:
            enemy_targets = self.in_combat_with

        potential_squares_indices = np.transpose(np.nonzero((self.border_map - self.combat_zone_map) * (self.enemy_strength_map[1] > 0) * self.safe_to_move))
        potential_squares = [self.squares[c[0], c[1]] for c in potential_squares_indices]
        potential_squares.sort(key=lambda sq: self.own_strength_map[4, sq.x, sq.y] / self.enemy_strength_map[4, sq.x, sq.y])
        # We only want to bust through if we have a lot of strength here.
        breakthrough_count = 0
        breakthrough_limit = 5
        for sq in potential_squares:
            attack = False
            for n in sq.neighbors:
                if n.owner in enemy_targets:
                    attack = True
                    break
            if attack:
                if self.own_strength_map[4, sq.x, sq.y] > 500 and (self.own_strength_map[4, sq.x, sq.y] > 1.5 * self.enemy_strength_map[4, sq.x, sq.y]):
                    attack = False
                    for n in sq.neighbors:
                        if n.owner == self.my_id and self.distance_from_combat_zone[n.x, n.y] > 3:
                            attack = True
                    if attack:
                        success = self.attack_cell(sq, 1)
                        if success:
                            if self.distance_from_enemy[sq.x, sq.y] <= 2:
                                for n in sq.get_neighbors(2):
                                    if self.distance_from_enemy[n.x, n.y] <= 2 and self.enemy_strength_map[2, n.x, n.y] > 20:
                                        self.safe_to_move[n.x, n.y] = 0
                            breakthrough_count += 1
                            if breakthrough_count >= breakthrough_limit:
                                break  # Break through only 1 square per turn.

    @timethis
    def get_moves_other(self):
        idle_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))]

        if len(idle_squares) == 0:
            return

        idle_squares.sort(key=lambda sq: sq.strength, reverse=True)
        idle_squares.sort(key=lambda sq: self.distance_from_border[sq.x, sq.y])
        idle_squares.sort(key=lambda sq: self.distance_from_combat_zone[sq.x, sq.y])

        for sq in idle_squares:
            if (timer() - game.start) > MAX_TURN_TIME:
                return
            if sq.strength > sq.production * self.buildup_multiplier[sq.x, sq.y] and sq.move == -1 and sq.moving_here == []:
                if np.sum(self.combat_zone_map) == 0:
                    # self.find_nearest_non_owned_border(sq)
                    self.find_nearest_neutral_border(sq)
                else:
                    if self.distance_from_combat_zone[sq.x, sq.y] < 6 and sq.parity != game.parity:
                        continue
                    if self.enemy_strength_map[3, sq.x, sq.y] > 0 and sq.parity != game.parity:
                        self.make_move(sq, STILL)
                    else:
                        self.find_nearest_combat_zone(sq)

    def distance_between(self, sq1, sq2):
        dx = abs(sq1.x - sq2.x)
        dy = abs(sq1.y - sq2.y)
        if dx > self.w / 2:
            dx = self.w - dx
        if dy > self.h / 2:
            dy = self.h - dy
        return dx + dy

    def attack_cell(self, target, max_cells_out, min_cells_out=1):
        # Attempts to coordinate attack to a specific cell.
        cells_out = min_cells_out

        while cells_out <= max_cells_out:
            # If we're trying to attack a combat zone cell, this isn't the function to do it. cancel.
            if cells_out > 1 and self.combat_zone_map[target.x, target.y]:
                return False

            if np.sum(self.is_owned_map) <= 5 and self.near_enemy is False:
                free_squares = self.is_owned_map * (self.move_map == -1)
            else:
                if target.strength == 0 or self.value_production_map[target.x, target.y] <= 2:  # or target.production >= 5:  # or self.phase == 0:
                    free_squares = self.is_owned_map * (self.move_map == -1)
                else:
                    free_squares = self.is_owned_map * (self.move_map == -1) * (self.strength_map >= self.buildup_multiplier * self.production_map) * (self.moving_into_map == 0)
            target_distance_matrix = self.flood_fill([target], cells_out, True)
            target_distance_matrix[target_distance_matrix == -1] = 0
            target_distance_matrix = target_distance_matrix * free_squares
            available_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix, 1))

            target_distance_matrix_production = cells_out - target_distance_matrix
            target_distance_matrix_production[target_distance_matrix_production == cells_out] = 0  # Cells furthest out would be moving so no production
            target_distance_matrix_production = target_distance_matrix_production * free_squares
            available_production = np.sum(self.production_map * target_distance_matrix_production)

            if available_strength + available_production > target.strength + 0:
                attacking_cells = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(target_distance_matrix > 0))]

                still_cells = []
                if cells_out > 1:
                    still_cells = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(target_distance_matrix_production > 0))]
                moving_cells = list(set(attacking_cells) - set(still_cells))

                for square in still_cells:
                    self.make_move(square, STILL)

                still_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix_production, 1))
                needed_strength_from_movers = target.strength - available_production - still_strength + 1

                if needed_strength_from_movers > 0:
                    # Handle movement here
                    moving_cells.sort(key=lambda x: x.production)
                    moving_cells.sort(key=lambda x: x.strength, reverse=True)
                    moving_cells.sort(key=lambda x: self.distance_from_border[x.x, x.y], reverse=True)
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

        if direction == -1:  # Reset the square move
            if square.target is not None:
                square.target.moving_here.remove(square)
                self.moving_into_map[square.target.x, square.target.y] -= 1
                square.target = None
            square.move = -1
            return

        if square.move != -1:
            if square.target is not None:
                square.target.moving_here.remove(square)
                self.moving_into_map[square.target.x, square.target.y] -= 1
                square.target = None

        square.move = direction
        if direction != STILL:
            square.target = square.neighbors[direction]
            square.target.moving_here.append(square)
            self.moving_into_map[square.target.x, square.target.y] += 1

    def move_square_to_target(self, source, destination, through_friendly):
        # Get the distance matrix that we will use to determine movement.

        distance_matrix = self.flood_fill_until_target(source, destination, through_friendly)
        source_distance = distance_matrix[source.x, source.y]
        if source_distance == -1 or source_distance == 0:
            # We couldn't find a path to the destination or we're trying to move STILL
            return False

        path_choices = []
        for d in range(0, 4):
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
            if future_strength + source.strength <= self.str_cap:
                self.make_move(source, direction)
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
                    if future_strength <= self.str_cap:
                        # Ok, let's move the target square.
                        self.make_move(source, direction)  # Queue the move up, undo if it doesn't work
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in list(range(4))]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        n_neighbors.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for d, n in n_neighbors:
                            if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength  # + n.production
                                for n_moving in n.moving_here:
                                    future_n_t_strength += n_moving.strength
                                if future_n_t_strength <= self.str_cap:
                                    success = self.move_square_to_target_simple(target, n, True)
                                    if success:
                                        return True
                        # TODO: Logic to attempt to capture a neutral cell if we want.
                        self.make_move(source, -1)
        # Nothing to do left
        return False

    def move_square_to_target_simple(self, source, destination, through_friendly):
        # For large distances, we can probably get away with simple movement rules.
        dist_w = (source.x - destination.x) % self.w
        dist_e = (destination.x - source.x) % self.w
        dist_n = (source.y - destination.y) % self.h
        dist_s = (destination.y - source.y) % self.h

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
            if future_strength + source.strength <= self.str_cap:
                self.make_move(source, direction)
                return True

        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength  # + target.production
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= self.str_cap:
                self.make_move(source, direction)
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
                    if future_strength <= self.str_cap:
                        # Ok, let's move the target square.
                        self.make_move(source, direction)  # Queue the move up, undo if it doesn't work
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in list(range(4))]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        n_neighbors.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                        # Ok, none of these has worked, let's try moving to a neighbor square instead then.
                        for d, n in n_neighbors:
                            if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                                # Can we move into this square safely?
                                future_n_t_strength = target.strength
                                if n.move == STILL or n.move == -1:
                                    future_n_t_strength += n.strength  # + n.production
                                for n_moving in n.moving_here:
                                    future_n_t_strength += n_moving.strength
                                if future_n_t_strength <= self.str_cap:
                                    success = self.move_square_to_target_simple(target, n, True)
                                    if success:
                                        return True
                        # TODO: Logic to attempt to capture a neutral cell if we want.
                        self.make_move(source, -1)
        # Nothing to do left
        return False

    def find_nearest_non_owned_border(self, sq):
        current_distance = self.distance_from_border[sq.x, sq.y]
        if current_distance == 1:
            self.make_move(sq, STILL)
        targets = []

        for n in sq.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_border[n.x, n.y] == current_distance - 1:
                    targets.append(n)

        targets.sort(key=lambda s: self.own_strength_map[5, s.x, s.y])
        for n in targets:
            success = self.move_square_to_target(sq, n, True)
            if success:
                break

    def find_nearest_neutral_border(self, sq):
        current_distance = self.distance_from_neutral[sq.x, sq.y]
        if current_distance == 1:
            self.make_move(sq, STILL)
        targets = []

        for n in sq.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_neutral[n.x, n.y] == current_distance - 1:
                    targets.append(n)

        targets.sort(key=lambda s: self.own_strength_map[5, s.x, s.y])
        for n in targets:
            success = self.move_square_to_target(sq, n, True)
            if success:
                break

    def find_nearest_combat_zone(self, sq):
        current_distance = self.distance_from_combat_zone[sq.x, sq.y]
        targets = []

        for n in sq.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if self.distance_from_combat_zone[n.x, n.y] == current_distance - 1:
                    targets.append(n)

        targets.sort(key=lambda s: self.own_strength_map[7, s.x, s.y])
        for n in targets:
            success = self.move_square_to_target(sq, n, True)
            if success:
                break

    def move_towards_map_old(self, sq, distance_map, through_friendly=True):
        current_distance = distance_map[sq.x, sq.y]
        possible_moves = []
        for n in sq.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if distance_map[n.x, n.y] == current_distance - 1:
                    possible_moves.append(n)
        if len(possible_moves) > 0:
            possible_moves.sort(key=lambda sq: sq.production)
            possible_moves.sort(key=lambda sq: self.enemy_strength_map[4, sq.x, sq.y], reverse=True)
            self.move_square_to_target(sq, possible_moves[0], True)

    def flood_fill_until_target(self, source, destination, friendly_only):
        # Does a BFS flood fill to find shortest distance from source to target.
        # Starts the fill AT destination and then stops once we hit the target.
        q = [destination]
        distance_matrix = np.ones((self.w, self.h), dtype=int) * -1
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

    def flood_fill_to_border(self, sources):
        q = sources
        distance_matrix = np.ones((self.w, self.h)) * -1
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

    def flood_fill_enemy_map(self, sources):
        q = sources
        distance_matrix = np.ones((self.w, self.h)) * -1
        if len(sources) == 0:
            return distance_matrix

        for sq in sources:
            distance_matrix[sq.x, sq.y] = 0

        while len(q) > 0:
            c = q.pop(0)
            c_dist = distance_matrix[c.x, c.y]
            for n in c.neighbors:
                if distance_matrix[n.x, n.y] == -1 or distance_matrix[n.x, n.y] > (c_dist + 1):
                    if (n.owner == 0 and n.strength == 0) or n.owner == self.my_id:
                        distance_matrix[n.x, n.y] = c_dist + 1
                        q.append(n)

        return distance_matrix

    def flood_fill(self, sources, max_distance=999, friendly_only=True):
        # sources is a list of Squares
        q = sources
        distance_matrix = np.ones((self.w, self.h)) * -1
        if len(sources) == 0:
            return distance_matrix

        for sq in sources:
            distance_matrix[sq.x, sq.y] = 0

        while len(q) > 0:
            c = q.pop(0)
            c_dist = distance_matrix[c.x, c.y]
            for n in c.neighbors:
                if distance_matrix[n.x, n.y] == -1 or distance_matrix[n.x, n.y] > (c_dist + 1):
                    if (friendly_only and n.owner == self.my_id) or (not friendly_only and n.owner != self.my_id):
                        distance_matrix[n.x, n.y] = c_dist + 1
                        if c_dist < max_distance - 1:
                            q.append(n)

        return distance_matrix

    @timethis
    def last_resort_strength_check(self):
        # Calculates the projected strength map and identifies squares that are violating it.
        # Ignore strength overloads due to production for now
        projected_strength_map = np.zeros((self.w, self.h), dtype=int)
        # We only care about our moves.
        for sq in itertools.chain.from_iterable(self.squares):
            if sq.owner == self.my_id:
                if sq.move == -1 or sq.move == STILL:
                    projected_strength_map[sq.x, sq.y] += sq.strength  # + sq.production
                else:
                    dx, dy = get_offset(sq.move)
                    projected_strength_map[(sq.x + dx) % self.w, (sq.y + dy) % self.h] += sq.strength

        # Get a list of squares that are over the cap
        violation_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero((projected_strength_map > self.str_cap)))]
        violation_count = len(violation_squares)

        violation_squares.sort(key=lambda sq: sq.strength, reverse=True)
        violation_squares.sort(key=lambda sq: self.distance_from_combat_zone[sq.x, sq.y])

        for sq in violation_squares:
            if (timer() - game.start) > MAX_TURN_TIME:
                return
            if sq.owner == self.my_id and (sq.move == -1 or sq.move == STILL):
                # We can try to move this square to an neighbor.
                possible_paths = []
                for d in range(0, 4):
                    # Move to the lowest strength neighbor. this might cause a collision but we'll resolve it with multiple iterations
                    n = sq.neighbors[d]
                    if n.owner == self.my_id and self.enemy_strength_map[2, n.x, n.y] == 0:
                        possible_paths.append((d, n, projected_strength_map[n.x, n.y]))
                    elif n.owner == 0:
                        # Try attacking a bordering cell
                        if (sq.strength > (2 * n.strength)) and (n.production > 2):
                            possible_paths.append((d, n, n.strength + 255))

                possible_paths.sort(key=lambda x: x[2])
                possible_paths.sort(key=lambda x: self.distance_from_border[x[1].x, x[1].y], reverse=True)
                # Force a move there
                if len(possible_paths) > 0:
                    self.make_move(sq, possible_paths[0][0])
            else:
                # We aren't the problem. one of the squares that's moving here is going to collide with us.
                # How do we resolve this?
                options_list = []
                for n in sq.neighbors:
                    if n.owner == self.my_id:
                        options_list.append((n, projected_strength_map[n.x, n.y]))
                options_list.sort(key=lambda x: x[1])
                # Let's try having the smallest one stay still instead
                for opt in options_list:
                    self.make_move(opt[0], STILL)

        return violation_count

    @timethis
    def stop_swaps(self):
        # Check if two squares are swapping places for no reason.
        for x in range(self.w):
            for y in range(self.h):
                if (timer() - game.start) > MAX_TURN_TIME:
                    return
                if self.is_owned_map[x, y]:
                    s = self.squares[x, y]
                    if s.target is not None:
                        if s.target in s.moving_here:
                            if s.strength >= s.target.strength:
                                if self.distance_from_combat_zone[s.x, s.y] <= self.combat_radius:
                                    if self.distance_from_combat_zone[s.x, s.y] <= self.distance_from_combat_zone[s.target.x, s.target.y]:
                                        self.make_move(s.target, STILL)
                                        self.make_move(s, STILL)
                                elif self.distance_from_border[s.x, s.y] <= self.distance_from_border[s.target.x, s.target.y]:
                                    if (s.strength - s.target.strength) <= 0:
                                        self.make_move(s.target, STILL)
                                        self.make_move(s, STILL)

    @timethis
    def check_parity(self):
        indices = np.transpose(np.nonzero((self.is_owned_map * self.enemy_strength_map[3])))
        squares = [self.squares[c[0], c[1]] for c in indices]
        squares.sort(key=lambda sq: sq.strength, reverse=True)

        for s in squares:
            if (timer() - game.start) > MAX_TURN_TIME:
                return
            if (self.enemy_strength_map[2, s.x, s.y] == 0) and s.parity != game.parity and (s.move != STILL and s.move != -1):
                self.make_move(s, STILL)
                future_strength = s.strength + sum(x.strength for x in s.moving_here)
                if future_strength > self.str_cap:
                    s.moving_here.sort(key=lambda x: x.strength)
                    while future_strength > 255:
                        future_strength -= s.moving_here[0].strength
                        self.make_move(s.moving_here[0], STILL)
            elif (self.enemy_strength_map[2, s.x, s.y] > 0) and (s.move == STILL or s.move == -1):
                # Try to capture a neutral cell
                neutral_targets = []
                friendly_targets = []
                near_combat = False
                for t in s.neighbors:
                    if t.owner == self.my_id:
                        friendly_targets.append(t)
                    else:
                        if s.strength > t.strength:
                            if t.strength == 0:
                                near_combat = True
                            neutral_targets.append(t)
                friendly_targets.sort(key=lambda x: sum(y.strength for y in x.moving_here))
                success = False
                if near_combat:
                    for t in friendly_targets:
                        future_strength = sum(x.strength for x in t.moving_here) + t.strength if (t.move == STILL or t.move == -1) else 0
                        if future_strength + s.strength <= self.str_cap:
                            success = self.move_square_to_target_simple(s, t, True)
                            if success:
                                break
                    if not success:
                        neutral_targets.sort(key=lambda x: sum(y.strength for y in x.moving_here))
                        for t in neutral_targets:
                            future_strength = sum(x.strength for x in t.moving_here)
                            if future_strength + s.strength <= self.str_cap:
                                success = self.move_square_to_target_simple(s, t, False)
                                if success:
                                    break

    def overkill_check(self):
        one_away_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.distance_from_combat_zone == 1))]
        for sq in one_away_squares:
            # Isolated squares that are near enemies can stay still.
            if sq.owner == self.my_id:
                if sq.is_isolated():
                    # Check diagonals & Plus for isolated.
                    diagonals = [(-1, -1), (1, 1), (-1, 1), (1, -1), (0, 2), (-2, 0), (2, 0), (0, -2)]
                    should_still = False
                    enemy_count = 0
                    for n in sq.get_neighbors(2):
                        if n.owner != 0 and n.owner != self.my_id:
                            enemy_count += 1
                    if enemy_count > 1:
                        should_still = True
                        neighbor_combat_squares = 0
                        for n in sq.neighbors:
                            if self.combat_zone_map[n.x, n.y] and self.enemy_strength_map[1, n.x, n.y] >= 1:
                                neighbor_combat_squares += 1
                        if neighbor_combat_squares <= 1:
                            should_still = False
                            continue
                        if self.enemy_strength_map[2, sq.x, sq.y] < sq.strength:
                            should_still = False
                            continue
                        for (dx, dy) in diagonals:
                            dsq = self.squares[(sq.x + dx) % self.w, (sq.y + dy) % self.h]
                            if dsq.owner == self.my_id and dsq.is_isolated() and (dsq.move == 4 or dsq.move == -1):
                                should_still = False
                                break
                    if should_still:
                        self.make_move(sq, STILL)

        two_away_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.distance_from_combat_zone == 2))]
        two_away_squares = [x for x in two_away_squares if x.owner == self.my_id and x.strength > 0]
        for sq in two_away_squares:
            # Check the situation in which A -> B -> 0 <- C
            if sq.move != -1 and sq.move != 4:
                b = sq.target
                if b.move != -1 and b.move != 4:
                    if b.target.owner == 0 and b.target.strength == 0:
                        f, n, c, e = b.around()
                        if c == 1:
                            # Check Oh
                            oh = b.target
                            of, on, oc, oe = oh.around()
                            if oe >= 1:
                                # A is going to B, B is going to 0, C might be going to 0, keep A Still
                                e_str = self.enemy_strength_map[1, oh.x, oh.y]
                                if b.strength > e_str:
                                    self.make_move(sq, STILL)
                                else:
                                    self.make_move(b, STILL)

        czsq = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]

        for sq in czsq:
            # look for patterns:    N1 F1
            #                          N2 F2
            # and all possible rotations of it. Move F1 to N1. and move F2 STILL
            for d in range(0, 4):
                if sq.neighbors[d].owner == self.my_id and sq.neighbors[d].strength > 0:
                    # Look perpendicular.
                    perp_d = [(d + 1) % 4, (d - 1) % 4]
                    for d2 in perp_d:
                        n2 = sq.neighbors[d].neighbors[d2]
                        if self.combat_zone_map[n2.x, n2.y]:
                            # Look in the same direction as d.
                            if sq.neighbors[d].neighbors[d2].neighbors[d].owner == self.my_id and sq.neighbors[d].neighbors[d2].neighbors[d].strength > 0:
                                # We found a matching pattern. Make the moves.
                                self.make_move(sq.neighbors[d].neighbors[d2].neighbors[d], STILL)
                                self.make_move(sq.neighbors[d], opposite_direction(d))
                                break

    def get_attack_patterns(self):
        return
        czsq = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]

        for sq in czsq:
            # look for patterns:    N1 F1
            #                          N2 F2
            # and all possible rotations of it. Move F1 to N1. and move F2 STILL
            for d in range(0, 4):
                if sq.neighbors[d].owner == self.my_id and sq.neighbors[d].strength > sq.neighbors[d].production * 3:
                    # Look perpendicular.
                    perp_d = [(d + 1) % 4, (d - 1) % 4]
                    for d2 in perp_d:
                        if sq.neighbors[d].neighbors[d2].owner == 0 and sq.neighbors[d].neighbors[d2].strength == 0:
                            # Look in the same direction as d.
                            if sq.neighbors[d].neighbors[d2].neighbors[d].owner == self.my_id and sq.neighbors[d].neighbors[d2].neighbors[d].strength > sq.neighbors[d].production * 3:
                                # We found a matching pattern. Make the moves.
                                self.make_move(sq.neighbors[d].neighbors[d2].neighbors[d], STILL)
                                self.make_move(sq.neighbors[d], opposite_direction(d))
                                break


# ==============================================================================
# Square class
# ==============================================================================


class Square:

    def __init__(self, game, x, y, production):
        self.game = game
        self.x, self.y = x, y
        self.production = production
        self.w, self.h = game.w, game.h
        self.vertex = x * self.h + y
        self.target = None
        self.moving_here = []
        self.parity = (x + y) % 2

    def after_init_update(self):
        # Should only be called after all squares in game have been initialized.
        self.north = self.game.squares[(self.x + 0) % self.w, (self.y - 1) % self.h]
        self.east = self.game.squares[(self.x + 1) % self.w, (self.y + 0) % self.h]
        self.south = self.game.squares[(self.x + 0) % self.w, (self.y + 1) % self.h]
        self.west = self.game.squares[(self.x - 1) % self.w, (self.y + 0) % self.h]
        self.neighbors = [self.north, self.east, self.south, self.west]  # doesn't include self

    def get_neighbors(self, n=1, include_self=False):
        # Returns a list containing all neighbors within n squares, excluding self unless include_self = True
        assert isinstance(include_self, bool)
        assert isinstance(n, int) and n > 0
        if n == 1:
            if not include_self:
                return self.neighbors
        combos = ((dx, dy) for dy in range(-n, n + 1) for dx in range(-n, n + 1) if abs(dx) + abs(dy) <= n)
        return (self.game.squares[(self.x + dx) % self.w][(self.y + dy) % self.h] for dx, dy in combos if include_self or dx or dy)

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

    def around(self):
        # Returns (Friendly, Neutral > 0, Neutral == 0, enemy)
        friendly = 0
        neutral = 0
        combat = 0
        enemy = 0
        for sq in self.neighbors:
            if sq.owner == self.game.my_id:
                friendly += 1
            elif sq.owner != 0:
                enemy += 1
            else:
                if sq.strength > 0:
                    neutral += 1
                else:
                    combat += 1
        return (friendly, neutral, combat, enemy)

    def overkill_safe(self):
        # Is it safe to move to this square??
        # STILL
        move_away = []
        check = [self.north, self.west, self.east, self.south, self]
        for n in check:
            for m in n.moving_here:
                move_away.append(n)
            if len(move_away) > 1:
                return (False, move_away)
        # NORTH
        move_away = []
        check = [self.north.north, self.north.west, self.north.east, self.north, self]
        for n in check:
            if n.owner == self.my_id and (n.move == -1 or n.move == 4):
                move_away.append(n)
            for m in n.moving_here:
                move_away.append(n)
            if len(move_away) > 1:
                return (False, move_away)
        # South
        move_away = []
        check = [self.south.south, self.south.west, self.south.east, self.south, self]
        for n in check:
            if n.owner == self.my_id and (n.move == -1 or n.move == 4):
                move_away.append(n)
            for m in n.moving_here:
                move_away.append(n)
            if len(move_away) > 1:
                return (False, move_away)
        # West
        move_away = []
        check = [self.west.south, self.west.north, self.west.west, self.west, self]
        for n in check:
            if n.owner == self.my_id and (n.move == -1 or n.move == 4):
                move_away.append(n)
            for m in n.moving_here:
                move_away.append(n)
            if len(move_away) > 1:
                return (False, move_away)
        # East
        move_away = []
        check = [self.east.south, self.east.north, self.east.east, self.east, self]
        for n in check:
            if n.owner == self.my_id and (n.move == -1 or n.move == 4):
                move_away.append(n)
            for m in n.moving_here:
                move_away.append(n)
            if len(move_away) > 1:
                return (False, move_away)
        return (True, [])

    def is_isolated(self):
        isolated = True
        for n in self.neighbors:
            if n.owner == self.owner:
                isolated = False
        return isolated

####################
# Helper Functions #
####################


def get_offset(direction):
    return ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[direction]


def opposite_direction(direction):
    return (direction + 2) % 4 if direction != STILL else STILL


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


def get_all_d_away(d):
    combos = []
    for x in range(0, d + 1):
        x_vals = list(set([x, -x]))
        y_vals = list(set([d - x, -(d - x)]))
        combos.extend(list(itertools.product(x_vals, y_vals)))
    return list(set(combos))


def rebase_map(map_a, total=True):
    # Takes a map and returns a rebased version where numpy.sum(map) = self.w * self.h
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

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()

    if (timer() - game.start) > MAX_TURN_TIME:
        return
    # game.check_parity()

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()

    # game.overkill_check()

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        if (timer() - game.start) > MAX_TURN_TIME:
            return
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()


# #####################
# Game run-time code #
# #####################


logging.basicConfig(filename='logging.log', level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)

if (profile):
    pr = cProfile.Profile()
    pr.enable()

game = Game()

while True:

    game.get_frame()
    game_loop()
    game.send_frame()

    if profile and game.frame == 199:
        pr.disable()
        pr.dump_stats("test.prof")
