# ==============================================================================
# Imports
# ==============================================================================
import functools
import itertools
import logging
import math
import numpy as np
import random
import scipy.sparse
import scipy.ndimage.filters
import sys
import time
import copy

# ==============================================================================
# Variables
# ==============================================================================
botname = "shummie v51-3-1"
strength_buffer = 0
print_maps = False


def print_map(npmap, name):
    directory = "Maps/"
    if print_maps and (game.frame % 50 == 0 or game.frame == 1):
        np.savetxt(directory + name + str(game.frame) + ".txt", npmap)

# ==============================================================================
# Game Class
# ==============================================================================


class Game:

    def __init__(self):
        # This should only be called once, and at the beginning of the game
        self.my_id = int(get_string())

        map_size_string = get_string()
        self.w, self.h = tuple(map(int, map_size_string.split()))

        production_map_string = get_string()
        self.production_map = np.array(list(map(int, production_map_string.split()))).reshape((self.h, self.w)).transpose()

        self.create_squares_list()

        self.frame = -1
        self.phase = 0

        self.get_frame()

        self.starting_player_count = np.amax(self.owner_map)  # Note, for range you'd need to increase the range by 1

        self.create_one_time_maps()

        self.max_turns = 10 * ((self.w * self.h) ** 0.5)

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
        self.move_map2 = np.ones((self.w, self.h)) * -1  # Could possibly expand this in the future to consider enemy moves...

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
        self.buildup = 5
        # self.buildup_multiplier = np.minimum(np.maximum(self.production_map, 4), 9)
        # self.pre_combat_threshold = -3
        self.combat_radius = 8
        # self.production_cells_out = 15
        # self.phase = 0
        # Find the "global max"
        # self.global_max_square = None
        # self.total_avg_cost_to_global = 0

    def update_configs(self):
        self.buildup = 5

        # if np.sum(self.combat_zone_map) > 3:
        #     self.production_cells_out = int(self.w / self.starting_player_count / 2.5)

        if self.phase == 0:
            if np.sum(self.is_owned_map) > 5:
                self.phase = 1

        if self.percent_owned > 0.6:
            self.buildup -= 1
            # self.pre_combat_threshold = 0
            # self.combat_radius = 10

        elif self.my_production_sum / self.next_highest_production_sum > 1.1:
            self.buildup += 1

    def create_one_time_maps(self):
        # self.distance_map = self.create_distance_map()
        self.distance_map_no_decay = self.create_distance_map()

        self.production_map_01 = np.maximum(self.production_map, 0.1)
        self.production_map_1 = np.maximum(self.production_map, 1)

        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)
        start = time.time()
        self.create_dijkstra_maps()
        end = time.time()
        logging.debug("Creating dijkstra maps: " + str(end - start))

    def create_distance_map(self):
        # Creates a distance map so that we can easily divide a map to get ratios that we are interested in
        # self.distance_map[x, y, :, :] returns an array of (width, height) that gives the distance (x, y) is from (i, j) for all i, j
        # Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.

        # Create the base map for 0, 0
        zero_zero_map = np.zeros((self.w, self.h))

        for x in range(self.w):
            for y in range(self.h):
                dist_x = min(x, -x % self.w)
                dist_y = min(y, -y % self.w)
                zero_zero_map[x, y] = max(dist_x + dist_y, 1)
        # zero_zero_map = zero_zero_map ** falloff

        distance_map = np.zeros((self.w, self.h, self.w, self.h))
        for x in range(self.w):
            for y in range(self.h):
                distance_map[x, y, :, :] = roll_xy(zero_zero_map, x, y)

        return distance_map

    def create_dijkstra_maps(self):
        def get_cost_recov(cellnum):
            x = cellnum // self.h
            y = cellnum % self.h
            return self.strength_map_1[x, y] / self.production_map_01[x, y]

        def get_cost_prod(cellnum):
            x = cellnum // self.h
            y = cellnum % self.h
            return (self.production_map_1[x, y] // 2 + 1)

        self.do_prod_dij = True
        if max(self.w, self.h) >= 0:
            self.do_prod_dij = False

        dij_recov_costs = scipy.sparse.dok_matrix((self.w * self.h, self.w * self.h))
        dij_prod_costs = scipy.sparse.dok_matrix((self.w * self.h, self.w * self.h))

        # start = time.time()
        for x in range(self.w):
            for y in range(self.h):
                coord = x * self.h + y

                dij_recov_costs[coord, ((x + 1) % self.w) * self.h + ((y + 0) % self.h)] = get_cost_recov(((x + 1) % self.w) * self.h + ((y + 0) % self.h))
                dij_recov_costs[coord, ((x - 1) % self.w) * self.h + ((y + 0) % self.h)] = get_cost_recov(((x - 1) % self.w) * self.h + ((y + 0) % self.h))
                dij_recov_costs[coord, ((x + 0) % self.w) * self.h + ((y + 1) % self.h)] = get_cost_recov(((x + 0) % self.w) * self.h + ((y + 1) % self.h))
                dij_recov_costs[coord, ((x + 0) % self.w) * self.h + ((y - 1) % self.h)] = get_cost_recov(((x + 0) % self.w) * self.h + ((y - 1) % self.h))

                if self.do_prod_dij:
                    dij_prod_costs[coord, ((x + 1) % self.w) * self.h + ((y + 0) % self.h)] = get_cost_prod(((x + 1) % self.w) * self.h + ((y + 0) % self.h))
                    dij_prod_costs[coord, ((x - 1) % self.w) * self.h + ((y + 0) % self.h)] = get_cost_prod(((x - 1) % self.w) * self.h + ((y + 0) % self.h))
                    dij_prod_costs[coord, ((x + 0) % self.w) * self.h + ((y + 1) % self.h)] = get_cost_prod(((x + 0) % self.w) * self.h + ((y + 1) % self.h))
                    dij_prod_costs[coord, ((x + 0) % self.w) * self.h + ((y - 1) % self.h)] = get_cost_prod(((x + 0) % self.w) * self.h + ((y - 1) % self.h))

        # end = time.time()
        # logging.debug("init dijkstra dok maps: " + str(end - start))
        # start = time.time()
        self.dij_recov_cost, self.dij_recov_route = scipy.sparse.csgraph.dijkstra(dij_recov_costs, return_predecessors=True)
        self.dij_recov_distance_map = np.zeros((self.w, self.h, self.w, self.h))
        self.dij_recov_route_map = np.zeros((self.w, self.h, self.w, self.h))

        if self.do_prod_dij:
            self.dij_prod_cost, self.dij_prod_route = scipy.sparse.csgraph.dijkstra(dij_prod_costs, return_predecessors=True)
            self.dij_prod_distance_map = np.zeros((self.w, self.h, self.w, self.h))
            self.dij_prod_route_map = np.zeros((self.w, self.h, self.w, self.h))

        # end = time.time()
        # logging.debug("running dijkstra maps: " + str(end - start))
        # start = time.time()
        for x in range(self.w):
            for y in range(self.h):
                self.dij_recov_distance_map[x, y, :, :] = self.dij_recov_cost[x * self.h + y].reshape((self.w, self.h))
                self.dij_recov_route_map[x, y, :, :] = self.dij_recov_route[x * self.h + y].reshape((self.w, self.h))
                if self.do_prod_dij:
                    self.dij_prod_distance_map[x, y, :, :] = self.dij_recov_cost[x * self.h + y].reshape((self.w, self.h))
                    self.dij_prod_route_map[x, y, :, :] = self.dij_recov_route[x * self.h + y].reshape((self.w, self.h))
        # end = time.time()
        # logging.debug("reshape dijkstra maps: " + str(end - start))

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
        # end = time.time()
        # logging.debug("update_recover Frame: " + str(game.frame) + " : " + str(end - start))
        # start = time.time()
        self.update_value_maps()
        # end = time.time()
        # logging.debug("update_value_maps Frame: " + str(game.frame) + " : " + str(end - start))
        self.update_controlled_influence_production_maps()

    def update_calc_maps(self):
        self.strength_map_01 = np.maximum(self.strength_map, 0.1)
        self.strength_map_1 = np.maximum(self.strength_map, 1)

    def update_owner_maps(self):
        self.is_owned_map = np.zeros((self.w, self.h))
        self.is_neutral_map = np.zeros((self.w, self.h))
        self.is_enemy_map = np.zeros((self.w, self.h))

        # minor speed up opportunity here? loop through once to create map instead of 3 times?
        self.is_owned_map[np.where(self.owner_map == self.my_id)] = 1
        self.is_neutral_map[np.where(self.owner_map == 0)] = 1
        self.is_enemy_map = 1 - self.is_owned_map - self.is_neutral_map

    def update_border_maps(self):
        self.border_map = np.zeros((self.w, self.h))
        self.combat_zone_map = np.zeros((self.w, self.h))

        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == 0:
                for n in square.neighbors:
                    if n.owner == self.my_id:
                        self.border_map[square.x, square.y] = 1
                        if square.strength == 0:
                            self.combat_zone_map[square.x, square.y] = 1
                        continue

        border_squares_indices = np.transpose(np.nonzero(self.border_map))
        border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        self.distance_from_border = self.flood_fill(border_squares, max(self.w, self.h), True)

        owned_squares_indices = np.transpose(np.nonzero(self.is_owned_map))
        owned_squares = [self.squares[c[0], c[1]] for c in owned_squares_indices]
        self.distance_from_owned = self.flood_fill(owned_squares, max(self.w, self.h), False)
        print_map(self.distance_from_owned, "distance_from_owned")

        if self.starting_player_count > 1 and np.sum(self.combat_zone_map) >= 1:  # Breaks in single player mode otherwise.
            combat_squares_indices = np.transpose(np.nonzero(self.combat_zone_map))
            combat_squares = [self.squares[c[0], c[1]] for c in combat_squares_indices]
            self.distance_from_combat_zone = self.flood_fill(combat_squares, max(self.w, self.h), True)
            self.distance_from_combat_zone[self.distance_from_combat_zone == -1] = 9999
            print_map(self.distance_from_combat_zone, "distance_from_combat_zone")

            enemy_squares_indices = np.transpose(np.nonzero(self.is_enemy_map))
            enemy_squares = [self.squares[c[0], c[1]] for c in enemy_squares_indices]
            self.distance_from_enemy = self.flood_fill(enemy_squares, max(self.w, self.h), False)
            print_map(self.distance_from_enemy, "distance_from_enemy")

        else:
            self.distance_from_combat_zone = np.ones((self.w, self.h)) * 999
            self.distance_from_enemy = np.ones((self.w, self.h)) * 999

    def update_enemy_maps(self):
        self.enemy_strength_map = np.zeros((5, self.w, self.h))
        self.enemy_strength_map[0] = self.strength_map * self.is_enemy_map

        for x in range(len(self.enemy_strength_map)):
            self.enemy_strength_map[x] = spread_n(self.enemy_strength_map[0], x)

        self.own_strength_map = np.zeros((5, self.w, self.h))
        self.own_strength_map[0] = self.strength_map * self.is_owned_map

        for x in range(len(self.own_strength_map)):
            self.own_strength_map[x] = spread_n(self.own_strength_map[0], x)

    def update_value_maps(self):
        self.base_value_map = np.divide(self.production_map_01, self.strength_map_1) * (self.is_neutral_map - self.combat_zone_map)
        # Each neutral cell gets assigned to the closest border non-combat cell
        global_targets_indices = np.transpose(np.nonzero(self.is_neutral_map - self.combat_zone_map))
        global_targets = [self.squares[c[0], c[1]] for c in global_targets_indices]
        # border_squares_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        # border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        self.global_border_map = np.zeros((self.w, self.h))

        for g in global_targets:
            # Find the closest border square that routes to g
            gb_map = self.dij_recov_distance_map[g.x, g.y] * (self.border_map - self.combat_zone_map)
            gb_map[gb_map == 0] = 9999
            tx, ty = np.unravel_index(gb_map.argmin(), (self.w, self.h))
            self.global_border_map[tx, ty] += self.base_value_map[g.x, g.y] / self.dij_recov_distance_map[g.x, g.y, tx, ty]

        self.value_map = 1 / np.maximum(self.base_value_map + self.global_border_map * 1, 0.001)
        print_map(self.global_border_map, "global_border_")
        print_map(self.base_value_map, "base_value_")
        print_map(self.value_map, "value_map_")

    def update_controlled_influence_production_maps(self):
        max_distance = 9
        self.controlled_production_influence_map = np.zeros((max_distance + 1, self.w, self.h))
        self.controlled_production_influence_map[0] = self.production_map * (self.is_enemy_map + self.is_owned_map)
        for distance in range(1, max_distance + 1):
            self.controlled_production_influence_map[distance] = spread_n(self.controlled_production_influence_map[distance - 1], 1)
            self.controlled_production_influence_map[distance] = rebase_map(self.controlled_production_influence_map[distance - 1], False)

    def flood_fill(self, sources, max_distance=999, friendly_only=True):
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

    def flood_fill_until_target(self, source, destination, friendly_only):
        # Does a BFS flood fill to find shortest distance from source to target.
        # Starts the fill AT destination and then stops once we hit the target.
        q = [destination]
        distance_matrix = np.ones((self.w, self.h)) * -1
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
        self.next_highest_production_sum = max(temp_production_sum)

    def get_moves(self):

        if self.phase == 0:
            self.early_game_production()

        self.attack_borders()
        self.get_moves_breakthrough()
        # if self.phase == 1:
        self.move_inner_squares()

    def early_game_production(self):
        ev_map = self.value_map * self.border_map
        ev_map[ev_map == 0] = 9999
        tx, ty = np.unravel_index(ev_map.argmin(), (self.w, self.h))
        target = self.squares[tx, ty]
        self.attack_cell(target, 5)

    def attack_borders(self):
        # get a list of border cells available for attack.
        # potential_targets_indices = np.transpose(np.nonzero(self.border_map))
        potential_targets_indices = np.transpose(np.nonzero(self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]

        # Sort targets by enemy strength then strength?
        potential_targets.sort(key=lambda x: self.enemy_strength_map[2, x.x, x.y], reverse=True)

        for square in potential_targets:
            if (square.strength == 0 and self.combat_zone_map[square.x, square.y] == 0):
                    # These are squares which should be captured by the lowest strength piece since higher strength pieces should be attacking.
                    n = [x for x in square.neighbors if x.owner == self.my_id and x.strength > 0]
                    n.sort(key=lambda x: x.strength)
                    self.move_square_to_target_simple(n[0], square, False)
            elif (square.x + square.y) % 2 == game.frame % 2:
            #     # Off parity square, don't force an attack (is this actually useful?)
                continue
            else:
                self.attack_cell(square, 1)

        combat_zone_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(self.combat_zone_map))]
        combat_distance_matrix = self.flood_fill(combat_zone_squares, self.combat_radius, True)
        combat_distance_matrix[combat_distance_matrix == -1] = 0
        combat_distance_matrix[combat_distance_matrix == 1] = 0
        combat_squares = [self.squares[c[0], c[1]] for c in np.transpose(np.nonzero(combat_distance_matrix))]
        combat_squares.sort(key=lambda x: x.strength, reverse=True)

        for square in combat_squares:
            if (square.strength > square.production * (self.buildup + self.distance_from_combat_zone[square.x, square.y])) and square.move == -1 and square.moving_here == [] and ((square.x + square.y) % 2 == self.frame % 2):
                # self.move_towards_map(square, self.distance_from_combat_zone)
                self.move_towards_map_old(square, combat_distance_matrix)
            elif square.strength >= square.production and square.move == -1 and self.distance_from_combat_zone[square.x, square.y] < 2:
                self.move_towards_map_old(square, combat_distance_matrix)
            else:
                self.make_move(square, STILL, -1)


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

    def move_inner_squares(self):
        idle_squares_indices = np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))
        idle_squares = [self.squares[c[0], c[1]] for c in idle_squares_indices]

        if len(idle_squares) == 0:
            return
        # Move squares closer to the border first.
        idle_squares.sort(key=lambda sq: self.distance_from_border[sq.x, sq.y])
        idle_squares.sort(key=lambda sq: sq.strength, reverse=True)

        avg_border_val = np.sum(self.value_map * (self.border_map - self.combat_zone_map)) / np.sum(self.border_map - self.combat_zone_map)

        for s in idle_squares:
            if s.strength > 240:
                # High square strengths are necessary to either reinforce combat zones, or should be used to capture the closest cell as to hopefully lower production wastage.
                # If we're at a border, then we should try to capture the border cell
                if self.distance_from_border[s.x, s.y] == 1:
                    targets = [x for x in s.neighbors if (x.owner == 0 and self.production_map[x.x, x.y] > 2)]
                    targets.sort(key=lambda x: self.strength_map[x.x, x.y] / self.production_map_01[x.x, x.y])
                    if len(targets) > 0:
                        self.attack_cell(targets[0], 1)
                        self.value_map[targets[0].x, targets[0].y] += 2
                        continue

            if s.strength > (s.production * self.buildup) and s.moving_here == []:
                # we want to move towards the border square with the LOWEST value.
                # Gets base values for border squares
                if np.sum(self.is_owned_map) > 120:
                    d_map = self.distance_map_no_decay[s.x, s.y]
                else:
                    d_map = self.flood_fill_to_border([s])
                    d_map[d_map == -1] = 9999
                value_map = (self.value_map + d_map * 1.4) * self.border_map
                # Adjust combat squares
                # value_map[np.nonzero(self.combat_zone_map)] = 6
                # value_map[np.nonzero(self.combat_zone_map)] = avg_border_val * 2.25
                value_map[np.nonzero(self.combat_zone_map)] = avg_border_val * 5
                value_map += d_map * 2 * self.combat_zone_map
                value_map -= self.controlled_production_influence_map[5, s.x, s.y] * 2 * self.combat_zone_map

                # cells that we zeroed out are set to 9999. There's a small tiny chance that a square is actually worth 0. If so, oops
                value_map[value_map == 0] = 9999

                tx, ty = np.unravel_index(value_map.argmin(), (self.w, self.h))
                t = self.squares[tx, ty]

                # If we're closer to the border, enforce parity movement
                if self.distance_between(s, t) < 4 and self.distance_from_combat_zone[s.x, s.y] < 5:
                    if (s.x + s.y) % 2 != game.frame % 2:
                        continue

                if self.distance_between(s, t) > 14:
                    self.move_square_to_target_simple(s, t, True)
                elif self.distance_between(s, t) > 4:
                    self.move_square_to_target(s, t, True)
                elif self.distance_between(s, t) > 1:
                    if self.combat_zone_map[t.x, t.y]:
                        # if (s.x + s.y) % 2 == game.frame % 2:
                        self.move_square_to_target(s, t, True)
                        #else:
                        #    continue
                    else:
                        self.move_square_to_target(s, t, True)
                else:
                    if s.strength > t.strength:
                        # self.move_square_to_target(s, t, False)
                        self.attack_cell(t, 1)

                # If we can capture this cell, we shouldn't have other cells move here?
                if s.strength > t.strength:
                    self.value_map[t.x, t.y] += 2
                else:
                    self.value_map[t.x, t.y] += s.strength / 200

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

            free_squares = self.is_owned_map * (self.move_map == -1)

            target_distance_matrix = self.flood_fill([target], cells_out, True)
            target_distance_matrix[target_distance_matrix == -1] = 0
            target_distance_matrix = target_distance_matrix * free_squares
            available_strength = np.sum(self.strength_map * np.minimum(target_distance_matrix, 1))

            target_distance_matrix_production = cells_out - target_distance_matrix
            target_distance_matrix_production[target_distance_matrix_production == cells_out] = 0  # Cells furthest out would be moving so no production
            target_distance_matrix_production = target_distance_matrix_production * free_squares
            available_production = np.sum(self.production_map * target_distance_matrix_production)

            if (available_strength + available_production) > (target.strength + 0):
                attacking_cells_indices = np.transpose(np.nonzero(target_distance_matrix > 0))
                attacking_cells = [self.squares[c[0], c[1]] for c in attacking_cells_indices]

                still_cells = []
                if cells_out > 1:
                    still_cells_indices = np.transpose(np.nonzero(target_distance_matrix_production > 0))
                    still_cells = [self.squares[c[0], c[1]] for c in still_cells_indices]
                moving_cells = list(set(attacking_cells) - set(still_cells))

                for square in still_cells:
                    self.make_move(square, STILL)

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

    def make_move(self, square, direction, direction2=-1):
        self.move_map[square.x, square.y] = int(direction)
        self.move_map2[square.x, square.y] = int(direction2)

        if direction == -1:  # Reset the square move
            if square.target is not None:
                square.target.moving_here.remove(square)
                square.target = None
                # square.far_target = None
            square.move = -1
            # square.far_target = None
            return

        if square.move != -1:
            if square.target is not None:
                square.target.moving_here.remove(square)
                square.target = None
            # square.far_target = None

        square.move = direction
        if direction != STILL:
            square.target = square.neighbors[int(direction)]
            square.target.moving_here.append(square)
            # square.far_target = far_target

    def move_towards_map_old(self, square, distance_map, through_friendly=True):
        current_distance = distance_map[square.x, square.y]
        possible_moves = []
        for n in square.neighbors:
            if self.is_owned_map[n.x, n.y]:
                if distance_map[n.x, n.y] < current_distance:
                    possible_moves.append(n)
        if len(possible_moves) > 0:
            random.shuffle(possible_moves)
            possible_moves.sort(key=lambda sq: self.enemy_strength_map[4, sq.x, sq.y], reverse=True)
            self.move_square_to_target(square, possible_moves[0], True)

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
        path_choices.sort(key=lambda x: len(x[1].moving_here), reverse=True)

        # if len(path_choices) > 1:
        #     self.make_move(source, path_choices[0][0], path_choices[1][0])
        # else:
        #     self.make_move(source, path_choices[0][0])

        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength  # + target.production
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= 255 + strength_buffer:
                self.make_move(source, direction, -1)
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
                        self.make_move(source, direction, -1)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        for (n_d, n) in n_neighbors:
                            # n = target.neighbors[n_d]
                            if n.owner == self.my_id:
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
                        self.make_move(source, -1, -1)
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

        path_choices.sort(key=lambda x: x[1].production)
        path_choices.sort(key=lambda x: len(x[1].moving_here), reverse=True)
        # if len(path_choices) > 1:
        #     self.make_move(source, path_choices[0][0], path_choices[1][0])
        # else:
        #     self.make_move(source, path_choices[0][0])

        # Try simple resolution
        for (direction, target) in path_choices:
            future_strength = 0
            if target.owner == self.my_id:
                if target.move == -1 or target.move == STILL:
                    future_strength = target.strength  # + target.production
            for sq in target.moving_here:
                future_strength += sq.strength
            if future_strength + source.strength <= 255 + strength_buffer:
                self.make_move(source, direction, -1)
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
                        self.make_move(source, direction, -1)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        for (n_d, n) in n_neighbors:
                            # n = target.neighbors[n_d]
                            if n.owner == self.my_id:
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
                        self.make_move(source, -1, -1)

    def last_resort_strength_check(self):
        # Calculates the projected strength map and identifies squares that are violating it.
        # Ignore strength overloads due to production for now
        # Validate moves
        projected_strength_map = np.zeros((self.w, self.h))
        # We only care about our moves.
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.my_id:
                if square.move == -1 or square.move == STILL:
                    projected_strength_map[square.x, square.y] += square.strength  # + square.production
                else:
                    dx, dy = get_offset(square.move)
                    projected_strength_map[(square.x + dx) % self.w, (square.y + dy) % self.h] += square.strength

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
                        # Try attacking a bordering cell
                        if square.strength > 2 * n.strength and n.production > 1:
                            possible_paths.append((d, n, n.strength))

                possible_paths.sort(key=lambda x: x[2])
                # Force a move there
                self.make_move(square, d, -1)
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
                    self.make_move(opt[0], STILL, -1)
                # self.make_move(options_list[0][0], STILL, None)

        return violation_count


    def last_resort_strength_check_new(self):
        # Calculates the projected strength map and identifies squares that are violating it.
        # Ignore strength overloads due to production for now
        # Validate moves
        projected_strength_map = np.zeros((self.w, self.h))
        # We only care about our moves.
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.my_id:
                if square.move == -1 or square.move == STILL:
                    projected_strength_map[square.x, square.y] += square.strength  # + square.production
                else:
                    dx, dy = get_offset(square.move)
                    projected_strength_map[(square.x + dx) % self.w, (square.y + dy) % self.h] += square.strength

        # Get a list of squares that are over the cap
        violation_indices = np.transpose(np.nonzero((projected_strength_map > 255 + strength_buffer)))
        violation_squares = [self.squares[c[0], c[1]] for c in violation_indices]
        violation_squares.sort(key=lambda s: self.distance_from_border[s.x, s.y])
        violation_count = len(violation_squares)

        for s in violation_squares:
            if (s.move == -1 or s.move == STILL):
                if ((projected_strength_map[s.x, s.y] - s.strength) <= (255 + strength_buffer)) or (((s.strength > (s.production * self.buildup))) and ((s.x + s.y) % 2 == game.frame % 2)):
                    # Moving out of the way will solve this collision. Let's try not to cause another collision.
                    pp = []
                    for d in range(0, 4):
                        n = s.neighbors[d]
                        if n.owner == self.my_id and ((projected_strength_map[n.x, n.y] + s.strength) <= (255 + strength_buffer)):
                            pp.append((d, projected_strength_map[n.x, n.y] + s.strength))
                        elif n.owner == 0 and s.strength > (2 * n.strength) and n.production > 2:
                            # We're ok attacking a cell if it produces at least 3.
                            pp.append((d, 300 - (n.strength / n.production)))  # A hack to deprioritize attacking a neighbor
                    pp.sort(key=lambda x: x[1])
                    if len(pp) > 0:
                        self.make_move(s, pp[0][0])
                        # Update the projected strength map for the move
                        projected_strength_map[s.x, s.y] -= s.strength
                        dx, dy = get_offset(pp[0][0])
                        projected_strength_map[(s.x + dx) % self.w, (s.y + dy) % self.h] += s.strength
            if projected_strength_map[s.x, s.y] > (255 + strength_buffer):
                # We've moved THIS cell if it's slated to move but it's still causing a problem. We need to move a DIFFERENT cell.
                # Check to see if a square that's moving here has a secondary move that we can use.
                has_another_move_would_resolve = []
                would_resolve = []
                for mh in s.moving_here:
                    if self.move_map2[mh.x, mh.y] != -1:
                        # Will actually moving this cell by itself solve the problem?
                        if (projected_strength_map[s.x, s.y] - mh.strength) <= (255 + strength_buffer):
                            # Ok, is the alternative move a valid move that won't cause strength issues?
                            would_resolve.append(mh)
                            dx, dy = get_offset(self.move_map2[mh.x, mh.y])
                            if projected_strength_map[(mh.x + dx) % self.w, (mh.y + dy) % self.h] + mh.strength <= (255 + strength_buffer):
                                has_another_move_would_resolve.append(mh)
                if len(has_another_move_would_resolve) > 0:
                    # Moving this cell, if valid, will resolve any collision.
                    has_another_move_would_resolve.sort(key=lambda s: s.strength)
                    t = has_another_move_would_resolve[0]
                    self.make_move(t, self.move_map2[t.x, t.y], self.move_map[t.x, t.y])  # Preserve alternative move just in case.
                    projected_strength_map[s.x, s.y] -= t.strength
                    dx, dy = get_offset(self.move_map2[t.x, t.y])
                    projected_strength_map[(t.x + dx) % self.w, (t.y + dy) % self.h] += t.strength
                    continue
                # Ok, so we can't get any free wins. Identify the cells that if moved would cause this to be ok.
                if len(would_resolve) > 0:
                    # Moving one of these cells would fix the issue, if we can find a suitable location for it.
                    # Start with the smallest cells (Large strength cells get priority in where they go)
                    would_resolve.sort(key=lambda s: s.strength)
                    t = would_resolve[0]
                    # look at all neighbors.
                    valid_targets = []
                    for d in range(0, 4):
                        n = t.neighbors[d]
                        if n.owner == 0 and (projected_strength_map[n.x, n.y] + t.strength) < (255 + strength_buffer):
                            valid_targets.append((d, n))
                    if len(valid_targets) > 0:
                        valid_targets.sort(key=lambda s: projected_strength_map[s[1].x, s[1].y])
                        self.make_move(t, valid_targets[0][0])
                        projected_strength_map[s.x, s.y] -= t.strength
                        dx, dy = get_offset(valid_targets[0][0])
                        projected_strength_map[(t.x + dx) % self.w, (t.y + dy) % self.h] += t.strength
                        continue
                # Ok, MULTIPLE cells are causing the issue.
                # Force cells to take alternative routes, then force cells to stay still until we are ok.
                has_another_move = []
                for mh in s.moving_here:
                    if self.move_map2[mh.x, mh.y] != -1:
                        has_another_move.append(mh)
                has_another_move.sort(key=lambda s: s.strength)
                for t in has_another_move:
                    # Can we make the move?
                    d = self.move_map2[t.x, t.y]
                    dx, dy = get_offset(d)
                    if (projected_strength_map[(t.x + dx) % self.w, (t.y + dy) % self.h] + t.strength) <= 255 + strength_buffer:
                        # Make the move
                        self.make_move(t, d, self.move_map[t.x, t.y])
                        projected_strength_map[(t.x + dx) % self.w, (t.y + dy) % self.h] += t.strength
                        projected_strength_map[s.x, s.y] -= t.strength
                        if projected_strength_map[s.x, s.y] <= 255 + strength_buffer:
                            break
                # At this point, we've done everything we can. Force cells to STAY STILL until we are ok.
                if projected_strength_map[s.x, s.y] > 255 + strength_buffer:
                    # s.moving_here.sort(key=lambda x: x.strength)
                    for mh in s.moving_here:
                        self.make_move(mh, STILL, self.move_map[mh.x, mh.y])
                        projected_strength_map[mh.x, mh.y] += mh.strength
                        projected_strength_map[s.x, s.y] -= mh.strength
                        if projected_strength_map[s.x, s.y] <= 255 + strength_buffer:
                            break
        # if violation_count > 0:
            # np.savetxt("Maps/psmap_" + str(self.frame) + ".txt", projected_strength_map)
        return violation_count

    def consolidate_strength(self):
        # Attempts to move OFF squares into ON squares to consolidate strength.
        # Look at all squares that are about 5 squares from a combat zone.

        projected_strength_map = np.zeros((self.w, self.h))
        # We only care about our moves.
        for square in itertools.chain.from_iterable(self.squares):
            if square.owner == self.my_id:
                if square.move == -1 or square.move == STILL:
                    projected_strength_map[square.x, square.y] += square.strength  # + square.production
                else:
                    dx, dy = get_offset(square.move)
                    projected_strength_map[(square.x + dx) % self.w, (square.y + dy) % self.h] += square.strength

        indices = np.transpose(np.nonzero((projected_strength_map > 255 + strength_buffer)))
        squares = [self.squares[c[0], c[1]] for c in indices]

        for x in range(self.w):
            for y in range(self.h):
                sq = self.squares[x, y]
                if sq.owner == self.my_id:
                    if self.distance_from_combat_zone[sq.x, sq.y] < 6:
                        if (sq.x + sq.y) % 2 != (self.frame) % 2:
                            continue

                            # self.make_move(sq, STILL, self.move_map[sq.x, sq.y])
                        else:
                            # Try to consolidate.
                            if sq.move != -1 and sq.move != 4:
                                t = sq.target
                                if len(t.moving_here) == 1:
                                    tval = self.distance_from_combat_zone[t.x, t.y]
                                    # Check friendly neighbors
                                    for d in range(0, 4):
                                        n = sq.neighbors[d]
                                        if n != t and n.owner == self.my_id and self.distance_from_combat_zone[n.x, n.y] <= tval and len(n.moving_here) >= 1 and (projected_strength_map[n.x, n.y] + sq.strength) <= (255 + strength_buffer):
                                            self.move_square_to_target_simple(sq, n, True)
                                            projected_strength_map[n.x, n.y] += sq.strength
                                            projected_strength_map[sq.x, sq.y] -= sq.strength
                                            continue

# ==============================================================================
# Square class
# ==============================================================================


class Square:

    def __init__(self, game, x, y, production):
        self.game = game
        self.x = x
        self.y = y
        self.production = production
        self.h = game.h
        self.w = game.w
        self.vertex = x * self.h + y
        self.target = None
        self.moving_here = []
        self.far_target = None

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
        self.far_target = None

####################
# Helper Functions #
####################


def get_offset(direction):
    return ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))[int(direction)]


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


def spread_n(M, n, decay=0, include_self=True):
    # Takes a matrix M, and then creates an influence map by offsetting by N in every direction.
    # Decay function is currently of the form exp(-decay * distance)
    if include_self is True:
        spread_map = np.copy(M)
    else:
        spread_map = np.zeros_like(M)
    distance = 1
    if decay != 0:
        while distance <= n:
            combos = get_all_d_away(distance)
            decay_factor = math.exp(-decay * distance)
            for c in combos:
                spread_map += roll_xy(np.multiply(decay_factor, M), c[0], c[1])
            distance += 1
    else:  # decay = 0
        while distance <= n:
            combos = get_all_d_away(distance)
            for c in combos:
                spread_map += roll_xy(M, c[0], c[1])
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
    game.get_frame()
    # logging.debug("Frame: " + str(game.frame))

    game.update()

    game.get_moves()

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()

    # game.consolidate_strength()

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()

    collision_check = 998
    last_collision_check = 999
    while collision_check < last_collision_check:
        last_collision_check = collision_check
        collision_check = game.last_resort_strength_check()

    game.send_frame()


# #####################
# Game run-time code #
# #####################


logging.basicConfig(filename='logging.log', level=logging.DEBUG)
# logging.debug('your message here')
NORTH, EAST, SOUTH, WEST, STILL = range(5)
directions = [NORTH, EAST, SOUTH, WEST, STILL]

game = Game()


while True:
    game_loop()