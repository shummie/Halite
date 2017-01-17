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
botname = "shummie v48-2-3"
strength_buffer = 0
print_maps = True


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
        # self.combat_radius = 6
        # self.production_cells_out = 15
        # self.phase = 0
        # Find the "global max"
        # self.global_max_square = None
        # self.total_avg_cost_to_global = 0

    def update_configs(self):
        self.buildup = 5

        # if np.sum(self.combat_zone_map) > 3:
        #     self.production_cells_out = int(self.w / self.starting_player_count / 2.5)

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
        # self.update_enemy_maps()
        # end = time.time()
        # logging.debug("update_enemymaps Frame: " + str(game.frame) + " : " + str(end - start))
        # start = time.time()
        # end = time.time()
        # logging.debug("update_recover Frame: " + str(game.frame) + " : " + str(end - start))
        # start = time.time()
        self.update_value_maps()
        # end = time.time()
        # logging.debug("update_value_maps Frame: " + str(game.frame) + " : " + str(end - start))

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



    def update_value_maps(self):
        base_value_map = np.divide(self.production_map_01, self.strength_map_1) * (self.is_neutral_map - self.combat_zone_map)
        # Each neutral cell gets assigned to the closest border non-combat cell
        global_targets_indices = np.transpose(np.nonzero(self.is_neutral_map - self.combat_zone_map))
        global_targets = [self.squares[c[0], c[1]] for c in global_targets_indices]
        # border_squares_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        # border_squares = [self.squares[c[0], c[1]] for c in border_squares_indices]
        global_border_map = np.zeros((self.w, self.h))

        for g in global_targets:
            # Find the closest border square that routes to g
            gb_map = self.dij_recov_distance_map[g.x, g.y] * (self.border_map - self.combat_zone_map)
            gb_map[gb_map == 0] = 9999
            tx, ty = np.unravel_index(gb_map.argmin(), (self.w, self.h))
            global_border_map[tx, ty] += base_value_map[g.x, g.y] / self.dij_recov_distance_map[g.x, g.y, tx, ty]

        self.value_map = 1 / np.maximum(base_value_map + global_border_map * 0.25, 0.001)
        print_map(global_border_map, "global_border_")
        print_map(base_value_map, "base_value_")
        print_map(self.value_map, "value_map_")


    def update_value_maps2(self):
        # Idea 2: All squares get assigned to a border square (closest) to add to it's value
        base_value_map = np.divide(self.production_map_01, self.strength_map_1) * (self.is_neutral_map - self.combat_zone_map)
        # base_value_map[base_value_map == 0] = -9999
        # recovery_cost_map = np.divide(self.strength_map, self.production_map_01) * (self.is_neutral_map - self.combat_zone_map)
        # Add in global contributions
        self.value_map = np.ones((self.w, self.h)) * 9999

        global_bonus = np.zeros((self.w, self.h))
        # potential_targets_indices = np.transpose(np.nonzero(self.border_map - self.combat_zone_map))
        # potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]
        logging.debug("Frame: " + str(self.frame))
        logging.debug(self.frame)
        for b in self:
            global_bonus[b.x, b.y] = np.sum(np.divide(base_value_map, np.maximum(self.dij_recov_distance_map[b.x, b.y], 1)))

            # self.value_map[b.x, b.y] = 1 / (base_value_map[b.x, b.y]) - (b_global_bonus) * 2
        self.value_map = 1 / (base_value_map + global_bonus)

        # self.value_map[self.value_map == 0] = 0.001
        print_map(global_bonus, "global_bonus_")
        print_map(1 / np.maximum(base_value_map, 0.01), "base_value_map_")
        print_map(self.value_map, "value_map_")

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

        self.attack_borders()
        self.move_inner_squares()

    def attack_borders(self):
        # get a list of border cells available for attack.
        # potential_targets_indices = np.transpose(np.nonzero(self.border_map))
        potential_targets_indices = np.transpose(np.nonzero(self.combat_zone_map))
        potential_targets = [self.squares[c[0], c[1]] for c in potential_targets_indices]

        # Sort targets by enemy strength then strength?
        potential_targets.sort(key=lambda x: self.distance_from_enemy[x.x, x.y])

        for square in potential_targets:
        #     if (square.x + square.y) % 2 == game.frame % 2:
        #         # Off parity square, don't force an attack (is this actually useful?)
        #         continue
            self.attack_cell(square, 1)

    def move_inner_squares(self):
        idle_squares_indices = np.transpose(np.nonzero((self.move_map == -1) * self.is_owned_map))
        idle_squares = [self.squares[c[0], c[1]] for c in idle_squares_indices]

        if len(idle_squares) == 0:
            return
        # Move squares closer to the border first.
        
        idle_squares.sort(key=lambda sq: sq.strength, reverse=True)
        idle_squares.sort(key=lambda sq: self.distance_from_border[sq.x, sq.y])

        avg_border_val = np.sum(self.value_map * (self.border_map - self.combat_zone_map)) / np.sum(self.border_map - self.combat_zone_map)
        
        for s in idle_squares:
            if s.strength > 240:
                # High square strengths are necessary to either reinforce combat zones, or should be used to capture the closest cell as to hopefully lower production wastage.
                # If we're at a border, then we should try to capture the border cell
                if self.distance_from_border[s.x, s.y] == 1:
                    targets = [x for x in s.neighbors if (x.owner == 0 and self.production_map[x.x, x.y] > 0)]
                    targets.sort(key=lambda x: self.strength_map[x.x, x.y] / self.production_map_01[x.x, x.y])
                    if len(targets) > 0:
                        self.attack_cell(targets[0], 1)
                        continue
                    
            if s.strength > (s.production * self.buildup) and s.moving_here == []:
                # we want to move towards the border square with the LOWEST value.
                # Gets base values for border squares
                if np.sum(self.is_owned_map) > 100:
                    d_map = self.distance_map_no_decay[s.x, s.y]
                else:
                    d_map = self.flood_fill_to_border([s])
                    d_map[d_map == -1] = 9999
                value_map = (self.value_map + d_map * 1.4) * self.border_map
                # Adjust combat squares                
                # value_map[np.nonzero(self.combat_zone_map)] = 6
                value_map[np.nonzero(self.combat_zone_map)] = avg_border_val / 3
                value_map += d_map * 1 * self.combat_zone_map

                # cells that we zeroed out are set to 9999. There's a small tiny chance that a square is actually worth 0. If so, oops
                value_map[value_map == 0] = 9999

                tx, ty = np.unravel_index(value_map.argmin(), (self.w, self.h))
                t = self.squares[tx, ty]

                # If we're closer to the border, enforce parity movement
                # if self.distance_between(s, t) < 3 and self.distance_from_combat_zone[s.x, s.y] < 4:
                #     if (s.x + s.y) % 2 != game.frame % 2:
                #         continue

                if self.distance_between(s, t) > 14:
                    self.move_square_to_target_simple(s, t, True)
                elif self.distance_between(s, t) > 4:
                    self.move_square_to_target(s, t, True)
                elif self.distance_between(s, t) > 1:
                    # if (s.x + s.y) % 2 == game.frame % 2:
                    self.move_square_to_target(s, t, True)
                else:
                    if s.strength > t.strength:
                        # self.move_square_to_target(s, t, False)
                        self.attack_cell(t, 2)
                        
                # If we can capture this cell, we shouldn't have other cells move here

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

    def make_move(self, square, direction):
        self.move_map[square.x, square.y] = direction

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
            square.target = square.neighbors[direction]
            square.target.moving_here.append(square)
            # square.far_target = far_target

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
                    if future_strength <= 255 + strength_buffer:
                        # Ok, let's move the target square.
                        self.make_move(source, direction)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        for (n_d, n) in n_neighbors:
                            # n = target.neighbors[n_d]
                            if n.owner == self.my_id and self.distance_from_combat_zone[n.x, n.y] > 3:
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
            if future_strength + source.strength <= 255 + strength_buffer:
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
            if future_strength + source.strength <= 255 + strength_buffer:
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
                    if future_strength <= 255 + strength_buffer:
                        # Ok, let's move the target square.
                        self.make_move(source, direction)  # Queue the move up, undo if it doesn't work
                        n_directions = list(range(4))
                        random.shuffle(n_directions)
                        n_neighbors = [(nd, target.neighbors[nd]) for nd in n_directions]
                        n_neighbors.sort(key=lambda x: x[1].production)
                        for (n_d, n) in n_neighbors:
                            # n = target.neighbors[n_d]
                            if n.owner == self.my_id and self.distance_from_combat_zone[n.x, n.y] > 3:
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
                        self.make_move(source, -1)
        # Nothing to do left
        return False

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
                        if square.strength > (2 * n.strength) and n.production > 2:
                            possible_paths.append((d, n, n.strength * 2))

                possible_paths.sort(key=lambda x: x[2])
                # Force a move there
                if len(possible_paths) > 0:
                    d, n = possible_paths[0][0], possible_paths[0][1]
                    self.make_move(square, d)
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
                    self.make_move(opt[0], STILL)
                # self.make_move(options_list[0][0], STILL, None)

        return violation_count

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