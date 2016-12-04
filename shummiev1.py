# 1st submission: Ranking is approximately 125ish
from hlt_shummiev1 import *

myID, game_map = get_init()
send_init("shummie v1")

def heuristic(cell):
    if cell.owner == 0:
        if cell.strength > 0:
            return cell.production / cell.strength
        else:
            # Check to see if there is anyone already moving here
            if len(game_map.get_all_moving_to(cell)) > 0:
                return 0
            else: 
                return cell.production
    else:
        total_damage = 0
        directions = [NORTH, EAST, SOUTH, WEST]
        random.shuffle(directions)
        for d in directions:
            target = game_map.get_target(cell, d)
            if target.owner != 0 and target.owner != myID:
                total_damage += target.strength
        return total_damage
        

def find_nearest_enemy_direction(square):

    direction = NORTH
    max_distance = min(game_map.width, game_map.height) / 2
    
    dir_distance = list()
    
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        location = game_map.get_target(square, d)
        while (location.owner == myID)and (distance < max_distance):
            distance += 1
            location = game_map.get_target(location, d)
        
        if distance < max_distance:
            target = game_map.get_target(square, d)
            direction = d
            max_distance = distance
    
    return direction
    
    
def get_move(square):
    buildup_multiplier = 5

    border = False    
    target_dir = None
    target_val = -1
    
    for d in (NORTH, EAST, SOUTH, WEST):
        target = game_map.get_target(square, d)
        if target.owner != myID:
            border = True
            val = heuristic(target)
            if val > target_val:
                target_val = val
                target_dir = d
    
    # If we can attack an enemy square, let's do so.
    if target_dir is not None and game_map.get_target(square, target_dir).strength < square.strength:
        game_map.make_move(square, target_dir)
    # If we don't have enough strength to make it worth moving yet, stay still
    elif square.strength < (square.production * buildup_multiplier):
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
            if game_map.owner_map[y, x] == myID:    # We only care about our cells.
                if game_map.next_uncapped_strength_map[y, x, myID] > (255 + strength_buffer):
                    cells_over.append((x, y))
    
    cells_over_count = len(cells_over)
    
    # cells_over contains a list of all cells that are over:
    while len(cells_over) > 0:
        x, y = cells_over.pop(0)        
        moving_into = []
        # Get a list of all squares that will be moving INTO this cell next turn.
        for direction in (NORTH, EAST, SOUTH, WEST):
            dx, dy = directions[direction]
            if game_map.owner_map[(y + dy) % game_map.height, (x + dx) % game_map.width] == myID: # only care about our cells moving into this square
                if game_map.move_map[(y + dy) % game_map.height, (x + dx) % game_map.width] == opposite_direction(direction):
                    moving_into.append(((x + dx) % game_map.width, (y + dy) % game_map.height))
        # Case 1: NO squares are moving into this cell AND we are not moving -- Going over due to overproduction. 
        if len(moving_into) == 0 and game_map.move_map[y, x] == STILL:
            # Move into the cell (even if it's staying still) that has the least future strength.
            # Check projected strength for next turn
            cell_strengths = [(STILL, game_map.production_map[y, x])]
            for direction in (NORTH, EAST, SOUTH, WEST):
                dx, dy = directions[direction]
                cell_strengths.append((direction, game_map.next_uncapped_strength_map[y, x, myID]))
            cell_strengths.sort(key = lambda tup: tup[1])            
            game_map.move_map[y, x] = cell_strengths[0][0]
        # Case 2: Squares are moving into this cell AND we are not moving - Move to the square with the least future strength?
        elif len(moving_into) > 0 and game_map.move_map[y, x] == STILL:
            # Will moving out of this square fix the problem?
            if game_map.next_uncapped_strength_map[y, x, myID] - game_map.strength_map[y, x] - game_map.production_map[y, x] < (255 + strength_buffer):
                # Yes it will, so let's move out into the lowest future strength square.
                cell_strengths = []
                for direction in (NORTH, EAST, SOUTH, WEST):
                    dx, dy = directions[direction]
                    cell_strengths.append((direction, game_map.next_uncapped_strength_map[(y + dy) % game_map.height, (x + dx) % game_map.width, myID]))
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
    for square in game_map:
        if square.owner == game_map.myID:
            if game_map.move_map[square.y, square.x] == STILL:
                # Is this a border cell?
                if square.strength < square.production * 5:
                    # Check for neighbors
                    # We need to check the following:
                    # 1: Are there any neighbors that are STILL and owned by us
                    available_mine = []
                    available_npc = []
                    for n in game_map.neighbors(square):
                        if n.owner == myID and game_map.move_map[n.y, n.x] == STILL:
                            available_mine.append(n)
                        elif n.owner == 0:
                            available_npc.append(n)
                    # We now have a list of neighbors that we can either GO to, or have come TO us.
                    # Let's keep this simple for now...
                    # Sort by strength.
                    available_mine.sort(key = lambda x: x.strength, reverse = True)
                    available_strength = sum(x.strength for x in available_mine) + square.strength
                    available_npc.sort(key = lambda x: heuristic(x), reverse = True)
                    target_found = False
                    while not target_found and len(available_npc) > 0:
                        test_target = available_npc.pop(0)
                        if test_target.strength < available_strength:
                            current_strength = square.strength
                            while current_strength < test_target.strength:
                                source = available_mine.pop(0)
                                current_strength += source.strength
                                game_map.move_to_target(source, square)
                                target_found = True    
    
while True:
    game_map.get_frame()
    
    # Have each individual square decide on their own movement
    for square in game_map:
        if square.owner == game_map.myID: 
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
    
