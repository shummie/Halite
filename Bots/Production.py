from hlt import *
from networking import *

myID, game_map = getInit()
sendInit("ProductionBot")


def find_nearest_enemy_direction(location):

	direction = NORTH
	max_distance = min(game_map.width, game_map.height) // 2
	
	for d in CARDINALS:
		distance = 0
		current = location
		site = game_map.getSite(current, d)
		while (site.owner == myID and distance < max_distance):
			distance += 1
			current = game_map.getLocation(current, d)
			site = game_map.getSite(current)
		
		if distance < max_distance:
			direction = d
			max_distance = distance
	
	return direction


def move(location):
	site = game_map.getSite(location)
	border = False
	
	target_dir = None
	target_prod = 0
	
	for d in CARDINALS:
		neighbor_site = game_map.getSite(location, d)
		if neighbor_site.owner != myID:
			border = True
			if neighbor_site.production > target_prod:
				target_prod = neighbor_site.production
				target_dir = d
			
	if target_dir is not None and neighbor_site.strength < site.strength:
		return Move(location, target_dir)
	
	if site.strength == 0:
		# No point moving a strength 0 unit... right?
		return Move(location, STILL)
	if site.strength < site.production * 5 :
		return Move(location, STILL)
	
	if not border:
		return Move(location, find_nearest_enemy_direction(location))
		
	return Move(location, STILL)



while True:
    moves = []
    game_map = getFrame()
    for y in range(game_map.height):
        for x in range(game_map.width):
            location = Location(x, y)
            if game_map.getSite(location).owner == myID:
                moves.append(move(location))
    sendFrame(moves)

	