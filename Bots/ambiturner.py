from hlt import *
import copy


myID, game_map = get_init()
send_init("ambiturner")

def find_nearest_enemy_direction(square):
	direction = NORTH
	max_distance = min(game_map.width, game_map.height) / 2
	
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
	
	# If we can attack an enemy square, let's do so.
	if target_dir is not None and game_map.get_target(square, target_dir).strength < square.strength:
		return make_move(square, target_dir)
	# If we don't have enough strength to make it worth moving yet, stay still
	if square.strength < (square.production * buildup_multiplier):
		return make_move(square, STILL)
	# If we aren't at a border, move towards the closest one
	if not border:
		return make_move(square, find_nearest_enemy_direction(square))
	# Else, we're at a border, don't have the strength to attack anyone adjacent, and have less than the buildup multiplier
	
	return make_move(square, STILL)
	
def make_move(square, direction):
	# Updates the next turn map, then returns the move object
	# game_map.make_move(square, direction)
	return Move(square, direction)

while True:
	game_map.get_frame()
	moves = [get_move(square) for square in game_map if square.owner == myID]
	send_frame(moves)
