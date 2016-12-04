from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("FindClosestImprovedBot")


def findNearestEnemyDirection(location):
	# Simple implementation. Searches all 4 direction, then moves in that direction.
	direction = NORTH
	maxDistance = min(gameMap.width, gameMap.height) // 2

	for d in CARDINALS:
		distance = 0
		current = location
		site = gameMap.getSite(current, d)
		while (site.owner == myID and distance < maxDistance):
			distance += 1
			current = gameMap.getLocation(current, d)
			site = gameMap.getSite(current)
		
		if distance < maxDistance:
			direction = d
			maxDistance = distance
			
	return direction



def move(location):
	site = gameMap.getSite(location)
	border = False
	
	for d in CARDINALS:
		neighbour_site = gameMap.getSite(location, d)
		if neighbour_site.owner != myID:
			border = True
			if neighbour_site.strength < site.strength:
				return Move(location, d)
	
	if site.strength == 0:
		# No point moving a strength 0 unit... right?
		return Move(location, STILL)
	if site.strength < site.production * 5 :
		return Move(location, STILL)
	
	if not border:
		return Move(location, findNearestEnemyDirection(location))
		
	return Move(location, STILL)



while True:
    moves = []
    gameMap = getFrame()
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            location = Location(x, y)
            if gameMap.getSite(location).owner == myID:
                moves.append(move(location))
    sendFrame(moves)

	