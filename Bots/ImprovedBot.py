from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("ImprovedBot")

def move(location):
	site = gameMap.getSite(location)
	for d in CARDINALS:
		neighbour_site = gameMap.getSite(location, d)
		if neighbour_site.owner != myID and neighbour_site.strength < site.strength:
			return Move(location, d)
	
	if site.strength == 0:
		# No point moving a strength 0 unit... right?
		return Move(location, STILL)
	elif site.strength < site.production * 5 :
		return Move(location, STILL)
	else:
		return Move(location, NORTH if random.random() > 0.5 else WEST)



while True:
    moves = []
    gameMap = getFrame()
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            location = Location(x, y)
            if gameMap.getSite(location).owner == myID:
                moves.append(move(location))
    sendFrame(moves)

	