from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("ImprovedBot2")

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
		return Move(location, NORTH if random.random() > 0.5 else WEST)
		
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

	