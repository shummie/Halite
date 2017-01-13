package main

import (
    "bufio"
    "fmt"
    "io"
    "log"
    "os"
    "math"
    "strings"
    "strconv"
)

var botname = "shummie v48-1-1"

type Square struct {
    X, Y int
    Strength, Production, Owner float64
    Vertex int
    Target *Square
    Width, Height int
    Game *Game
    North, South, East, West *Square
    Neighbors []*Square
    MovingHere []*Square
    Move int
    ResetStatus bool
}

type Direction int

const (
    NORTH Direction = iota
    EAST
    SOUTH
    WEST
    STILL
)


type Game struct {
    Width, Height int
    MyID, StartingPlayerCount float64
    MaxTurns float64
    Buildup float64
    MoveMap [][]float64
    ProductionMap, StrengthMap, OwnerMap [][]float64
    StrengthMap1, StrengthMap01, ProductionMap1, ProductionMap01 [][]float64
    IsOwnedMap, IsNeutralMap, IsEnemyMap [][]float64
    DistanceFromBorder, DistanceFromOwned, DistanceFromCombat [][]float64
    BorderMap, CombatZoneMap [][]float64
    DistanceMapNoDecay [][][][]float64
    Frame, Phase int
    Reader *bufio.Reader
    Writer io.Writer
    Squares [][]Square
}

func NewGame() Game {
    game := Game{
        Reader: bufio.NewReader(os.Stdin),
        Writer: os.Stdout,
    }
    game.MyID = float64(game.getInt())
    game.deserializeMapSize()
    game.deserializeProductions()

    game.Squares = make([][]Square, game.Width)
    for x := 0; x < game.Width; x++ {
        game.Squares[x] = make([]Square, game.Height)
        for y := 0; y < game.Height; y++ {
            game.Squares[x][y] = Square{X:x, Y:y, Production:game.ProductionMap[x][y], Vertex:x * game.Height + y, Width:game.Width, Height:game.Height, Game:&game}
        }
    }
    for _, squareX := range game.Squares {
        for _, square := range squareX {
            square.afterInitUpdate()
        }
    }

    game.Frame = -1

    game.OwnerMap = make([][]float64, game.Width)
    for x := 0; x < game.Width; x++ {
        game.OwnerMap[x] = make([]float64, game.Height)
    }
    game.StrengthMap = make([][]float64, game.Width)
    for x := 0; x < game.Width; x++ {
        game.StrengthMap[x] = make([]float64, game.Height)
    }
    game.MoveMap = make([][]float64, game.Width)
    for x := 0; x < game.Width; x++ {
        game.MoveMap[x] = make([]float64, game.Height)
    }
    game.getFrame()

    game.StartingPlayerCount = Max2d(game.OwnerMap)

    game.MaxTurns = 10 * math.Pow(float64(game.Width * game.Height), 0.5)

    game.setConfigs()

    fmt.Println(botname)

    return game

}

func (g *Game) getFrame() {
    // Updates the map information from the latest frame provided by the game environment
    mapString := g.getString()

    // The state of the map (including owner and strength values, but excluding production values) is sent in the following way:
    // One integer, COUNTER, representing the number of tiles with the same owner consecutively.
    // One integer, OWNER, representing the owner of the tiles COUNTER encodes.
    // The above repeats until the COUNTER total is equal to the area of the map.
    // It fills in the map from row 1 to row Height and within a row from column 1 to column Width.
    // Please be aware that the top row is the first row, as Halite uses screen-type coordinates.
    splitString := strings.Split(g.getString(), " ")


    var x, y, owner, counter int

    for y != g.Height {
        counter, splitString = int_str_array_pop(splitString)
        owner, splitString = int_str_array_pop(splitString)
        for a := 0; a < counter; a++ {
            g.OwnerMap[x][y] = float64(owner)
            x += 1
            if x == g.Width {
                x = 0
                y += 1
            }
        }
    }

    for y := 0; y < g.Height; y++ {
        for x := 0; x < g.Width; x++ {
            var strValue int
            strValue, splitString = int_str_array_pop(splitString)
            g.StrengthMap[x][y] = float64(strValue)
            g.MoveMap[x][y] = -1  // Reset the move map
        }
    }

    g.Frame += 1

}

func (g *Game) setConfigs() {
    g.Buildup = 5
    g.Phase = 0  // Temporary, we might get rid of this in this version
}

func (g *Game) createOneTimeMaps() {
    g.DistanceMapNoDecay = g.createDistanceMap(0)
    g.StrengthMap1 = MaxAcross2d(g.StrengthMap, 1)
    g.StrengthMap01 = MaxAcross2d(g.StrengthMap, 0.1)
    g.ProductionMap1 = MaxAcross2d(g.ProductionMap, 1)
    g.ProductionMap01 = MaxAcross2d(g.ProductionMap, 0.1)

    // g.createDijkstraMaps()
}

func (g *Game) createDistanceMap(decay float64) [][][][]float64 {
    // Creates a distance map so that we can easily divide a map to get ratios that we are interested in
    // self.distance_map[x, y, :, :] returns an array of (Width, Height) that gives the distance (x, y) is from (i, j) for all i, j
    // Note that the actual distance from x, y, to i, j is set to 1 to avoid divide by zero errors. Anything that utilizes this function should be aware of this fact.

    // Create the base map for 0, 0
    zz := make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        zz[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            dist_x := math.Min(float64(x), float64(-x % g.Width))
            dist_y := math.Min(float64(y), float64(-y % g.Height))
            dist := math.Max(dist_x + dist_y, 1)
            zz[x][y] = float64(dist * math.Exp(decay))
        }
    }

    distance_map := make([][][][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        distance_map[x] = make([][][]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            distance_map[x][y] = make([][]float64, g.Width)
            for i := 0; i < g.Width; i++ {
                distance_map[x][y][i] = make([]float64, g.Width)
            }
            distance_map[x][y] = roll_xy(zz, x, y)
        }
    }
    return distance_map
}

func (g *Game) update() {
    g.updateMaps()
    g.updateStats()
    g.updateConfigs()
}

func (g *Game) updateMaps() {
    g.updateCalcMaps()
    g.updateOwnerMaps()
    g.updateBorderMaps()
    g.updateDistanceMaps()
    // g.updateEnemyMaps()  We'll do this later...
}

func (g *Game) updateCalcMaps() {
    g.StrengthMap1 = MaxAcross2d(g.StrengthMap, 1)
    g.StrengthMap01 = MaxAcross2d(g.StrengthMap, 0.1)
}

func (g *Game) updateOwnerMaps() {
    g.IsOwnedMap = make([][]float64, g.Width)
    g.IsNeutralMap = make([][]float64, g.Width)
    g.IsEnemyMap = make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        g.IsOwnedMap[x] = make([]float64, g.Height)
        g.IsNeutralMap[x] = make([]float64, g.Height)
        g.IsEnemyMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            switch g.OwnerMap[x][y]{
            case 0:
                g.IsNeutralMap[x][y] = 1
            case g.MyID:
                g.IsOwnedMap[x][y] = 1
            default:
                g.IsEnemyMap[x][y] = 1
            }
        }
    }
}

func (g *Game) updateDistanceMaps() {
    borderSquares := make([]Square, 20)
    ownedSquares := make([]Square, 20)
    combatSquares := make([]Square, 20)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.Squares[x][y].Owner == 0 {
                borderSquares = append(borderSquares, g.Squares[x][y])
                if g.Squares[x][y].Strength == 0 {
                    combatSquares = append(combatSquares, g.Squares[x][y])
                }
            }
            if g.Squares[x][y].Owner == g.MyID {
                ownedSquares = append(ownedSquares, g.Squares[x][y])
            }
        }
    }
    g.DistanceFromBorder = g.floodFill(borderSquares, 999, true)
    g.DistanceFromOwned = g.floodFill(ownedSquares, 999, false)
    g.DistanceFromCombat = g.floodFill(combatSquares, 999, true)

}

func (g *Game) updateBorderMaps() {
    g.BorderMap = make([][]float64, g.Width)
    g.CombatZoneMap = make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        g.BorderMap[x] = make([]float64, g.Height)
        g.CombatZoneMap[x] = make([]float64, g.Width)
        for y := 0; y < g.Height; y++ {
            if g.Squares[x][y].Owner == 0 {
                for _, n := range g.Squares[x][y].Neighbors {
                    if n.Owner == g.MyID {
                        g.BorderMap[x][y] = 1
                        if n.Strength == 0 {
                            g.CombatZoneMap[x][y] = 1
                        }
                    }
                }
            }
        }
    }
}



func (g *Game) floodFill(sources []Square, maxDistance float64, friendly_only bool) [][]float64 {
    // Returns a [][]int that contains the distance to the source through friendly squares only.
    // -1 : Non friendly spaces or friendly spaces unreachable to sources through friendly squares
    // 0 : Source squares
    // >0 : Friendly square distance to closest source square.
    q := sources
    distanceMap := make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        distanceMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            distanceMap[x][y] = -1
        }
    }

    // Set all source squares to 0
    for _, square := range sources {
        distanceMap[square.X][square.Y] = 0
    }

    for ; len(q) > 0 ; {
        c := q[0]
        q = q[1:]
        currentDistance := distanceMap[c.X][c.Y]
        for _, n := range c.Neighbors {
            if (distanceMap[n.X][n.Y] == -1 || distanceMap[n.X][n.Y] > (currentDistance + 1)) {
                if (friendly_only && n.Owner == g.MyID) || (!friendly_only && n.Owner != g.MyID) {
                    distanceMap[n.X][n.Y] = currentDistance + 1
                    if currentDistance < maxDistance - 1 {
                        q := append(q, *n)
                    }
                }
            }
        }
    }
    return distanceMap
}


func roll_xy(mat [][]float64, ox int, oy int ) [][]float64 {
    // Offsets the map in the x axis by the # of spaces.
    x_len := len(mat)
    newMatrix := make([][]float64, x_len)
    for x := 0; x < x_len; x++ {
        y_len := len(mat[x])
        newMatrix[x] = make([]float64, y_len)
        for y := 0; y < y_len; y++ {
            newMatrix[x][y] = mat[(x - ox) % x_len][(y - oy) % y_len]
        }
    }
    return newMatrix
}

func roll_x(mat [][]float64, offset int) [][]float64 {
    // Offsets the map in the x axis by the # of spaces.
    x_len := len(mat)
    newMatrix := make([][]float64, x_len)
    for x := 0; x < x_len; x++ {
        y_len := len(mat[x])
        newMatrix[x] = make([]float64, y_len)
        for y := 0; y < y_len; y++ {
            newMatrix[x][y] = mat[(x - offset) % x_len][y]
        }
    }
    return newMatrix
}

func roll_y(mat [][]float64, offset int) [][]float64 {
    // Offsets the map in the x axis by the # of spaces.
    x_len := len(mat)
    newMatrix := make([][]float64, x_len)
    for x := 0; x < x_len; x++ {
        y_len := len(mat[x])
        newMatrix[x] = make([]float64, y_len)
        for y := 0; y < y_len; y++ {
            newMatrix[x][y] = mat[x][(y - offset) % y_len]
        }
    }
    return newMatrix
}


func (s *Square) afterInitUpdate() {
    // Should only be called after all squares are initialized
    s.North = &s.Game.Squares[s.X][(s.Y - 1) % s.Height]
    s.East = &s.Game.Squares[(s.X + 1) % s.Width][s.Y]
    s.South = &s.Game.Squares[s.X][(s.Y + 1) % s.Height]
    s.West = &s.Game.Squares[(s.X - 1) % s.Width][s.Y]
    s.Neighbors = []*Square{s.North, s.East, s.South, s.West}  // doesn't include self.
    s.ResetStatus = true
    s.MovingHere = make([]*Square, 4)
}

func (s *Square) update(owner float64, strength float64) {
    s.Owner = owner
    s.Strength = strength
    s.resetMove()
}

func (s *Square) resetMove() {
    s.Move = -1
    s.ResetStatus = true
    s.MovingHere = make([]*Square, 4)
}

func (g *Game) sendFrame() {
    var outString string
    for _, squareX := range g.Squares {
        for _, square := range squareX {
            if square.Owner == g.MyID {
                if square.Strength == 0 {  // Squares with 0 strength shouldn't move
                    square.Move = 4
                }
                if square.Move == -1 {
                    // If we didn't actually assign a move, make sure it's still coded to STILL
                    square.Move = 4
                }
                outString = fmt.Sprintf("%s %d %d %d", outString, square.X, square.Y, square.Move)
            }
        }
    }
    fmt.Println(outString)
}


func (g *Game) deserializeMapSize() {
    splitString := strings.Split(g.getString(), " ")
    g.Width, splitString = int_str_array_pop(splitString)
    g.Height, splitString = int_str_array_pop(splitString)
}

func (g *Game) deserializeProductions() {
    splitString := strings.Split(g.getString(), " ")

    yxproductions := make([][]int, g.Height)
    for y := 0; y < g.Height; y++ {
        yxproductions[y] = make([]int, g.Width)
        for x := 0; x < g.Width; x++ {
            yxproductions[y][x], splitString = int_str_array_pop(splitString)
        }
    }
    // Transpose the matrix so that we can work with it x, y style
    g.ProductionMap = make([][]float64, g.Width)
    for x := 0; x < g.Height; x++ {
        g.ProductionMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            g.ProductionMap[x][y] = float64(yxproductions[y][x])
        }
    }

}

func (g *Game) getString() string {
    retstr, _ := g.Reader.ReadString('\n')
    retstr = strings.TrimSpace(retstr)
    return retstr
}

func (g *Game) getInt() int {
    i, err := strconv.Atoi(g.getString())
    if err != nil {
        log.Printf("Whoopse", err)
    }
    return i
}

func int_str_array_pop(input []string) (int, []string) {
    ret, err := strconv.Atoi(input[0])
    input = input[1:]
    if err != nil {
        log.Printf("Whoopse", err)
    }
    return ret, input
}

func translate_cardinal(d Direction) Direction {
    // Cardinal index used by the framework is:
    // NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3, STILL = 4
    // Cardinal index used by the game is:
    // STILL = 0, NORTH = 1, EAST = 2, SOUTH = 3, WEST = 4
    return (d + 1) % 5
}

func getOffset(d Direction) (int, int) {
    switch d {
    case NORTH:
        return 0, -1
    case EAST:
        return 1, 0
    case SOUTH:
        return 0, 1
    case WEST:
        return -1, 0
    case STILL:
        return 0, 0
    }
    return 999, 999
}

func oppositeDirection(d Direction) Direction {
    if d == STILL {
        return STILL
    }
    return (d + 2) % 4
}

func Max2d(array [][]float64) float64 {
    // Takes a 2d slice and returns the max value
    max := array[0][0]
    for x := 0; x < len(array); x++ {
        for y := 0; y < len(array[x]); y++ {
            if array[x][y] > max {
                max = array[x][y]
            }
        }
    }
    return max
}

func Min2d(array [][]float64) float64 {
    // Takes a 2d slice and returns the min value
    min := array[0][0]
    for x := 0; x < len(array); x++ {
        for y := 0; y < len(array[x]); y++ {
            if array[x][y] < min {
                min = array[x][y]
            }
        }
    }
    return min
}


func MinAcross2d(array [][]float64, minVal float64) [][]float64 {
    // Returns an array which does a piecewise min between an array and a float64
    retArray := make([][]float64, len(array))
    for x:= 0; x < len(array); x++ {
        retArray[x] = make([]float64, len(array[x]))
        for y:= 0; y < len(array[x]); y++ {
            retArray[x][y] = math.Min(array[x][y], minVal)
        }
    }
    return retArray
}

func MaxAcross2d(array [][]float64, maxVal float64) [][]float64 {
    // Returns an array which does a piecewise max between an array and a float64
    retArray := make([][]float64, len(array))
    for x:= 0; x < len(array); x++ {
        retArray[x] = make([]float64, len(array[x]))
        for y:= 0; y < len(array[x]); y++ {
            retArray[x][y] = math.Max(array[x][y], maxVal)
        }
    }
    return retArray
}
