package main

import (
    "bufio"
    "io"
    "os"
    "math"
    "strings"
    "strconv"
)

var botname := "shummie v47-7-1"

type Square struct {
    X, Y int
    Strength, Production, Owner int
    Vertex int
    Target *Square
    Moving_here [4]Square
    Width, Height int
    Game *Game
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
    MyID int
    ProductionMap [][]int
    Reader *bufio.Reader
    Writer io.Writer
    Squares [][]Square
}


type Direction int

func NewGame() Game {
    game := Game{
        reader: bufio.NewReader(os.Stdin),
        writer: os.Stdout,
    }
    game.MyID = game.getInt()
    game.deserializeMapSize()
    game.deserializeProductions()

    game.Squares = make([][]Square, game.width)
    for x := 0; x < game.width; x++ {
        game.Squares[x] = make([]Square, game.height)
        for y := 0; y < game.height; y++ {
            game.Squares[x][y] = Square{X = x, Y = y, Production = game.ProductionMap[x][y], Vertex = x * game.height + y, Width = game.width, Height = game.height, Game = &game}
        }
    }
    for _, squareX := range game.Squares {
        for _, square := range squareX {
            square.afterInitUpdate()
        }
    }

    game.Frame = -1

    game.OwnerMap = make([][]int, game.width)
    for x := 0; x < game.width; x++ {
        game.OwnerMap[x] = make([]int, game.height)
    }
    game.StrengthMap = make([][]int, game.width)
    for x := 0; x < game.width; x++ {
        game.StrengthMap[x] = make([]int, game.height)
    }
    game.MoveMap = make([][]int, game.width)
    for x := 0; x < game.width; x++ {
        game.MoveMap[x] = make([]int, game.height)
    }
    game.getFrame()

    game.StartingPlayerCount = Max2d(game.OwnerMap)

    game.MaxTurns = 10 * math.pow((game.width * game.height), 0.5)

    game.setConfigs()

    fmt.Println(botname)

}

func (g *Game) getFrame() {
    // Updates the map information from the latest frame provided by the game environment
    mapString = g.getString()

    // The state of the map (including owner and strength values, but excluding production values) is sent in the following way:
    // One integer, COUNTER, representing the number of tiles with the same owner consecutively.
    // One integer, OWNER, representing the owner of the tiles COUNTER encodes.
    // The above repeats until the COUNTER total is equal to the area of the map.
    // It fills in the map from row 1 to row HEIGHT and within a row from column 1 to column WIDTH.
    // Please be aware that the top row is the first row, as Halite uses screen-type coordinates.
    splitString := strings.Split(c.getString(), " ")


    var x, y, owner, counter int

    for y != g.Height {
        counter, splitString = int_str_array_pop(splitString)
        owner, splitString = int_str_array_pop(splitString)
        for a := 0; a < counter; a++ {
            g.OwnerMap[x][y] = owner
            x += 1
            if x == g.Width {
                x = 0
                y += 1
            }
        }
    }

    for y := 0; y < g.Height; y++ {
        for x := 0; x < g.Width; x++ {
            g.StrengthMap[x][y], splitString = int_str_array_pop(splitString)
            g.MoveMap[x][y] = -1  // Reset the move map
        }
    }

    g.frame += 1

}

func (g *Game) setConfigs() {
    g.Buildup = 5
    g.Phase = 0  // Temporary, we might get rid of this in this version
}


func (s *Square) afterInitUpdate() {
    // Should only be called after all squares are initialized
    s.North = &game.squares[x][(y - 1) % s.height]
    s.East = &game.squares[(x + 1) % s.width][y]
    s.South = &game.squares[x][(y + 1) % s.height]
    s.West = &game.squares[(x - 1) % s.width][y]
    s.Neighbors = []Square{s.North, s.East, s.South, s.West}  // doesn't include self.
}

func (s *Square) update(owner int, strength int) {
    s.Owner = owner
    s.Strength = strength
    s.resetMove()
}

func (s *Square) resetMove() {
    s.Move = -1
    s.Target = nil
    s.Moving_here = make([]Square, 4)
    s.FarTarget = nil  // Are we still going to use this?
}

func (g *Game) sendFrame() {
    var outString string
    for _, squareX := range g.Squares {
        for _, square := range squareX {
            if square.Owner == g.MyID {
                if square.Strength == 0 {  // Squares with 0 strength shouldn't move
                    square.Move = STILL
                }
                if square.Move == -1 {
                    // If we didn't actually assign a move, make sure it's still coded to STILL
                    square.Move = STILL
                }
                outString = fmt.Sprintf("%s %d %d %d", outString, square.X, square.Y, square.Move)
            }
        }
    }
    fmt.Println(outString)
}


func (g *Game) deserializeMapSize() {
    splitString := strings.Split(g.getString(), " ")
    g.width, splitString = int_str_array_pop(splitString)
    g.height, splitString = int_str_array_pop(splitString)
}

func (g *Game) deserializeProductions() {
    splitString := strings.Split(g.getString(), " ")

    yxproductions = make([][]int, g.height)
    for y := 0; y < c.height; y++ {
        yxproductions[y] = make([]int, g.width)
        for x := 0; x < c.width; x++ {
            yxproductions[y][x], splitString = int_str_array_pop(splitString)
        }
    }
    // Transpose the matrix so that we can work with it x, y style
    g.ProductionMap = make([][]int, g.width)
    for x := 0; x < g.height; x++ {
        g.ProductionMap[x] = make([]int, g.height)
        for y := 0; y < g.height; y++ {
            g.ProductionMap[x][y] = yxproductions[y][x]
        }
    }

}

func (g *Game) getString() string {
    retstr, _ := g.reader.ReadString('\n')
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

func getOffset(d Direction) (dx dy int) {
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
}

func oppositeDirection(d Direction) Direction {
    if d == STILL {
        return STILL
    }
    return (d + 2) % 4
}

func Max2d(array [][]int) int {
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

func Min2d(array [][]int) int {
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
