package main

import (
    "bufio"
    "container/heap"
    "fmt"
    "io"
    "log"
    "os"
    "math"
    "sort"
    "strings"
    "strconv"
    "time"
)

var Rx [][]int

var botname = "shummie v60-1-1-Go"

var Height, Width int

type Square struct {
    X, Y int
    Strength, Production, Owner float64
    Vertex int
    Target *Square
    Width, Height int
    Game *Game
    North, South, East, West *Square
    Neighbors []*Square
    MovingHere map[int]*Square
    Move int
    ResetStatus bool
    DistanceFromBorder, EnemyStrengthMap2, EnemyStrengthMap4 float64
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
    MaxTurns, TurnsLeft float64
    PercentOwned float64
    Buildup, StrengthBuffer, CombatRadius, PreCombatThreshold float64
    ProductionCellsOut int
    MoveMap [][]float64
    ProductionMap, StrengthMap, OwnerMap [][]float64
    StrengthMap1, StrengthMap01, ProductionMap1, ProductionMap01 [][]float64
    IsOwnedMap, IsNeutralMap, IsEnemyMap [][]float64
    DistanceFromBorder, DistanceFromOwned, DistanceFromCombat [][]float64
    BorderMap, CombatZoneMap [][]float64
    DistanceMapNoDecay [][][][]float64
    DijkstraRecoveryCosts [][][][]float64
    DijkstraRecoveryPaths [][][][]int
    DijkstraRecoveryCosts2 [][][][]float64
    DijkstraRecoveryPaths2 [][][][]int
    DijkstraRecoveryCosts3 [][][][]float64
    DijkstraRecoveryPaths3 [][][][]int
    DijkstraRecoveryDone3 [][][][]bool
    EnemyStrengthMap, OwnStrengthMap [][][]float64

    RecoveryCostMap, GlobalContributionMap, ValueMap [][]float64
    BaseValueMap, GlobalBorderMap [][]float64
    Frame, Phase int
    Reader *bufio.Reader
    Writer io.Writer
    Squares [][]*Square
}

func NewGame() Game {
    game := Game{
        Reader: bufio.NewReader(os.Stdin),
        Writer: os.Stdout,
    }
    game.MyID = float64(game.getInt())
    game.deserializeMapSize()
    Height = game.Height
    Width = game.Width
    game.deserializeProductions()

    game.Squares = make([][]*Square, game.Width)
    for x := 0; x < game.Width; x++ {
        game.Squares[x] = make([]*Square, game.Height)
        for y := 0; y < game.Height; y++ {
            game.Squares[x][y] = &Square{X:x, Y:y, Production:game.ProductionMap[x][y], Vertex:x * game.Height + y, Width:game.Width, Height:game.Height, Game:&game}
        }
    }
    for x := 0; x < game.Width; x++ {
        for y := 0 ; y < game.Height; y++ {
            game.Squares[x][y].afterInitUpdate()
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

    game.createOneTimeMaps()
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
    splitString := strings.Split(mapString, " ")


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
            g.Squares[x][y].update(g.OwnerMap[x][y], g.StrengthMap[x][y])
        }
    }

    g.Frame += 1
}

func (g *Game) setConfigs() {
    g.StrengthBuffer = 0
    g.Buildup = 5
    g.Phase = 0  // Temporary, we might get rid of this in this version
    g.CombatRadius = 8
    g.ProductionCellsOut = int(float64(g.Width) / g.StartingPlayerCount / 1.5)
    g.PreCombatThreshold = 2

}

func (g *Game) updateConfigs() {
    g.Buildup = 5
    g.CombatRadius = 8

    if Sum2d(g.CombatZoneMap) > 3 {
        g.ProductionCellsOut = int(float64(g.Width) / g.StartingPlayerCount / 2.5)
    }

    if g.PercentOwned > 0.6 {
        g.Buildup -= 1
        g.CombatRadius = 10
        g.PreCombatThreshold = 0
    }
    // else if
    // self.my_production_sum / self.next_highest_production_sum > 1.1:
    //        self.buildup_multiplier += 1

}

func (g *Game) createOneTimeMaps() {
    g.DistanceMapNoDecay = g.createDistanceMap(0)
    g.StrengthMap1 = MaxAcross2d(g.StrengthMap, 1)
    g.StrengthMap01 = MaxAcross2d(g.StrengthMap, 0.1)
    g.ProductionMap1 = MaxAcross2d(g.ProductionMap, 1)
    g.ProductionMap01 = MaxAcross2d(g.ProductionMap, 0.1)
    start := time.Now()
    g.createDijkstraMaps()

    end := time.Since(start)
    log.Println("Dijkstra took %s", end)

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
            dist_x := math.Min(float64(x), float64((g.Width-x) % g.Width))
            dist_y := math.Min(float64(y), float64((g.Height-y) % g.Height))
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
                distance_map[x][y][i] = make([]float64, g.Height)
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
    g.updateEnemyMaps()
    g.updateValueMaps()
    // g.updateControlledInfluenceProductonMaps()
    g.updateSquareValues()
}

func (g *Game) updateEnemyMaps() {
    maxRadius := 5
    g.EnemyStrengthMap = make([][][]float64, maxRadius + 1)
    g.OwnStrengthMap = make([][][]float64, maxRadius + 1)
    for d := 0; d <= maxRadius; d++ {
        g.EnemyStrengthMap[d] = make([][]float64, g.Width)
        g.OwnStrengthMap[d] = make([][]float64, g.Width)
        for x := 0; x < g.Width; x++ {
            g.EnemyStrengthMap[d][x] = make([]float64, g.Height)
            g.OwnStrengthMap[d][x] = make([]float64, g.Height)
            for y := 0; y < g.Height; y++ {
                if d == 0 {
                    g.EnemyStrengthMap[0][x][y] = g.IsEnemyMap[x][y] * g.StrengthMap[x][y]
                    g.OwnStrengthMap[0][x][y] = g.IsOwnedMap[x][y] * g.StrengthMap[x][y]
                }
            }
        }
    }

    for d := 1; d <= maxRadius; d++ {
        for x := 0; x < g.Width; x++ {
            // Does a "deep copy" of the array
            copy(g.EnemyStrengthMap[d][x], g.EnemyStrengthMap[d-1][x])
            copy(g.OwnStrengthMap[d][x], g.OwnStrengthMap[d-1][x])
        }
        for i := 0; i <= d; i++ {
            x := i
            y := d - i

            g.EnemyStrengthMap[d] = roll_xy_onto(g.EnemyStrengthMap[0], x, y, g.EnemyStrengthMap[d])
            g.OwnStrengthMap[d] = roll_xy_onto(g.OwnStrengthMap[0], x, y, g.OwnStrengthMap[d])

            if x != 0 {
                g.EnemyStrengthMap[d] = roll_xy_onto(g.EnemyStrengthMap[0], -x, y, g.EnemyStrengthMap[d])
                g.OwnStrengthMap[d] = roll_xy_onto(g.OwnStrengthMap[0], -x, y, g.OwnStrengthMap[d])
            }
            if y != 0 {
                g.EnemyStrengthMap[d] = roll_xy_onto(g.EnemyStrengthMap[0], x, -y, g.EnemyStrengthMap[d])
                g.OwnStrengthMap[d] = roll_xy_onto(g.OwnStrengthMap[0], x, -y, g.OwnStrengthMap[d])
            }
            if x != 0 && y != 0 {
                g.EnemyStrengthMap[d] = roll_xy_onto(g.EnemyStrengthMap[0], -x, -y, g.EnemyStrengthMap[d])
                g.OwnStrengthMap[d] = roll_xy_onto(g.OwnStrengthMap[0], -x, -y, g.OwnStrengthMap[d])
            }

        }
    }

}

func (g *Game) updateValueMaps() {

    // Calculate a recovery cost map for every neutral cell.
    g.RecoveryCostMap = make([][]float64, g.Width)
    g.BaseValueMap = make([][]float64, g.Width)
    g.GlobalBorderMap = make([][]float64, g.Width)
    g.ValueMap = make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        g.RecoveryCostMap[x] = make([]float64, g.Height)
        g.BaseValueMap[x] = make([]float64, g.Width)
        g.GlobalBorderMap[x] = make([]float64, g.Width)
        g.ValueMap[x] = make([]float64, g.Width)
        for y := 0; y < g.Height; y++ {
            if g.OwnerMap[x][y] == 0 {
                g.RecoveryCostMap[x][y] = g.StrengthMap[x][y] / g.ProductionMap01[x][y]
                if g.CombatZoneMap[x][y] == 0 {
                    g.BaseValueMap[x][y] = g.StrengthMap1[x][y] / g.ProductionMap01[x][y]
                }
            }
        }
    }

    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.OwnerMap[x][y] == 0 && g.CombatZoneMap[x][y] == 0 {
                // This is a neutral global square that we need to assign to the closest border.

                minVal := 9999.0
                minx, miny := -1, -1
                for i := 0; i < g.Width; i++ {
                    for j := 0; j < g.Height; j++ {
                        if g.BorderMap[i][j] == 1 && g.CombatZoneMap[i][j] == 0 {
                            if g.DijkstraRecoveryCosts[x][y][i][j] < minVal {
                                minVal = g.DijkstraRecoveryCosts[x][y][i][j]
                                minx, miny = i, j
                            }
                        }
                    }
                }
                g.GlobalBorderMap[minx][miny] += g.BaseValueMap[x][y] / minVal
            }
        }
    }

    turns_left := g.MaxTurns - float64(g.Frame)
    recover_threshold := turns_left * 0.6

    avg_recov_threshold := 2.0
    avg_map_recovery := Sum2d(Multiply2d(g.StrengthMap, g.BorderMap)) / Sum2d(Multiply2d(g.ProductionMap, g.BorderMap))


    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y ++ {
            g.ValueMap[x][y] = 1 / math.Max(g.BaseValueMap[x][y] + g.GlobalBorderMap[x][y] * 1, 0.001)
            if g.BorderMap[x][y] == 0 || g.CombatZoneMap[x][y] == 1 || g.EnemyStrengthMap[1][x][y] > 0 {
                g.ValueMap[x][y] = 9999
            }

            if g.ValueMap[x][y] > recover_threshold || g.ValueMap[x][y] > (avg_recov_threshold * avg_map_recovery) {
                g.ValueMap[x][y] = 9999
            }
        }
    }
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
    borderSquares := make([]*Square, 0, 20)
    ownedSquares := make([]*Square, 0, 20)
    combatSquares := make([]*Square, 0, 20)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.Squares[x][y].Owner == 0 {
                isBorder := false
                for j := 0; j < 4; j++ {
                    if g.Squares[x][y].Neighbors[j].Owner == g.MyID {
                        isBorder = true
                    }
                }
                if isBorder == true {
                    borderSquares = append(borderSquares, g.Squares[x][y])
                    if g.Squares[x][y].Strength == 0 {
                       combatSquares = append(combatSquares, g.Squares[x][y])
                    }
                }

            }
            if g.Squares[x][y].Owner == g.MyID {
                ownedSquares = append(ownedSquares, g.Squares[x][y])
            }
        }
    }
    g.DistanceFromBorder = g.floodFill(borderSquares, 999, true)
    log.Println(g.DistanceFromBorder)
    g.DistanceFromOwned = g.floodFill(ownedSquares, 999, false)
    g.DistanceFromCombat = g.floodFill(combatSquares, 999, true)
}

func (g *Game) updateBorderMaps() {
    g.BorderMap = make([][]float64, g.Width)
    g.CombatZoneMap = make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        g.BorderMap[x] = make([]float64, g.Height)
        g.CombatZoneMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            if g.Squares[x][y].Owner == 0 {
                for _, n := range g.Squares[x][y].Neighbors {
                    if n.Owner == g.MyID {
                        g.BorderMap[x][y] = 1
                        if g.Squares[x][y].Strength == 0 {
                            g.CombatZoneMap[x][y] = 1
                        }
                    }
                }
            }
        }
    }
}

func (g *Game) updateSquareValues() {
    // Updates square values for sorting purposes. Add values here for more sorts
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            g.Squares[x][y].DistanceFromBorder = g.DistanceFromBorder[x][y]
            g.Squares[x][y].EnemyStrengthMap2 = g.EnemyStrengthMap[2][x][y]
            g.Squares[x][y].EnemyStrengthMap4 = g.EnemyStrengthMap[4][x][y]
        }
    }

}

func (g *Game) updateStats() {
    g.TurnsLeft = g.MaxTurns - float64(g.Frame)
    g.PercentOwned = Sum2d(g.IsOwnedMap) / float64(g.Width * g.Height)
    //     self.production_values = [0]
    // for i in range(1, self.starting_player_count + 1):
    //     self.production_values.append(np.sum(self.production_map * (self.owner_map == i)))
    // self.my_production_sum = self.production_values[self.my_id]
    // temp_production_sum = copy.copy(self.production_values)
    // temp_production_sum.pop(self.my_id)
    // temp_production_sum.pop(0)
    // self.next_highest_production_sum = max(temp_production_sum)

}

func (g *Game) getMoves() {
    // Main logic controlling code
    // g.attackBorders()
    // g.moveInnerSquares()
    // g.eachSquareMoves()


    // Find super high production cells
    // g.getPreCombatProduction()
    // 1 - Find combat zone cells and attack them.
    log.Println("move - attack")
    g.getMovesAttack()
    log.Println("move - preparestr")
    // 2 - Build up strength
    g.getMovesPrepareStrength()
    log.Println("move - prod")
    // 3 - Find production zone cells and attack them
    g.getMovesProduction()
    log.Println("move - oter")
    // 4 - Move all other unassigned cells.
    g.getMovesOther()
}

func (g *Game) getMovesAttack() {

    combatZoneSquares := make([]*Square, 0, g.Width)

    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.CombatZoneMap[x][y] == 1 {
                combatZoneSquares = append(combatZoneSquares, g.Squares[x][y])
            }
        }
    }

    // Attack into stronger enemy territory first.
    if len(combatZoneSquares) == 0 {
        return
    }
    sort.Sort(sort.Reverse(ByEnemyStrength2(combatZoneSquares)))

    // Does this "maximize" overkill damage? Is there a better way to do it? Send in smaller squares potentially?
    for _, sq := range combatZoneSquares {
        g.attackCell(sq, 1)
    }

    g.getMovesBreakthrough()

    // Get a list of all squares within combatRadius spaces of a combat zone
    // TODO: This causes some bounciness, floodfill from all combat zone squares instead?
    combatDistanceMatrix := g.floodFill(combatZoneSquares, g.CombatRadius, true)
    combatSquares := make([]*Square, 0, g.Width)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if combatDistanceMatrix[x][y] > 0 {
                combatSquares = append(combatSquares, g.Squares[x][y])
            }
        }
    }
    sort.Sort(sort.Reverse(ByStrength(combatSquares)))

    for _, sq := range combatSquares {
        if sq.Strength > 0 && combatDistanceMatrix[sq.X][sq.Y] == 1 && (sq.Move == -1 || sq.Move == 4) {
            targets := make([]*Square, 0, 4)
            altTargets := make([]*Square, 0, 4)

            for _, n := range sq.Neighbors {
                if n.Owner == 0 && n.Strength == 0 {
                    targets = append(targets, n)
                } else if n.Owner == g.MyID {
                    altTargets = append(altTargets, n)
                }
            }
            sort.Sort(sort.Reverse(ByEnemyStrength2(targets)))
            sort.Sort(ByStrength(altTargets))
            success := false
            for _, t := range targets {
                success = g.moveSquareToTargetSimple(sq, t, false)
                if success {
                    break
                }
            }
            if !success {
                for _, t := range altTargets {
                    success = g.moveSquareToTargetSimple(sq, t, true)
                    if success {
                        break
                    }
                }
            }
        } else if sq.Strength > (sq.Production * (g.Buildup + g.DistanceFromCombat[sq.X][sq.Y])) && (((sq.X + sq.Y) % 2) == (g.Frame % 2)) && sq.Move == -1 && len(sq.MovingHere) == 0 {
            g.moveTowardsMapOld(sq, combatDistanceMatrix)
        } else {
            if combatDistanceMatrix[sq.X][sq.Y] > 1 {
                g.makeMove(sq, 4)
            }
        }
    }
}

func (g *Game) getMovesBreakthrough() {
    // Determine if we should bust through and try to open up additional lanes of attack into enemy territory
    // Best to have a separate lane. so we should evaluate squares that are not next to already open channels.
    // We are only looking at squares which are next to the enemy already.
    // TODO: This entire function needs to be relooked at / rewritten
    potentialSquares := make([]*Square, 0, g.Width)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.BorderMap[x][y] == 1 && g.CombatZoneMap[x][y] == 0 && g.EnemyStrengthMap[1][x][y] > 0 {
                potentialSquares = append(potentialSquares, g.Squares[x][y])
            }
        }
    }

    for _, sq := range potentialSquares {
        if g.OwnStrengthMap[4][sq.X][sq.Y] > 750 && g.OwnStrengthMap[4][sq.X][sq.Y] > 1.5 * g.EnemyStrengthMap[4][sq.X][sq.Y] {
            g.attackCell(sq, 1)
        }
    }
}


func (g *Game) moveTowardsMapOld(sq *Square, distMap [][]float64) {
    // Note, this shouldn't be used to attack a combat zone square.
    cDist := distMap[sq.X][sq.Y]
    possibleMoves := make([]*Square, 0, 4)

    for _, n := range sq.Neighbors {
        if g.IsOwnedMap[n.X][n.Y] == 1 {
            if distMap[n.X][n.Y] <= cDist - 1 {
                possibleMoves = append(possibleMoves, n)
            }
        }
    }

    if len(possibleMoves) > 0 {
        sort.Sort(sort.Reverse(ByEnemyStrength2(possibleMoves)))
        sort.Sort(sort.Reverse(ByEnemyStrength4(possibleMoves)))
        g.moveSquareToTarget(sq, possibleMoves[0], true)
    }

}

func (g *Game) findNearestNonOwnedBorder(sq *Square) {
    cDist := g.DistanceFromBorder[sq.X][sq.Y]
    targets := make([]*Square, 0, 4)
    for _, n := range sq.Neighbors {
        if n.Owner == g.MyID && g.DistanceFromBorder[n.X][n.Y] < cDist {
            targets = append(targets, n)
        }
    }
    sort.Sort(ByProduction(targets))
    for _, t := range targets {
        success := g.moveSquareToTarget(sq, t, true)
        if success {
            break
        }
    }
}

func (g *Game) getMovesPrepareStrength() {
    // Attempts to build up strength prior to an immediate engagement, only if we aren't already in combat
    enemyBorderSquares := make([]*Square, 0, g.Width)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.BorderMap[x][y] == 1 && g.EnemyStrengthMap[1][x][y] > 0 {
                enemyBorderSquares = append(enemyBorderSquares, g.Squares[x][y])
            }
        }
    }
    if len(enemyBorderSquares) > 0 {
        combatDistanceMatrix := g.floodFill(enemyBorderSquares, 5, true)
        combatSquares := make([]*Square, 0, g.Width)
        for x := 0; x < g.Width; x++ {
            for y := 0; y < g.Height; y++ {
                if combatDistanceMatrix[x][y] == -1 {
                    combatDistanceMatrix[x][y] = 0
                }
                if combatDistanceMatrix[x][y] > 0 {
                    combatSquares = append(combatSquares, g.Squares[x][y])
                }
            }
        }
        for _, sq := range combatSquares {
            if g.DistanceFromBorder[sq.X][sq.Y] > 3 && sq.Strength > (sq.Production * g.Buildup + 5) && (sq.X + sq.Y) % 2 == g.Frame % 2 && sq.Move == -1 && len(sq.MovingHere) == 0 {
                g.moveTowardsMapOld(sq, combatDistanceMatrix)
            } else if (sq.Strength >= 240) && g.OwnStrengthMap[2][sq.X][sq.Y] >= 750 && combatDistanceMatrix[sq.X][sq.Y] == 1 {
                // Attack!
                targets := make([]*Square, 0, 4)
                for _, n := range sq.Neighbors {
                    if combatDistanceMatrix[n.X][n.Y] == 0 {
                        targets = append(targets, n)
                    }
                }
                sort.Sort(sort.Reverse(ByEnemyStrength2(targets)))  // Old version used ByEnemyStrength1, two should produce similar results.
                g.moveSquareToTargetSimple(sq, targets[0], false)
            } else if (sq.Move == -1) {
                g.makeMove(sq, 4)
            }
        }
    }
}

type ProductionChoice struct {
    sq *Square
    value float64
    cellsOut float64
}

type ByStrengthPC []*ProductionChoice

func (s ByStrengthPC) Len() int {
    return len(s)
}

func (s ByStrengthPC) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByStrengthPC) Less(i, j int) bool {
    return s[i].sq.Strength < s[j].sq.Strength
}

type ByTotalCostPC []*ProductionChoice

func (s ByTotalCostPC) Len() int {
    return len(s)
}

func (s ByTotalCostPC) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByTotalCostPC) Less(i, j int) bool {
    dPenalty := 1.0
    return s[i].value + s[i].cellsOut * dPenalty < s[j].value + s[j].cellsOut * dPenalty
}

func (g *Game) getMovesProduction() {
    // Tries to find the best cells to attack from a production standpoint.
    // Does not try to attack cells that are in combat zones.

    targets := make([]*ProductionChoice, 0, g.Width)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.ValueMap[x][y] < 8000 {
                for cells := 1; cells <= g.ProductionCellsOut; cells++ {
                    targets = append(targets, &ProductionChoice{sq: g.Squares[x][y], value: g.ValueMap[x][y], cellsOut: float64(cells)})
                }
            }
        }
    }
    log.Println("1")
    if len(targets) == 0 {
        return
    }
    sort.Sort(ByStrengthPC(targets))
    sort.Sort(ByTotalCostPC(targets))   // Distance penalty mod in sort function
    // Keep only the top X %ile?
    percentile := 0.85
    cutIndex := int(float64(len(targets)) * percentile + 0.5)
    removeTargets := targets[cutIndex:]
    targets = targets[:cutIndex]
    for _, rt := range removeTargets {
        g.ValueMap[rt.sq.X][rt.sq.Y] = 9999 // We don't want squares to move towards these "worthless" targets later on.
    }
    for ; len(targets) > 0 ; {
        log.Println(len(targets))
        t := targets[0]
        targets = targets[1:]
        success := g.attackCell(t.sq, int(t.cellsOut))
        if success && int(t.cellsOut) < g.ProductionCellsOut {
            // Remove all other instances of this square from the list.
            newtargets := make([]*ProductionChoice, 0, len(targets))
            for _, a := range targets {
                if a.sq != t.sq {
                    newtargets = append(newtargets, a)
                }
            }
            targets = newtargets
        }
    }

}

func (g *Game) getMovesOther() {
    idleSquares := make([]*Square, 0, g.Width)
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.MoveMap[x][y] == -1 && g.IsOwnedMap[x][y] == 1 {
                idleSquares = append(idleSquares, g.Squares[x][y])
            }
        }
    }
    if len(idleSquares) == 0 {
        return
    }

    // Move squares closer to the border first.
    if len(idleSquares) > 1 {
       sort.Sort(ByBorderDistance(idleSquares))
    }

    for _, s := range idleSquares {
        if s.Strength > s.Production * g.Buildup && len(s.MovingHere) == 0 {
            if g.PercentOwned > 0.65 {  // I wonder if this is still necessary in go.
                g.findNearestNonOwnedBorder(s)
            } else {
                bestTargetValue := 10000.0
                bx := -1
                by := -1
                for x := 0; x < g.Width; x++ {
                    for y := 0; y < g.Height; y++ {
                        tValue := 0.0
                        if g.BorderMap[x][y] == 1 && g.CombatZoneMap[x][y] == 0 {
                            tValue = g.ValueMap[x][y] + g.DistanceMapNoDecay[s.X][s.Y][x][y] * 1.0
                        } else if g.CombatZoneMap[x][y] == 1 {
                            tValue = g.DistanceMapNoDecay[s.X][s.Y][x][y] * 0.66
                        } else {
                            tValue = 9999.0
                        }
                        if tValue < bestTargetValue {
                            bestTargetValue = tValue
                            bx, by = x, y
                        }
                    }
                }
                b := g.Squares[bx][by]

                // We're targeting either a combat square or a production square. Don't move towards close production squares
                if g.distanceBetween(s, b) < 6 && g.DistanceFromCombat[s.X][s.Y] < 7 {
                    if (s.X + s.Y) % 2 != g.Frame % 2 {
                        continue
                    }
                }
                if g.EnemyStrengthMap[3][s.X][s.Y] > 0 && (s.X + s.Y) % 2 != g.Frame % 2 {
                    g.makeMove(s, 4)
                } else if g.CombatZoneMap[bx][by] == 1 {
                    if g.distanceBetween(s, b) > 14 {
                        g.moveSquareToTargetSimple(s, b, true)
                    } else if g.distanceBetween(s, b) > 1 {
                        g.moveSquareToTarget(s, b, true)
                    }
                } else {
                    if g.distanceBetween(s, b) > 14 {
                        g.moveSquareToTargetSimple(s, b, true)
                    } else if g.distanceBetween(s, b) > float64(g.ProductionCellsOut - 1) {
                        g.moveSquareToTarget(s, b, true)
                    }
                }
            }
        }
    }
}

func (g *Game) attackBorders() {

    // For now, a simple attack border code.
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if g.BorderMap[x][y] == 1 {
                square := g.Squares[x][y]
                g.attackCell(square, 1)
            }
        }
    }
}

func (g *Game) moveInnerSquares() {
    // Simple function. just move towards highest value border.

    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            square := g.Squares[x][y]
            if square.Owner == g.MyID && square.Move == -1 && square.Strength > (square.Production * g.Buildup) {
                var target *Square
                targetVal := 9999.0
                for i := 0; i < g.Width; i++ {
                    for j := 0; j < g.Height; j++ {
                        // Note, need to add some sort of distance modifier.
                        if g.BorderMap[i][j] == 1 && targetVal > g.ValueMap[i][j] {
                            targetVal = g.ValueMap[i][j]
                            target = g.Squares[i][j]
                        }
                    }
                }
                g.moveSquareToTarget(g.Squares[x][y], target, true)
            }
        }
    }
}

func (g *Game) attackCell(target *Square, maxDistance int) bool {
    // Attempts to coordinate attacks to the target Square by calling cells distance out.
    for cellsOut := 1.0; cellsOut <= float64(maxDistance); cellsOut++ {
        // Don't attempt to coordinate a multi-cell attack into a combat zone
        if cellsOut > 1 && g.CombatZoneMap[target.X][target.Y] == 1 {
            return false
        }

        movingCells := make([]*Square, 0, 5)
        stillCells := make([]*Square, 0, 5)
        targetDistMap := g.floodFill([]*Square{target}, cellsOut, true)
        availableStrength := 0.0
        availableProduction := 0.0
        stillStrength := 0.0
        for x := 0; x < g.Width; x++ {
            for y := 0; y < g.Height; y++ {
                if targetDistMap[x][y] == -1 {
                    targetDistMap[x][y] = 0
                } else if g.IsOwnedMap[x][y] != 1 || g.MoveMap[x][y] != -1 || g.StrengthMap[x][y] < g.Buildup * g.ProductionMap[x][y] {
                    targetDistMap[x][y] = 0
                } else if targetDistMap[x][y] > 0 {
                    availableStrength += g.StrengthMap[x][y]
                    availableProduction += (cellsOut - targetDistMap[x][y]) * g.ProductionMap[x][y]
                    if (cellsOut - targetDistMap[x][y]) > 0 {
                        stillCells = append(stillCells, g.Squares[x][y])
                        stillStrength += g.StrengthMap[x][y]
                    } else {
                        movingCells = append(movingCells, g.Squares[x][y])
                    }
                }
            }
        }

        if (availableProduction + availableStrength) > target.Strength {
            for _, square := range stillCells {
                g.makeMove(square, 4)
            }
            neededStrengthFromMovers := target.Strength - availableProduction - stillStrength + 1

            if neededStrengthFromMovers > 0 {
                sort.Sort(sort.Reverse(ByStrength(movingCells)))
                for _, square := range movingCells {
                    if square.Strength > 0 {
                        if cellsOut == 1 {
                            g.moveSquareToTarget(square, target, false)
                        } else {
                            g.moveSquareToTarget(square, target, true)
                        }
                        neededStrengthFromMovers -= square.Strength
                        if neededStrengthFromMovers < 0 {
                            break
                        }
                    }
                }
            }
            return true
        } else {
            cellsOut += 1
        }
    }
    return false
}

func (g *Game) moveSquareToTarget(s *Square, d *Square, throughFriendly bool) bool {
    // Does a "simple" movement based on a BFS.
    distanceMap := g.floodFillToTarget(s, d, throughFriendly)
    sDist := distanceMap[s.X][s.Y]
    if sDist == -1 || sDist == 0 {
        // Couldn't find a path to the destination or trying to move STILL
        return false
    }

    pathChoices := make([]int, 0, 2)
    for dir := 0; dir < 4; dir++ {
        n := s.Neighbors[dir]
        if distanceMap[n.X][n.Y] == (sDist - 1) {
            pathChoices = append(pathChoices, dir)
        }
    }

    // There should be at most 2 cells in pathChoices.
    if len(pathChoices) == 2 {
        d1x, d1y := getOffset(pathChoices[0])
        d2x, d2y := getOffset(pathChoices[1])
        x1 := (s.X + d1x + g.Width) % g.Width
        y1 := (s.Y + d1y + g.Height) % g.Height
        x2 := (s.X + d2x + g.Width) % g.Width
        y2 := (s.Y + d2y + g.Height) % g.Height
        if g.Squares[x1][y1].Production > g.Squares[x2][y2].Production{
            pathChoices[0], pathChoices[1] = pathChoices[1], pathChoices[0]
        }
    }

    // Strength collision code goes here.

    // Try simple resolution
    for _, dir := range pathChoices {
        futureStrength := 0.0
        t := s.Neighbors[dir]
        if t.Owner == g.MyID && (t.Move == -1 || t.Move == 4) {
            futureStrength += t.Strength  // + t.Production
        }
        for _, sq := range t.MovingHere {
            futureStrength += sq.Strength
        }
        if futureStrength + s.Strength <= 255 + g.StrengthBuffer {
            g.makeMove(s, dir)
            return true
        }
    }

    for _, dir := range pathChoices {
        t := s.Neighbors[dir]
        // Can we move the cell we are moving to?
        if t.Owner == g.MyID && (t.Move == -1 || t.Move == 4) {     // Should we allow this to move STILL pieces?? Maybe test later
            futureStrength := s.Strength
            for _, sq := range t.MovingHere {
                futureStrength += sq.Strength
            }
            if futureStrength <= 255 + g.StrengthBuffer {
                // Ok, we can move the target square and be ok.
                g.makeMove(s, dir)  // Queue the move up, undo if it doesn't work
                nNeighbors := []*Square{t.Neighbors[0], t.Neighbors[1], t.Neighbors[2], t.Neighbors[3]}

                sort.Sort(ByProduction(nNeighbors))
                sort.Sort(ByBorderDistance(nNeighbors))

                for _, nn := range nNeighbors {
                    if nn.Owner == g.MyID && g.EnemyStrengthMap[2][nn.X][nn.Y] == 0 {
                        // Can we move into this square safely?
                        future_n_t_strength := t.Strength
                        if nn.Move == 4 || nn.Move == -1 {
                            future_n_t_strength += nn.Strength
                        }
                        for _, n_moving := range nn.MovingHere {
                            future_n_t_strength += n_moving.Strength
                        }
                        if future_n_t_strength <= 255 + g.StrengthBuffer {
                            success := g.moveSquareToTargetSimple(t, nn, true)
                            if success {
                                return true
                            }
                        }
                    }
                }
                g.makeMove(s, -1) // Undo the move, we didn't find a success
            }
        }
    }
    return false
}

type pairDS struct{
    dir int
    square *Square
}

func (g *Game) moveSquareToTargetSimple(s *Square, d *Square, throughFriendly bool) bool {

    // For large distances, we can probably get away with simple movement rules.
    dist_w := (s.X - d.X + g.Width) % g.Width
    dist_e := (d.X - s.X + g.Width) % g.Width
    dist_n := (s.X - d.X + g.Height) % g.Height
    dist_s := (d.X - s.X + g.Height) % g.Height

    if dist_w == 0 && dist_n == 0{
        return false
    }

    ew_swap := false
    ns_swap := false

    var ew_move *pairDS
    var ns_move *pairDS

    n_neighbor := s.Neighbors[0]
    e_neighbor := s.Neighbors[1]
    s_neighbor := s.Neighbors[2]
    w_neighbor := s.Neighbors[3]

    if dist_w < dist_e {
        if throughFriendly && w_neighbor.Owner != g.MyID {
            if e_neighbor.Owner == g.MyID {
                ew_move = &pairDS{1, e_neighbor}
                ew_swap = true
            }
        } else {
            ew_move = &pairDS{3, w_neighbor}
        }
    } else if dist_e < dist_w {
        if throughFriendly && e_neighbor.Owner != g.MyID {
            if w_neighbor.Owner == g.MyID {
                ew_move = &pairDS{3, w_neighbor}
                ew_swap = true
            }
        } else {
            ew_move = &pairDS{1, e_neighbor}
        }
    } else if dist_w == dist_e {
        if throughFriendly && (w_neighbor.Owner != g.MyID || e_neighbor.Owner != g.MyID) {
            if w_neighbor.Owner != g.MyID && e_neighbor.Owner != g.MyID {

            } else if w_neighbor.Owner == g.MyID && e_neighbor.Owner != g.MyID {
                ew_move = &pairDS{3, w_neighbor}
            } else {
                ew_move = &pairDS{1, e_neighbor}
            }
        } else {
            // Prefer the move with lower production
            if e_neighbor.Production < w_neighbor.Production {
                ew_move = &pairDS{1, e_neighbor}
            } else {
                ew_move = &pairDS{3, w_neighbor}
            }
        }
    }

    if dist_s < dist_n {
        if throughFriendly && s_neighbor.Owner != g.MyID {
            if n_neighbor.Owner == g.MyID {
                ns_move = &pairDS{0, n_neighbor}
                ns_swap = true
            }
        } else {
            ns_move = &pairDS{2, s_neighbor}
        }
    } else if dist_n < dist_s {
        if throughFriendly && n_neighbor.Owner != g.MyID {
            if s_neighbor.Owner == g.MyID {
                ns_move = &pairDS{2, s_neighbor}
                ns_swap = true
            }
        } else {
            ns_move = &pairDS{0, n_neighbor}
        }
    } else if dist_s == dist_n {
        if throughFriendly && (s_neighbor.Owner != g.MyID || n_neighbor.Owner != g.MyID) {
            if s_neighbor.Owner != g.MyID && n_neighbor.Owner != g.MyID {

            } else if s_neighbor.Owner == g.MyID && n_neighbor.Owner != g.MyID {
                ns_move = &pairDS{2, s_neighbor}
            } else {
                ns_move = &pairDS{0, n_neighbor}
            }
        } else {
            // Prefer the move with lower production
            if n_neighbor.Production < s_neighbor.Production {
                ns_move = &pairDS{0, n_neighbor}
            } else {
                ns_move = &pairDS{2, s_neighbor}
            }
        }
    }

    if ns_move == nil && ew_move == nil {
        return false
    }

    pathChoices := make([]int, 0, 2)
    if ns_move == nil {
        pathChoices = append(pathChoices, ew_move.dir)
    } else if ew_move == nil {
        pathChoices = append(pathChoices, ns_move.dir)
    } else if ns_swap == true && ew_swap == false {
        pathChoices = append(pathChoices, ew_move.dir)
        pathChoices = append(pathChoices, ns_move.dir)
    } else if ns_swap == false && ew_swap == true {
        pathChoices = append(pathChoices, ns_move.dir)
        pathChoices = append(pathChoices, ew_move.dir)
    } else {
        if ew_move.square.Production < ns_move.square.Production {
            pathChoices = append(pathChoices, ew_move.dir)
            pathChoices = append(pathChoices, ns_move.dir)
        } else {
            pathChoices = append(pathChoices, ns_move.dir)
            pathChoices = append(pathChoices, ew_move.dir)
        }
    }

    // Strength collision code goes here.

    // Try simple resolution
    for _, dir := range pathChoices {
        futureStrength := 0.0
        t := s.Neighbors[dir]
        if t.Owner == g.MyID && (t.Move == -1 || t.Move == 4) {
            futureStrength += t.Strength  // + t.Production
        }
        for _, sq := range t.MovingHere {
            futureStrength += sq.Strength
        }
        if futureStrength + s.Strength <= 255 + g.StrengthBuffer {
            g.makeMove(s, dir)
            return true
        }
    }

    for _, dir := range pathChoices {
        t := s.Neighbors[dir]
        // Can we move the cell we are moving to?
        if t.Owner == g.MyID && (t.Move == -1 || t.Move == 4) {     // Should we allow this to move STILL pieces?? Maybe test later
            futureStrength := s.Strength
            for _, sq := range t.MovingHere {
                futureStrength += sq.Strength
            }
            if futureStrength <= 255 + g.StrengthBuffer {
                // Ok, we can move the target square and be ok.
                g.makeMove(s, dir)  // Queue the move up, undo if it doesn't work
                nNeighbors := []*Square{t.Neighbors[0], t.Neighbors[1], t.Neighbors[2], t.Neighbors[3]}

                sort.Sort(ByProduction(nNeighbors))
                sort.Sort(ByBorderDistance(nNeighbors))

                for _, nn := range nNeighbors {
                    if nn.Owner == g.MyID && g.EnemyStrengthMap[2][nn.X][nn.Y] == 0 {
                        // Can we move into this square safely?
                        future_n_t_strength := t.Strength
                        if nn.Move == 4 || nn.Move == -1 {
                            future_n_t_strength += nn.Strength
                        }
                        for _, n_moving := range nn.MovingHere {
                            future_n_t_strength += n_moving.Strength
                        }
                        if future_n_t_strength <= 255 + g.StrengthBuffer {
                            success := g.moveSquareToTargetSimple(t, nn, true)
                            if success {
                                return true
                            }
                        }
                    }
                }
                g.makeMove(s, -1) // Undo the move, we didn't find a success
            }
        }
    }
    return false
}

func (g *Game) lastResortStrengthCheck() int {

    // Calculates the projected strength map and identifies squares that are violating it.
    // Ignore strength overloads due to production for now
    // Validate moves
    projectedStrengthMap := make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        projectedStrengthMap[x] = make([]float64, g.Height)
    }

    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            //We only care about our moves.
            if g.Squares[x][y].Owner == g.MyID {
                if g.Squares[x][y].Move == -1 || g.Squares[x][y].Move == 4 {
                    projectedStrengthMap[x][y] += g.Squares[x][y].Strength  // + square.production
                } else {
                    dx, dy := getOffset(g.Squares[x][y].Move)
                    projectedStrengthMap[(x + dx + g.Width) % g.Width][(y + dy + g.Height) % g.Height] += g.Squares[x][y].Strength
                }
            }
        }
    }

    violationCount := 0

    violationSquares := make([]*Square, 0, g.Width)

    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            if projectedStrengthMap[x][y] > 255 + g.StrengthBuffer {
                violationSquares = append(violationSquares, g.Squares[x][y])
                violationCount += 1
            }
        }
    }

    for _, sq := range violationSquares {
        if sq.Owner == g.MyID && (sq.Move == -1 || sq.Move == 4) {
            // Try to move this square to a neighbor
            possible_paths := make([]pairDS, 0, 4)
            for d := 0; d < 4; d++ {
                // Move to the lowest strength neighbor. This might cause a collision but we'll resolve it with multiple iterations
                n := sq.Neighbors[d]
                if n.Owner == g.MyID && g.EnemyStrengthMap[2][n.X][n.Y] == 0 {
                    possible_paths = append(possible_paths, pairDS{d, n})
                } else {
                    // Try attacking a bordering cell
                    if sq.Strength > (2 * n.Strength) && n.Production > 1 {
                        possible_paths = append(possible_paths, pairDS{d, n})
                    }
                }

            }

            // force a move.
            maxVal := -1.0
            movedir := -1
            var tVal float64
            for _, pp := range possible_paths {
                dx, dy := getOffset(pp.dir)
                nx := (pp.square.X + dx + g.Width) % g.Width
                ny := (pp.square.Y + dy + g.Height) % g.Height
                if pp.square.Owner == g.MyID {
                    tVal = g.DistanceFromCombat[nx][ny] * 1000 + (500 - projectedStrengthMap[nx][ny])
                } else {
                    tVal = g.DistanceFromCombat[pp.square.X][pp.square.Y] * 1000 + (500 - g.StrengthMap[nx][ny] * 2)
                }
                if tVal > maxVal {
                    maxVal = tVal
                    movedir = pp.dir
                }
            }
            g.makeMove(sq, movedir)
        } else {
            // We aren't the problem. One of the squares that's moving here is going to collide with us.
            optionsList := make([]*Square, 0, 4)
            for _, n := range sq.MovingHere {
                if n.Owner == g.MyID && ((projectedStrengthMap[n.X][n.Y] + n.Strength) <= (255 + g.StrengthBuffer)) {
                    optionsList = append(optionsList, n)
                }
            }
            sort.Sort(ByStrength(optionsList))
            i := 0
            if len(optionsList) > 0 {
                for ; projectedStrengthMap[sq.X][sq.Y] > 255 + g.StrengthBuffer; {
                    n := optionsList[i]
                    projectedStrengthMap[sq.X][sq.Y] -= n.Strength
                    projectedStrengthMap[n.X][n.Y] += n.Strength
                    g.makeMove(n, 4)
                }
            }
        }
    }
    return violationCount
}

func (g *Game) distanceBetween(sq1, sq2 *Square) float64 {
    dx := math.Abs(float64(sq1.X - sq2.X))
    dy := math.Abs(float64(sq1.Y - sq2.Y))
    if dx > float64(g.Width) / 2 {
        dx = float64(g.Width) - dx
    }
    if dy > float64(g.Height) / 2 {
        dy = float64(g.Height) - dy
    }
    return dx + dy
}

func (g *Game) eachSquareMoves() {
    // Each square decides on their own whether or not to move.
    // For now, let's just loop through the list of squares to determine who moves
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            square := (g.Squares[x][y])
            if square.Owner == g.MyID && square.Move == -1 {
                // Check distance from border
                if g.DistanceFromBorder[x][y] == 1.0 {
                    // We're at a border, check if we can attack a cell
                    for d, n := range square.Neighbors {
                        if n.Owner != g.MyID && square.Strength > n.Strength {
                            g.makeMove(square, d)
                            break
                        }
                    }
                }
            }
        }
    }
}

func (g *Game) makeMove(square *Square, d int) {
    g.MoveMap[square.X][square.Y] = float64(d)

    if d == -1 {
        // Reset the square move
        if square.Target != nil {
            delete(square.Target.MovingHere, square.Vertex)
            square.Target = nil
        }
        square.Move = -1
        return
    }
    if square.Move != -1 {
        if square.Target != nil {
            delete(square.Target.MovingHere, square.Vertex)
            square.Target = nil
        }
    }
    square.Move = d
    // log.Println(d)
    if d != 4 {
        square.Target = square.Neighbors[d]
        square.Target.MovingHere[square.Vertex] = square
    }
}

func (g *Game) floodFill(sources []*Square, maxDistance float64, friendly_only bool) [][]float64 {
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

    if len(sources) == 0 {
        return distanceMap
    }

    // Set all source squares to 0
    for _, square := range q {
        distanceMap[square.X][square.Y] = 0
    }

    for ; len(q) > 0 ; {
        c := q[0]
        q = q[1:]
        currentDistance := distanceMap[c.X][c.Y]
        for _, n := range c.Neighbors {
            if (distanceMap[n.X][n.Y] == -1 || distanceMap[n.X][n.Y] > (currentDistance + 1)) {
                if (friendly_only && n.Owner == g.MyID) || (!friendly_only) {
                    distanceMap[n.X][n.Y] = currentDistance + 1
                    if currentDistance < maxDistance - 1 {
                        q = append(q, n)
                    }
                }
            }
        }
    }
    return distanceMap
}

func (g *Game) floodFillToTarget(source *Square, destination *Square, friendly_only bool) [][]float64 {
    // We start the fill AT the destination so we can get # of squares from source to destination.
    q := make([]*Square, 1, 5)
    q[0] = destination
    distanceMap := make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        distanceMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            distanceMap[x][y] = -1
        }
    }
    distanceMap[destination.X][destination.Y] = 0
    for ; len(q) > 0 && distanceMap[source.X][source.Y] == -1; {
        c := q[0]
        q = q[1:]
        currentDistance := distanceMap[c.X][c.Y]
        for _, n := range c.Neighbors {
            if (distanceMap[n.X][n.Y] == -1 || distanceMap[n.X][n.Y] > (currentDistance + 1)) {
                if (friendly_only && n.Owner == g.MyID) || (!friendly_only) {
                    distanceMap[n.X][n.Y] = currentDistance + 1
                    q = append(q, n)
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
            newMatrix[x][y] = mat[(x - ox + x_len) % x_len][(y - oy + y_len) % y_len]
        }
    }
    return newMatrix
}

func roll_xy_onto(mat [][]float64, ox int, oy int, onto [][]float64) [][]float64 {
    for x := 0; x < len(mat); x++ {
        for y := 0; y < len(mat[x]); y++ {
            onto[x][y] = mat[(x - ox + len(mat)) % len(mat)][(y - oy + len(mat[x])) % len(mat[x])]
        }
    }
    return onto
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
    s.North = s.Game.Squares[s.X][(s.Y - 1 + s.Height) % s.Height]
    s.East = s.Game.Squares[(s.X + 1 + s.Width) % s.Width][s.Y]
    s.South = s.Game.Squares[s.X][(s.Y + 1 + s.Height) % s.Height]
    s.West = s.Game.Squares[(s.X - 1 + s.Width) % s.Width][s.Y]
    s.Neighbors = []*Square{s.North, s.East, s.South, s.West}  // doesn't include self.
    s.ResetStatus = true
    s.MovingHere = make(map[int]*Square)
    s.Target = nil
}

func (s *Square) update(owner float64, strength float64) {
    s.Owner = owner
    s.Strength = strength
    s.resetMove()
}

func (s *Square) resetMove() {
    s.Move = -1
    s.ResetStatus = true
    s.MovingHere = make(map[int]*Square)
    s.Target = nil
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
                outString = fmt.Sprintf("%s %d %d %d", outString, square.X, square.Y, translate_cardinal(square.Move))
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

    g.ProductionMap = make([][]float64, g.Width)
    for x := 0; x < g.Width; x++ {
        g.ProductionMap[x] = make([]float64, g.Height)
        for y := 0; y < g.Height; y++ {
            var prod_val int
            prod_val, splitString = int_str_array_pop(splitString)
            g.ProductionMap[x][y] = float64(prod_val)
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

func translate_cardinal(d int) int {
    // Cardinal index used by the framework is:
    // NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3, STILL = 4
    // Cardinal index used by the game is:
    // STILL = 0, NORTH = 1, EAST = 2, SOUTH = 3, WEST = 4
    return (d + 1) % 5
}

func getOffset(d int) (int, int) {
    switch d {
    case 0:
        return 0, -1
    case 1:
        return 1, 0
    case 2:
        return 0, 1
    case 3:
        return -1, 0
    case 4:
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

func Sum2d(array [][]float64) float64 {
    // Takes a 2d array and returns the sum of all values
    val := 0.0
    for x := 0; x < len(array); x++ {
        for y := 0; y < len(array[x]); y++ {
            val += float64(array[x][y])
        }
    }
    return val
}

func Multiply2d(arr1 [][]float64, arr2 [][]float64) [][]float64 {
    // Takes a 2d array and multiplies the elements together
    retArray := make([][]float64, len(arr1))
    for x := 0; x < len(arr1); x++ {
        retArray[x] = make([]float64, len(arr1[x]))
        for y := 0; y < len(arr1[x]); y++ {
            retArray[x][y] = arr1[x][y] * arr2[x][y]
        }
    }
    return retArray
}

func Add2d(arr1 [][]float64, arr2 [][]float64) [][]float64 {
    // Takes 2 2-d arrays and adds the elements together
    retArray := make([][]float64, len(arr1))
    for x := 0; x < len(arr1); x++ {
        retArray[x] = make([]float64, len(arr1[x]))
        for y := 0; y < len(arr1[x]); y++ {
            retArray[x][y] = arr1[x][y] + arr2[x][y]
        }
    }
    return retArray
}

func main() {
    f, _ := os.OpenFile("gologfile.log", os.O_RDWR | os.O_CREATE | os.O_APPEND, 0666)
    log.SetOutput(f)
    log.Println("newgame")
    game := NewGame()
    log.Println("GameInitDone")
    for {
        log.Println("getting frame")
        game.getFrame()
        log.Println("doing update")
        game.update()
        log.Println("getting moves")
        game.getMoves()
        log.Println("doing coll 1")
        collision_check := 998
        last_collision_check := 999
        for ; collision_check < last_collision_check ; {
            last_collision_check = collision_check
            collision_check = game.lastResortStrengthCheck()
        }
        log.Println("doing coll 2")
        collision_check = 998
        last_collision_check = 999
        for ; collision_check < last_collision_check ; {
            last_collision_check = collision_check
            collision_check = game.lastResortStrengthCheck()
        }

        log.Println("doing coll 3")

        collision_check = 998
        last_collision_check = 999
        for ; collision_check < last_collision_check ; {
            last_collision_check = collision_check
            collision_check = game.lastResortStrengthCheck()
        }
        log.Println("sending framw")
        game.sendFrame()
        log.Println("turn done")
    }
}



func (g *Game) getxy(vertex int) (int, int) {
    x := int(math.Floor(float64(vertex) / float64(g.Height)))
    y := vertex % g.Height
    return x, y
}

func getxy(vertex int) (int, int) {
    x := int(math.Floor(float64(vertex) / float64(Height)))
    y := vertex % Height
    return x, y
}

func spreadN(M [][]float64, n int) [][]float64 {
    retArray := make([][]float64, len(M))
    for x:= 0; x < len(M); x++ {
        retArray[x] = make([]float64, len(M[x]))
        for y := 0; y < len(M[x]); y++ {
            retArray[x][y] = M[x][y]
        }
    }

    for d := 1; d <= n; d++ {
        for i := 0; i <= d; i++ {
            x := i
            y := d - i

            retArray = roll_xy_onto(M, x, y, retArray)

            if x != 0 {
                retArray = roll_xy_onto(M, -x, y, retArray)
            }
            if y != 0 {
                retArray = roll_xy_onto(M, x, -y, retArray)
            }
            if x != 0 && y != 0 {
                retArray = roll_xy_onto(M, -x, -y, retArray)
            }

        }
    }
    return retArray
}




func (g *Game) createDijkstraMaps() {
    // Creates the dijkstra map(s) that will be utilized in this bot.

    // A 4-d array is created which contains all the information on the costs and routes for every cell to every other cell.
    // Ignores who owns the cell

    // Run Dijkstra on recovery cost for all squares to all squares.

    // edges, nodes := g.makeGraphRecovery()
    _, nodes := g.makeGraphRecovery()
    g.DijkstraRecoveryCosts = make([][][][]float64, g.Width)
    g.DijkstraRecoveryPaths = make([][][][]int, g.Width)
    for x := 0; x < g.Width; x++ {
        g.DijkstraRecoveryCosts[x] = make([][][]float64, g.Height)
        g.DijkstraRecoveryPaths[x] = make([][][]int, g.Height)
        for y := 0; y < g.Height; y++ {
            g.DijkstraRecoveryCosts[x][y] = make([][]float64, g.Width)
            g.DijkstraRecoveryPaths[x][y] = make([][]int, g.Width)
            for a := 0; a < g.Width; a++ {
                g.DijkstraRecoveryCosts[x][y][a] = make([]float64, g.Height)
                g.DijkstraRecoveryPaths[x][y][a] = make([]int, g.Height)
            }
        }
    }
    for startV := 0; startV < g.Height * g.Width; startV++ {
        vx, vy := g.getxy(startV)
        pathList := dijkstra(nodes, nodes[startV], nil)
        for _, path := range pathList{
            target := path.targetVertex
            tx, ty := g.getxy(target)
            g.DijkstraRecoveryCosts[vx][vy][tx][ty] = path.length
            g.DijkstraRecoveryPaths[vx][vy][tx][ty] = path.path
            // if len(path.path) > 1 {
            //     g.DijkstraRecoveryPaths[vx][vy][tx][ty] = path.path[1]
            // } else {
            //     g.DijkstraRecoveryPaths[vx][vy][tx][ty] = -9999
            // }
        }
    }
}

func (g *Game) singledijkstra() {
    // Creates the dijkstra map(s) that will be utilized in this bot.

    // A 4-d array is created which contains all the information on the costs and routes for every cell to every other cell.
    // Ignores who owns the cell

    // Run Dijkstra on recovery cost for all squares to all squares.

    // edges, nodes := g.makeGraphRecovery()
    _, nodes := g.makeGraphRecovery()
    g.DijkstraRecoveryCosts = make([][][][]float64, g.Width)
    g.DijkstraRecoveryPaths = make([][][][]int, g.Width)
    for x := 0; x < g.Width; x++ {
        g.DijkstraRecoveryCosts[x] = make([][][]float64, g.Height)
        g.DijkstraRecoveryPaths[x] = make([][][]int, g.Height)
        for y := 0; y < g.Height; y++ {
            g.DijkstraRecoveryCosts[x][y] = make([][]float64, g.Width)
            g.DijkstraRecoveryPaths[x][y] = make([][]int, g.Width)
            for a := 0; a < g.Width; a++ {
                g.DijkstraRecoveryCosts[x][y][a] = make([]float64, g.Height)
                g.DijkstraRecoveryPaths[x][y][a] = make([]int, g.Height)
            }
        }
    }
    for startV := 0; startV < 1; startV++ {
        vx, vy := g.getxy(startV)
        pathList := dijkstra(nodes, nodes[startV], nil)
        for _, path := range pathList{
            target := path.targetVertex
            tx, ty := g.getxy(target)
            g.DijkstraRecoveryCosts[vx][vy][tx][ty] = path.length
            g.DijkstraRecoveryPaths[vx][vy][tx][ty] = path.path
            // if len(path.path) > 1 {
            //     g.DijkstraRecoveryPaths[vx][vy][tx][ty] = path.path[1]
            // } else {
            //     g.DijkstraRecoveryPaths[vx][vy][tx][ty] = -9999
            // }
        }
    }
}


type Edge struct {
    V1, V2 int
    Dist float64
}

type Node struct {
    V int
    TDist float64  // tentative distance
    Prev *Node
    Done bool  // True when Tdist and Prev represent the shortest path
    Neighbors []Neighbor  // Edges from this vertex
    Rx int  // heap.Remove index
}

type Neighbor struct {
    Node *Node // Node corresponding to a vertex
    Dist float64  // Distance to this node, from whatever node references this
}

func (g *Game) makeGraphRecovery() ([]Edge, []*Node) {
    // Creates a graph from the Squares object to be used for Dijkstra's
    // To hopefully reduce # of calls, this actually builds the graph the opposite way. Builds the graph from neighbors instead of TO neighbors.
    graph := make([]Edge, 0, int(g.Width * g.Height * 4))
    nodes := make([]*Node, int(g.Width * g.Height))
    for x := 0; x < g.Width; x++ {
        for y := 0; y < g.Height; y++ {
            // Builds edges based on how much it costs to get to THIS cell FROM neighbors.
            graph = append(graph, Edge{V1: g.Squares[x][y].North.Vertex, V2: g.Squares[x][y].Vertex, Dist: g.StrengthMap1[x][y] / g.ProductionMap01[x][y]})
            graph = append(graph, Edge{V1: g.Squares[x][y].South.Vertex, V2: g.Squares[x][y].Vertex, Dist: g.StrengthMap1[x][y] / g.ProductionMap01[x][y]})
            graph = append(graph, Edge{V1: g.Squares[x][y].West.Vertex, V2: g.Squares[x][y].Vertex, Dist: g.StrengthMap1[x][y] / g.ProductionMap01[x][y]})
            graph = append(graph, Edge{V1: g.Squares[x][y].East.Vertex, V2: g.Squares[x][y].Vertex, Dist: g.StrengthMap1[x][y] / g.ProductionMap01[x][y]})

            // Add the node
            nodes[g.Squares[x][y].Vertex] = &Node{V: g.Squares[x][y].Vertex}
        }
    }

    for _, e := range graph {
        n1 := nodes[e.V1]
        n2 := nodes[e.V2]
        n1.Neighbors = append(n1.Neighbors, Neighbor{n2, e.Dist})
    }
    return graph, nodes
}

type path struct {
    // path []int
    path int
    length float64
    targetVertex int
}

func dijkstra(allNodes []*Node, startNode, endNode *Node) (pathList []path) {
    // 1. Assign to every node a tentative distance value: set it to zero for our initial node and to infinity for all other nodes.
    // 2. Set the initial node as current. Mark all other nodes unvisited. Create a set of all the unvisited nodes called the unvisited set.
    for _, nd := range allNodes {
        nd.TDist = math.Inf(1)
        nd.Done = false
        nd.Prev = nil
        nd.Rx = -1
    }

    current := startNode
    current.TDist = 0
    var unvisited ndList
    for {
        // 3. For the current node, consider all of its unvisited neighbors and calculate their tentative distances. Compare the newly calculated tentative distance to the current assigned value and assign the smaller one. For example, if the current node A is marked with a distance of 6, and the edge connecting it with a neighbor B has length 2, then the distance to B (through A) will be 6 + 2 = 8. If B was previously marked with a distance greater than 8 then change it to 8. Otherwise, keep the current value.
        for _, nb := range current.Neighbors {
            if nd := nb.Node; !nd.Done {
                if d := current.TDist + nb.Dist; d < nd.TDist {
                    nd.TDist = d
                    nd.Prev = current
                    if nd.Rx < 0 {
                        heap.Push(&unvisited, nd)
                    } else {
                        heap.Fix(&unvisited, nd.Rx)
                    }
                }
            }
        }
        // 4. When we are done considering all of the neighbors of the current node, mark the current node as visited and remove it from the unvisited set. A visited node will never be checked again.
        current.Done = true
        if endNode == nil || current == endNode {
            // Record path and distance for return value
            distance := current.TDist
            // Recover path by tracing prev links
            var p []int
            target := current.V
            for ; current != nil; current = current.Prev {
                p = append(p, current.V)
            }
            // Reverse the list
            // for i := (len(p) + 1) / 2; i > 0; i-- {
            //     p[i-1], p[len(p)-i] = p[len(p)-1], p[i-1]
            // }
            // pathList = append(pathList, path{p, distance, target})
            var prior int
            if len(p) > 1 {
                prior = p[len(p) - 1]
            } else {
                prior = -9999
            }

            pathList = append(pathList, path{prior, distance, target})
            // 5. If the destination node has been marked visited (when planning a route between two specific nodes) or if the smallest tentative distance among the nodes in the unvisited set is infinity (when planning a complete traversal; occurs when there is no connection between the initial node and remaining unvisited nodes), then stop. The algorithm has finished.
            if endNode != nil {
                return
            }
        }
        if len(unvisited) == 0 {
            break  // No more reachable nodes
        }
        // 6. Otherwise, select the unvisited node that is marked with the smallest tentative distance, set it as the new "current node", and go back to step 3.
        current = heap.Pop(&unvisited).(*Node)
    }
    return
}

// ndList implements a container/heap
type ndList []*Node

func (n ndList) Len() int {
    return len(n)
}

func (n ndList) Less(i, j int) bool {
    return n[i].TDist < n[j].TDist
}

func (n ndList) Swap(i, j int) {
    n[i], n[j] = n[j], n[i]
    n[i].Rx = i
    n[j].Rx = j
}

func (n *ndList) Push(x interface{}) {
    nd := x.(*Node)
    nd.Rx = len(*n)
    *n = append(*n, nd)
}

func (n *ndList) Pop() interface{} {
    s := *n
    last := len(s) - 1
    r := s[last]
    *n = s[:last]
    r.Rx = -1
    return r
}


type ByStrength []*Square

func (s ByStrength) Len() int {
    return len(s)
}

func (s ByStrength) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByStrength) Less(i, j int) bool {
    return s[i].Strength < s[j].Strength
}

type ByProduction []*Square

func (s ByProduction) Len() int {
    return len(s)
}

func (s ByProduction) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByProduction) Less(i, j int) bool {
    return s[i].Production < s[j].Production
}

type ByBorderDistance []*Square

func (s ByBorderDistance) Len() int {
    return len(s)
}

func (s ByBorderDistance) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByBorderDistance) Less(i, j int) bool {
    return s[i].DistanceFromBorder < s[j].DistanceFromBorder
}

type ByEnemyStrength2 []*Square

func (s ByEnemyStrength2) Len() int {
    return len(s)
}

func (s ByEnemyStrength2) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByEnemyStrength2) Less(i, j int) bool {
    return s[i].EnemyStrengthMap2 < s[j].EnemyStrengthMap2
}

type ByEnemyStrength4 []*Square

func (s ByEnemyStrength4) Len() int {
    return len(s)
}

func (s ByEnemyStrength4) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

func (s ByEnemyStrength4) Less(i, j int) bool {
    return s[i].EnemyStrengthMap4 < s[j].EnemyStrengthMap4
}
