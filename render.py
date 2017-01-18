#!/usr/bin/env python

"""
Spartan command-line halite.io replay viewer.

Pass a replay as the first argument and watch in wonder as your .hlt
file is rendered in glorious octocolour.

Demands a lot of horizontal character space ((7 * map's X-dimension )+ 2).

Made with help from:
    @DanielVF halitenotebook and
    http://gnosis.cx/publish/programming/charming_python_6.html
"""


import sys
import json
import gzip
import curses
import traceback
import numpy as np


HELP_STRING = '. '.join([
    'Turn {}/{}',
    '←/→ advances 1 turn',
    '↑/↓ advances 10 turns',
    '[wasd] positions camera',
    'q quits.'
])


def main(stdscr):
    current_frame = 0
    rollx, rolly = 0, 0

    try:
        replay = Replay(sys.argv[1])
    except IndexError:
        raise IndexError(
            "Missing argument. Run with ./render.py some_replay_file.hlt."
        )

    board = replay.map_at(current_frame, 0, 0)

    dim_x, dim_y = board['height'] + 4, board['width'] * 7 + 2

    # This doesn't work properly for some reason.
    # max_y, max_x = stdscr.getmaxyx()

    # if max_x < dim_x or max_y < dim_y:
    #     raise TerminalSizeError(
    #         ("Your terminal is too small :(. Resize to at least {}x{}")
    #         .format(dim_x, dim_y)
    #     )

    global screen

    try:
        screen = stdscr.subwin(dim_x, dim_y, 0, 0)
    except curses.error:
        raise TerminalSizeError(
            ("Your terminal seems too small. Resize to at least {}x{}")
            .format(dim_x, dim_y)
        )

    screen.box()
    keypress = ''
    while keypress != ord('q'):
        board = replay.map_at(current_frame, rollx, rolly)
        stacked = stack_map(board)
        fractions = np.apply_along_axis(format_as_fraction, 2, stacked).T

        for x in range(fractions.shape[0]):
            for y in range(fractions.shape[1]):
                if board['owner'][y, x] == 0 and board['strength'][y, x] == 0:
                    color = curses.color_pair(8)
                else:
                    color = curses.color_pair(board['owner'][y, x] + 1)
                stdscr.addstr(
                    y + 1, x * 7 + 1, fractions[x, y], color
                )

        stdscr.addstr(y + 2, 2, '')  # Better way to position cursor?
        for i, name in enumerate(replay.player_names):
            stdscr.addstr(name, curses.color_pair(i + 2))
            stdscr.addstr(' ')

        stdscr.addstr(
            y + 3, 2, HELP_STRING.format(current_frame, board['num_frames'])
        )

        # There must be a better way to do this
        keypress = stdscr.getch()
        if keypress == curses.KEY_LEFT:
            current_frame = max(0, current_frame - 1)
        elif keypress == curses.KEY_RIGHT:
            current_frame = min(board['num_frames'] - 1, current_frame + 1)
        elif keypress == curses.KEY_UP:
            current_frame = max(0, current_frame - 10)
        elif keypress == curses.KEY_DOWN:
            current_frame = min(board['num_frames'] - 1, current_frame + 10)
        elif keypress == ord('w'):
            rolly += 1
        elif keypress == ord('a'):
            rollx += 1
        elif keypress == ord('s'):
            rolly -= 1
        elif keypress == ord('d'):
            rollx -= 1


class Replay(object):
    """Handle ETL of replay files and expose map_at to get
    information about the game for a specific frame.
    """

    def __init__(self, filename=None):
        if ".gz" in filename:
            with gzip.open(filename, 'rb') as f:
                data = json.load(f)
        else:
            with open(filename) as f:
                data = json.load(f)

        self.data = data
        self.width = data["width"]
        self.height = data["height"]
        self.num_players = data["num_players"]
        self.num_frames = data["num_frames"]
        self.player_names = ['Shummie', 'DexGroves']  # data["player_names"]

    def map_at(self, turn, rollx, rolly):
        """Return the map at a given turn. rollx and rolly can be
        used to offset the arrays.
        """
        frame = np.array(self.data['frames'][turn])
        production = np.array(self.data['productions'])
        production = np.roll(np.roll(production, rollx, 1), rolly, 0)
        strength = frame[:, :, 1]
        strength = np.roll(np.roll(strength, rollx, 1), rolly, 0)
        owner = frame[:, :, 0]
        owner = np.roll(np.roll(owner, rollx, 1), rolly, 0)
        gm = {
            'production': production,
            'strength': strength,
            'owner': owner,
            'width': self.width,
            'height': self.height,
            'num_frames': self.num_frames - 1
        }
        return gm


def stack_map(board):
    """Stack production, strength and owner to make fractions later."""
    strength = board["strength"]
    owner = board["owner"]
    production = board["production"]

    return np.stack([production, strength, owner], axis=2).astype(int)


def format_as_fraction(element):
    """Format a 3D map representation from stack_map into a 2D
    array of strings formatted as str/prod.
    """
    numerator = justify_int(element[1], 3, 'right')
    denominator = justify_int(element[0], 2, 'left')
    str_ = numerator + '/' + denominator
    return str_ + ' '


def justify_int(element, to, how='left'):
    """lpad or rpad an integer."""
    s = str(element)
    if how == 'left':
        return s + ' ' * (to - len(s))
    if how == 'right':
        return ' ' * (to - len(s)) + s


def setup_colors():
    """Setup the colors for each player. Entry 8 is reserved for
    zero-strength unowned squares.
    """
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_RED)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_BLUE)
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLACK)


class TerminalSizeError(Exception):
    """Your terminal is too small!"""
    pass


if __name__ == '__main__':
    try:
        # Initialize curses
        stdscr = curses.initscr()
        # Turn off echoing of keys, and enter cbreak mode,
        # where no buffering is performed on keyboard input
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        setup_colors()

        # In keypad mode, escape sequences for special keys
        # (like the cursor keys) will be interpreted and
        # a special value like curses.KEY_LEFT will be returned
        stdscr.keypad(1)
        main(stdscr)                    # Enter the main loop

        # Set everything back to normal
        stdscr.keypad(0)
        curses.echo()
        curses.nocbreak()
        curses.endwin()                 # Terminate curses
    except:
        # In event of error, restore terminal to sane state.
        stdscr.keypad(0)
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        traceback.print_exc()           # Print the exception