#!/usr/bin/env python3
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

import copy
import os
import random
import sys
import math
import sqlite3
import argparse
import datetime
import shutil
import skills
from skills import trueskill
from subprocess import Popen, PIPE, call
#from keyboard_detection import keyboard_detection


halite_command = "halite"
replay_dir = "replays"
db_filename = "db.sqlite3"
browser_binary = "chrome"

def max_match_rounds(width, height):
    return math.sqrt(width * height) * 10.0

def update_skills(players, ranks):
    """ Update player skills based on ranks from a match """
    teams = [skills.Team({player.name: skills.GaussianRating(player.mu, player.sigma)}) for player in players]
    match = skills.Match(teams, ranks)
    calc = trueskill.FactorGraphTrueSkillCalculator()
    game_info = trueskill.TrueSkillGameInfo()
    updated = calc.new_ratings(match, game_info)
    print ("Updating ranks")
    for team in updated:
        player_name, skill_data = next(iter(team.items()))    #in Halite, teams will always be a team of one player
        player = next(player for player in players if player.name == str(player_name))   #this suggests that players should be a dictionary instead of a list
        player.mu = skill_data.mean
        player.sigma = skill_data.stdev
        player.update_skill()
        print("skill = %4f  mu = %3f  sigma = %3f  name = %s" % (player.skill, player.mu, player.sigma, str(player_name)))


class Match:
    def __init__(self, players, width, height, seed, time_limit, keep_replays):
        self.map_seed = seed
        self.width = width
        self.height = height
        self.players = players
        self.paths = [player.path for player in players]
        self.finished = False
        self.results = [0 for _ in players]
        self.return_code = None
        self.results_string = ""
        self.replay_file = ""
        self.total_time_limit = time_limit
        self.timeouts = []
        self.num_players = len(players)
        self.keep_replay = keep_replays

    def __repr__(self):
        title1 = "Match between " + ", ".join([p.name for p in self.players]) + "\n"
        title2 = "Binaries are " + ", ".join(self.paths) + "\n"
        dims = "dimensions = " + str(self.width) + ", " + str(self.height) + "\n"
        results = "\n".join([str(i) + " " + j for i, j in zip(self.results, [p.name for p in self.players])]) + "\n"
        replay = self.replay_file + "\n" #\n"
        return title1 + title2 + dims + results + replay

    def get_command(self, halite_binary):
        dims = "-d " + str(self.width) + " " + str(self.height)
        quiet = "-q"
        seed = "-s " + str(self.map_seed)
        result = [halite_binary, dims, quiet, seed]
        return result + self.paths

    def run_match(self, halite_binary):
        command = self.get_command(halite_binary)
        p = Popen(command, stdin=None, stdout=PIPE, stderr=None)
        results, _ = p.communicate(None, self.total_time_limit)
        self.results_string = results.decode('ascii')
        self.return_code = p.returncode
        self.parse_results_string()
        update_skills(self.players, copy.deepcopy(self.results))
        if self.keep_replay:
            print("Keeping replay\n")
            if not os.path.exists(replay_dir):
                os.makedirs(replay_dir)
            shutil.move(self.replay_file, replay_dir)
        else: 
            print("Deleting replay\n")
            os.remove(self.replay_file)

    def parse_results_string(self):
        lines = self.results_string.split("\n")
        if len(lines) < (2 + (2 * self.num_players)):
            raise ValueError("Not enough lines in match output")
        else:
            for count, line in enumerate(lines):
                if count == self.num_players: # replay file and seed
                    self.replay_file = line.split(" ")[0]
                elif count == (self.num_players * 2) + 1: # timeouts
                    self.timeouts = (line.split(" "))
                elif count < self.num_players: # names
                    pass
                elif count < (self.num_players * 2) + 1:
                    player_index, rank = map(int, line.split())
                    player_index -= 1   #zero-based indexing
                    self.results[player_index] = rank


class Manager:
    def __init__(self, halite_binary, players=None, rounds=-1):
        self.halite_binary = halite_binary
        self.players = players
        self.players_min = 2
        self.rounds = rounds
        self.round_count = 0
        self.keep_replays = True
        self.priority_sigma = True
        self.exclude_inactive = False
        self.db = Database()

    def run_round(self, contestants, width, height, seed):
        m = Match(contestants, width, height, seed, 2 * len(contestants) * max_match_rounds(width, height), self.keep_replays)
        print(m)
        m.run_match(self.halite_binary)
        print(m)
        self.save_players(contestants)
        self.db.update_player_ranks()
        self.db.add_match(m)
        self.show_ranks()

    def save_players(self, players):
        for player in players:
            print("Saving player %s with %f skill" % (player.name, player.skill))
            self.db.save_player(player)

    def pick_contestants(self, num):
        pool = list(self.players)   #this makes a copy
        contestants = list()
        if self.priority_sigma:
            high_sigma_index = max((player.sigma, i) for i, player in enumerate(self.players))[1]
            high_sigma_contestant = self.players[high_sigma_index]
            contestants.append(high_sigma_contestant)
            pool.remove(high_sigma_contestant)
            num -= 1
        random.shuffle(pool)
        contestants.extend(pool[:num])
        random.shuffle(contestants)
        return contestants


    def run_rounds(self):
        while ((self.rounds < 0) or (self.round_count < self.rounds)):
            num_contestants = random.choice([2] * 5 + [3] * 4 + [4] * 3 + [5] * 2 + [6])
            contestants = self.pick_contestants(num_contestants)
            size_w = random.choice([20, 25, 25] + [30] * 3 + [35] * 4 + [40] * 3 + [45, 45, 50])
            size_h = size_w
            seed = random.randint(10000, 2073741824)
            print ("\n------------------- running new match... -------------------\n")
            self.run_round(contestants, size_w, size_h, seed)
            self.round_count += 1

    def add_player(self, name, path):
        p = self.db.get_player((name,))
        if len(p) == 0:
            self.db.add_player(name, path)
        else:
            print ("Bot name %s already used, no bot added" %(name))


    def show_ranks(self, tsv=False):
        print()
        if tsv:
            print ("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("name", "last_seen", "rank", "skill", "mu", "sigma", "ngames", "active"))
        else:
            print ("%s\t\t%s\t\t%s\t%s\t\t%s\t\t%s\t\t%s\t%s" % ("name", "last_seen", "rank", "skill", "mu", "sigma", "ngames", "active"))
        sql = "select * from players where active > 0 order by skill desc" if self.exclude_inactive else "select * from players order by skill desc"
        for p in self.db.retrieve(sql):
            print(str(parse_player_record(p)))

            
class Database:
    def __init__(self, filename=db_filename):
        self.db = sqlite3.connect(filename)
        self.recreate()

    def __del__(self):
        try:
            self.db.close()
        except: pass

    def now(self):
        return datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    def recreate(self):
        cursor = self.db.cursor()
        try:
            cursor.execute("create table games(id integer primary key, game_id integer, name text, finish integer, field_size integer, map_size integer, map_seed integer, timestamp date, replay_file text)")
            cursor.execute("create table players(id integer primary key, name text unique, path text, lastseen date, rank integer default 1000, skill real default 0.0, mu real default 25.0, sigma real default 8.33,ngames integer default 0, active integer default 1)")
            self.db.commit()
        except:
            pass

    def update_deferred( self, sql, tup=() ):
        cursor = self.db.cursor()        
        cursor.execute(sql,tup)
        
    def update( self, sql, tup=() ):
        self.update_deferred(sql,tup)
        self.db.commit()

    def update_many(self, sql, iterable):
        cursor = self.db.cursor()
        cursor.executemany(sql, iterable)
        self.db.commit()
        
    def retrieve( self, sql, tup=() ):
        cursor = self.db.cursor()        
        cursor.execute(sql,tup)
        return cursor.fetchall()

    def add_match( self, match ):
        sql = 'SELECT max(game_id) FROM games'
        game_id = self.retrieve(sql)[0][0]
        game_id = int(game_id) + 1 if game_id else 1
        self.update_many("INSERT INTO games (game_id, name, finish, field_size, map_size, map_seed, timestamp, replay_file) VALUES (?,?,?,?,?,?,?,?)", [(game_id, player.name, rank, match.num_players, match.width, match.map_seed, self.now(), match.replay_file) for player, rank in zip(match.players, match.results)])

    def add_player(self, name, path):
        self.update("insert into players values(?,?,?,?,?,?,?,?,?,?)", (None, name, path, self.now(), 1000, 0.0, 25.0, 25.0/3.0, 0, True))

    def delete_player(self, name):
        self.update("delete from players where name=?", [name])

    def get_player( self, names ):
        sql = 'select * from players where name=? '  + ' '.join('or name=?' for _ in names[1:])
        return self.retrieve(sql, names )
        
    def save_player(self, player):
        self.update_player_skill(player.name, player.skill, player.mu, player.sigma)

    def update_player_skill(self, name, skill, mu, sigma ):
        self.update("update players set ngames=ngames+1,lastseen=?,skill=?,mu=?,sigma=? where name=?", (self.now(), skill, mu, sigma, name))
	
    def update_player_rank( self, name, rank ):
        self.update("update players set rank=? where name=?", (rank, name))

    def update_player_ranks(self):
        for i, p in enumerate(self.retrieve("select name from players order by skill desc",())):
            self.update_player_rank( p[0], i+1 )
        
    def activate_player(self, name):
        self.update("update players set active=? where name=?", (1, name))

    def deactivate_player(self, name):
        self.update("update players set active=? where name=?", (0, name))


class Player:
    def __init__(self, name, path, last_seen = "", rank = 1000, skill = 0.0, mu = 25.0, sigma = (25.0 / 3.0), ngames = 0, active = 1):
        self.name = name
        self.path = path
        self.last_seen = last_seen
        self.rank = rank
        self.skill = skill
        self.mu = mu
        self.sigma = sigma
        self.ngames = ngames
        self.active = active

    def __repr__(self):
        return "%s\t%s\t%d\t%3f\t%3f\t%3f\t%d\t%d" % (self.name, self.last_seen, self.rank, self.skill, self.mu, self.sigma, self.ngames, self.active)

    def update_skill(self):
        self.skill = self.mu - (self.sigma * 3)

def parse_player_record (player):
    (player_id, name, path, last_seen, rank, skill, mu, sigma, ngames, active) = player
    return Player(name, path, last_seen, rank, skill, mu, sigma, ngames, active)
    

class Commandline:
    def __init__(self):
        self.manager = Manager(halite_command)
        self.cmds = None
        self.parser = argparse.ArgumentParser()
        self.no_args = False
        self.parser.add_argument("-A", "--addBot", dest="addBot",
                                 action = "store", default = "",
                                 help = "Add a new bot with a name")

        self.parser.add_argument("-D", "--deleteBot", dest="deleteBot",
                                 action = "store", default = "",
                                 help = "Delete the named bot")

        self.parser.add_argument("-a", "--activateBot", dest="activateBot",
                                 action = "store", default = "",
                                 help = "Activate the named bot")

        self.parser.add_argument("-d", "--deactivateBot", dest="deactivateBot",
                                 action = "store", default = "",
                                 help = "Deactivate the named bot")

        self.parser.add_argument("-p", "--botPath", dest="botPath",
                                 action = "store", default = "",
                                 help = "Specify the path for a new bot")

        self.parser.add_argument("-r", "--showRanks", dest="showRanks",
                                 action = "store_true", default = False,
                                 help = "Show a list of all bots, ordered by skill")

        self.parser.add_argument("-t", "--showRanksTsv", dest="showRanksTsv",
                                 action = "store_true", default = False,
                                 help = "Show a list of all bots ordered by skill, with headings in TSV format like the rest of the data")

        self.parser.add_argument("-m", "--match", dest="match",
                                 action = "store_true", default = False,
                                 help = "Run a single match")

        self.parser.add_argument("-f", "--forever", dest="forever",
                                 action = "store_true", default = False,
                                 help = "Run games forever (or until interrupted)")

        self.parser.add_argument("-v", "--view", dest="view",
                                 action = "store", default = "",
                                 help = "View a replay in the web browser")

        self.parser.add_argument("-n", "--no-replays", dest="deleteReplays",
                                 action = "store_true", default = False,
                                 help = "Do not store replays")

        self.parser.add_argument("-e", "--equal-priority", dest="equalPriority",
                                 action = "store_true", default = False,
                                 help = "Equal priority for all active bots (otherwise highest sigma will always be selected)")

        self.parser.add_argument("-E", "--exclude-inactive", dest="excludeInactive",
                                 action = "store_true", default = False,
                                 help = "Exclude inactive bots from ranking table")

    def parse(self, args):
        self.no_args = not args
        self.cmds = self.parser.parse_args(args)

    def add_bot(self, bot, path):
        self.manager.add_player(bot, path)

    def delete_bot(self, bot):
        self.manager.db.delete_player(bot)

    def valid_botfile(self, path):
        return True

    def run_matches(self, rounds):
        player_records = self.manager.db.retrieve("select * from players where active > 0")
        players = [parse_player_record(player) for player in player_records]
        if len(players) < 2:
            print("Not enough players for a game. Need at least " + str(self.manager.players_min) + ", only have " + str(len(players)))
            print("use the -h flag to get help")
        else:
            self.manager.players = players
            self.manager.rounds = rounds
            self.manager.run_rounds()

    def act(self):
        if self.cmds.deleteReplays:
            print("keep_replays = False")
            self.manager.keep_replays = False
            
        if self.cmds.equalPriority:
            print("priority_sigma = False")
            self.manager.priority_sigma = False
            
        if self.cmds.excludeInactive:
            print("exclude_inactive = True")
            self.manager.exclude_inactive = True

        if self.cmds.addBot:
            print("Adding new bot...")
            if self.cmds.botPath == "":
                print ("You must specify the path for the new bot")
            elif self.valid_botfile(self.cmds.botPath):
                self.add_bot(self.cmds.addBot, self.cmds.botPath)
        
        elif self.cmds.deleteBot:
            print("Deleting bot...")
            self.delete_bot(self.cmds.deleteBot)
        
        elif self.cmds.activateBot:
            print("Activating bot %s" %(self.cmds.activateBot))
            self.manager.db.activate_player(self.cmds.activateBot)
        
        elif self.cmds.deactivateBot:
            print("Deactivating bot %s" %(self.cmds.deactivateBot))
            self.manager.db.deactivate_player(self.cmds.deactivateBot)
        
        elif self.cmds.view:
            print("Viewing replay %s" %(self.cmds.view))
            view_replay(self.cmds.view)
        
        elif self.cmds.showRanks:
            self.manager.show_ranks(tsv=False)
        
        elif self.cmds.showRanksTsv:
            self.manager.show_ranks(tsv=True)
        
        elif self.cmds.match:
            print ("Running a single match.")
            self.run_matches(1)
        
        elif self.cmds.forever:
            print ("Running matches until interrupted. Press any key to exit safely at the end of the current match.")
            self.run_matches(-1)
        
        elif self.no_args:
            self.parser.print_help()


def view_replay(filename):
    output_filename = filename.replace(".hlt", ".htm")
    if not os.path.exists(output_filename):
        with open(filename, 'r') as f:
            replay_data = f.read()
        with open("replays/Visualizer.htm") as f:
            html = f.read()
        html = html.replace("FILENAME", filename)
        html = html.replace("REPLAY_DATA", replay_data)
        with open(output_filename, 'w') as f:
            f.write(html)
    call ([browser_binary, output_filename])

    

cmdline = Commandline()
cmdline.parse(sys.argv[1:])
cmdline.act()