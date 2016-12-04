#!/usr/bin/env python3
# Stolen from https://github.com/rossmacarthur/halite-match-manager

import click
import os
import pickle
import random
import shutil
import subprocess
import statistics
import trueskill


HALITEBIN = '.\halite'
REPLAYDIR = 'replays/'
if not os.path.exists(REPLAYDIR):
    os.makedirs(REPLAYDIR)


def external(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


class Database:
    def __init__(self, filename):
        self.filename = filename
        if os.path.isfile(filename):
            self.db = pickle.load(open(self.filename, 'rb'))
        else:
            self.db = {}

    def __str__(self):
        if self.db:
            rows = []
            for name in self.db:
                rows.append((name,
                             '{:.2f}'.format(self.db[name]['rating'].mu),
                             '{:.2f}'.format(self.db[name]['rating'].sigma),
                             self.db[name]['games'],
                             self.db[name]['command']))
            rows = sorted(rows, key=lambda x: x[1], reverse=True)
            max_name = max([4]+[len(x[0]) for x in rows])
            template = '{{:>4}}  {{:<{}}}  {{:>7}}  {{:>5}} ' \
                       ' {{:>5}}  {{}}\n'.format(max_name)
            s = template.format('Rank', 'Name', 'Rating',
                                'Sigma', 'Games', 'Command')
            for i in range(len(rows)):
                s += template.format(i+1, *rows[i])
            return 79*'=' + '\n' + s + 79*'='
        else:
            return 'Database empty'

    def save(self):
        with open(self.filename, 'wb') as output_file:
            pickle.dump(self.db, output_file, -1)

    def names(self):
        return list(self.db)

    def add(self, name, command, rating=trueskill.Rating(), games=0):
        self.db[name] = {'command': command,
                         'rating': rating,
                         'games': games}

    def rm(self, name):
        del self.db[name]

    def get_command(self, name):
        return self.db[name]['command']

    def set_rating(self, name, rating):
        self.db[name]['rating'] = rating
        self.db[name]['games'] += 1

    def get_rating(self, name):
        return self.db[name]['rating']

    def reset_rating(self, name):
        self.db[name]['rating'] = trueskill.Rating()
        self.db[name]['games'] = 0


def pick_contestants(db, prioritize_new=True):
    number = random.randint(2, min(6, len(db.names())))
    contestants = []

    pool = list(db.names())
    non_high = number
    sigmadev = statistics.stdev(db.get_rating(n).sigma for n in db.names())
    sigmamean = statistics.mean(db.get_rating(n).sigma for n in db.names())
    for n in db.names():
        if db.get_rating(n).sigma > sigmamean + 2*sigmadev:
            contestants.append(n)
            pool.remove(n)
            non_high -= 1
    random.shuffle(pool)
    contestants.extend(pool[:non_high])

    return contestants


def match(db, width, height, contestants):
    number = len(contestants)

    # Print out some stuff
    click.echo(' MATCH: {} x {}, {}'.format(width, height,
                                            ' vs '.join(contestants)))

    # Run halite game
    cmd = '{} -q -d "{} {}" '.format(HALITEBIN, width, height)
    cmd += ' '.join('"{}"'.format(db.get_command(c)) for c in contestants)
    _, stdout, _ = external(cmd)

    # Parse output
    lines = stdout.decode('utf-8').strip().split('\n')[number:]
    replay_file = lines[0].split(' ')[0]
    ranks = [int(x.split()[1])-1 for x in lines[1:1+number]]
    players = [(db.get_rating(c),) for c in contestants]

    # Calculate and assign new rating
    ts = trueskill.TrueSkill(draw_probability=0)
    ratings = ts.rate(players, ranks)
    for i in range(number):
        db.set_rating(contestants[i], ratings[i][0])

    # Move replay file
    #shutil.move(replay_file, os.path.join(REPLAYDIR, replay_file))

    # Print out some stuff
    d = dict(zip(contestants, ranks))
    contestants.sort(key=d.get)
    click.echo('\n'.join(['   #{} {:<}'.format(i+1, contestants[i])
                          for i in range(number)]))
    click.echo(' Replay file: {}'.format(os.path.join(REPLAYDIR, replay_file)))
    click.echo()
    click.echo()


def random_match(db):
    dimension = random.choice(range(20, 51, 5))
    contestants = pick_contestants(db)
    match(db, dimension, dimension, contestants)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
def cli(ctx):
    """
    Utility to run batch matches between Halite bots. Bots are rated using the
    TrueSkill rating system.
    """
    ctx.obj = Database('manager.db')


@cli.command()
@click.argument('name')
@click.argument('command')
@click.pass_context
def add(ctx, name, command):
    """
    Add bot to the manager.

    \b
    NAME is a unique name for the bot e.g. MyBot.
    COMMAND is the command to run the bot e.g. python3 bots/MyBot.py
    """
    db = ctx.obj

    if name not in db.names():
        db.add(name, command)
        db.save()
        click.echo('New bot added!')
    else:
        click.echo('Bot with that name already exists')


@cli.command()
@click.argument('name', nargs=-1)
@click.pass_context
def rm(ctx, name):
    """
    Remove bot(s) from manager.

    NAME is the unique name of the bot to remove.
    """
    db = ctx.obj

    for n in name:
        if n in db.names():
            db.rm(n)
            db.save()
            click.echo('Removed {}'.format(n))
        else:
            click.echo('{} not found'.format(n))


@cli.command()
@click.option('--number', '-n', default=-1,
              help='Number of games to run. Default is forever (-1).')
@click.pass_context
def run(ctx, number):
    """
    Run some games.

    \b
    If no --number/-n option is given then games will continuously be run until
    a keyboard interrupt.
    """
    db = ctx.obj

    try:
        if number <= 0:
            while True:
                random_match(db)
        else:
            for _ in range(number):
                random_match(db)
    except KeyboardInterrupt:
        pass
    db.save()
    click.echo(db)


@cli.command()
@click.option('--reset', '-n', is_flag=True,
              help='Reset all ratings.')
@click.pass_context
def rankings(ctx, reset):
    """
    Display the rankings.
    """
    db = ctx.obj

    if reset:
        for name in db.names():
            db.reset_rating(name)
        db.save()
    click.echo(db)

if __name__ == '__main__':
    cli()