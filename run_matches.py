# halite -d "30 30" "python3 shummiev3-6.py" "python3 shummiev3-5.py"
import subprocess
import re
from collections import Counter

num_games = 100
games_played = 0
rank_list = []


while games_played < num_games:
    if games_played % 5 == 0:
        print("Running Game #:" + str(games_played))
    #stdoutdata = subprocess.getoutput('halite -d "30 30" "python3 shummiev3-7.py" "python3 shummiev3-5.py"')
    stdoutdata = subprocess.getoutput('halite -d "30 30" "python shummiev7-10-1.py" "python shummiev7-10-2.py" "python shummiev7-10-3.py"')

    players = re.findall("Player #[0-9], (.*), came", stdoutdata)
    rank = re.findall("came in rank #([0-9])", stdoutdata)
    
    for a, b in zip(players, rank):
        rank_list.append(str(a) + ": " + str(b))

    #subprocess.getoutput("rm *.hlt")
    print(Counter(rank_list).most_common())
    
    games_played += 1
    
    
    
# Display stats.
#print(Counter(rank_list).most_common())


    
