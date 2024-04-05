import json
import os

all_ids = "46480 45159 44781 41510 41085 41083 41003 40147 11712 11178 10751 10685 102234 101917 101908 101773 10143 45189 45194 10068 45503 45238 45662 45594 45420 45710 45801 45841 45243 46123 46172 46443 46440 46462 46466 19179 20043 32932 31249 34178 34610 45677 45290 44853 45271 45135 45261 45168 22367 44962".split(" ")

for json_path in [
    "data/gapartnet/grasps.json",
    "data/gapartnet/selected.json",
]:
    data = json.load(open(json_path, 'r'))
    data = { k: v for k, v in data.items() if k in all_ids }
    json.dump(data, open(json_path, 'w'), indent=2)
    
