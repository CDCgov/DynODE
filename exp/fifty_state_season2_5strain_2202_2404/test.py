import json
import os

import numpy as np

path = "/output/fifty_state_season2_5strain_2202_2404/SMH_5strains_240819_v20/"
fpath = os.path.join(path, "AK", "checkpoint.json")
with open(fpath) as json_file:
    d = json.load(json_file)
print(d.keys())
g = {key: np.array(values)}
print(np.array(d["INTRODUCTION_TIMES_0"][0][:50]).shape)
