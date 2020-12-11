import numpy as np
import matplotlib.pyplot as plt

task = 2

# read data
with open('outputDataT'+str(task)+'.txt', 'r') as f:
    t1_data = [x.strip() for x in f.readlines()]

# extract data
data = {}
for result in t1_data:
    i, a, s, e, h, r = result.split(', ')
    h, r = float(h), float(r)
    instance_name = i[-7:-4]
    if not instance_name in data.keys():
        data[instance_name] = {}
    if not a in data[instance_name].keys():
        data[instance_name][a] = {}
    if not h in data[instance_name][a].keys():
        data[instance_name][a][h] = []
    data[instance_name][a][h].append(r)


# plot
for i_name, i_results in data.items():
    plt.title("Instance: " + i_name)
    for a_name, a_res in i_results.items():
        x = list(a_res.keys())
        vals = np.array([list(a_res[x]) for x in a_res.keys()])
        y = np.mean(vals, axis=1)
        plt.plot(x, y, label=a_name, marker='x')
    plt.legend()    
    plt.grid(True)
    plt.xlabel("Horizon")
    plt.ylabel("Regret")
    plt.xscale("log")
    plt.show()