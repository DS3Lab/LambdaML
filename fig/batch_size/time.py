import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


#matplotlib.rcParams['font.family'] = "sans-serif"
#matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
#matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.style'] = "italic"
matplotlib.rcParams['font.weight'] = "bold"

matplotlib.rcParams['lines.linewidth'] = 3

#matplotlib.rcParams['axes.linewidth'] = 2
#matplotlib.rcParams['axes.edgecolor'] = 'Grey'
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['figure.titleweight'] = 'bold'

matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 7

#matplotlib.rcParams['hatch.linewidth'] = 0.1

matplotlib.rcParams['legend.fontsize'] = 5
matplotlib.rcParams['legend.edgecolor'] = 'Black'


hatches = ('', '\\\\\\', '+++', 'xxx', '///', '...', '', '*', 'o', '.', 'O')
colors = ['Red', 'Skyblue', 'Orange', 'LightGrey', 'MediumSlateBlue', 'Tomato', 'Palegreen', 'Azure']

# SketchML, Adam, ZipML, DGC, QSGD, TernGrad
methods = ['batch size=10K', 'batch size=1K', 'batch size=100']

N = 2
bar_width = 0.3
indexes = np.arange(N)

# computation, communication
batch10k = [1.5, 11]
batch1k = [1.37, 10.5]
batch100 = [1.75, 10]


def autolabel(rects, label):
    i = 0
    for rect in rects:
        plt.text(rect.get_x() + 0.5*rect.get_width(), 1.0*rect.get_height(), label[i], fontsize=5, ha='center', va='bottom')
        i = i + 1


plt.figure(figsize=(4, 2))

rec1 = plt.bar(indexes+0.2, batch10k, width=bar_width, color=colors[0], edgecolor="Black", linewidth=1, hatch=hatches[0], label=methods[0])
rec2 = plt.bar(indexes+0.5, batch1k, width=bar_width, color=colors[1], edgecolor="Black", linewidth=1, hatch=hatches[1], label=methods[1])
rec3 = plt.bar(indexes+0.8, batch100, width=bar_width, color=colors[2], edgecolor="Black", linewidth=1, hatch=hatches[2], label=methods[2])

plt.title("Time per batch", fontweight="bold")
plt.ylabel("seconds")
plt.xlim(0, N)
plt.yscale("log")
plt.ylim(0.001, 1000)
plt.xticks(indexes+0.5, ('Computation', 'Communication'))
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=2, shadow=False)

plt.tight_layout()

plt.savefig("time.pdf")
#plt.show()