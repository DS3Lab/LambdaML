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
methods = ['LR (Higgs, 28)', 'Mobilenet (Cifar10, 12M)', 'Resnet (Cifar10, 89M)']

N = 2
bar_width = 0.3
indexes = np.arange(N)

# computation, communication
lr = [0.004, 1.93]
mobile = [2.19, 3.48]
resnet = [6.76, 7.44]


def autolabel(rects, label):
    i = 0
    for rect in rects:
        plt.text(rect.get_x() + 0.5*rect.get_width(), 1.0*rect.get_height(), label[i], fontsize=5, ha='center', va='bottom')
        i = i + 1


plt.figure(figsize=(4, 2))

rec1 = plt.bar(indexes+0.2, lr, width=bar_width, color=colors[0], edgecolor="Black", linewidth=1, hatch=hatches[0], label=methods[0])
autolabel(rec1, lr)
rec2 = plt.bar(indexes+0.5, mobile, width=bar_width, color=colors[1], edgecolor="Black", linewidth=1, hatch=hatches[1], label=methods[1])
autolabel(rec2, mobile)
rec3 = plt.bar(indexes+0.8, resnet, width=bar_width, color=colors[2], edgecolor="Black", linewidth=1, hatch=hatches[2], label=methods[2])
autolabel(rec3, resnet)

plt.title("Time per batch", fontweight="bold")
plt.ylabel("seconds")
plt.xlim(0, N)
plt.yscale("log")
plt.ylim(0.001, 1000)
plt.xticks(indexes+0.5, ('Computation', 'Communication'))
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=2, shadow=False)

plt.tight_layout()

plt.savefig("model_size.pdf")
#plt.show()