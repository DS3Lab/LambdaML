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
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 6

#matplotlib.rcParams['hatch.linewidth'] = 0.1

matplotlib.rcParams['legend.fontsize'] = 6
matplotlib.rcParams['legend.edgecolor'] = 'Black'


hatches = ('', '**', '++', 'oo', '', '*', 'o', '.', 'O')
colors = ['Red', 'Skyblue', 'Orange', 'LightGrey', 'MediumSlateBlue', 'Tomato', 'Palegreen', 'Azure']

# SketchML, Adam, ZipML
methods = ['ADMM', 'model-average']

admm_time = [9.5, 19.0, 28.5, 38.0, 47.5, 57.0, 66.5, 76.0, 85.5, 95.0, 106.5, 116.0, 125.5, 135.0, 144.5, 154.0, 163.5, 173.0, 182.5, 192.0]
admm_loss = [63.1330, 60.1968, 59.8505, 59.5648, 59.3204, 59.1074, 58.9203, 58.7542, 58.6059, 58.4731, 58.3534, 58.2450, 58.1469, 58.0576, 57.9761, 57.9017, 57.8332, 57.7705, 57.7126, 57.6594]


model_time = [15.8179, 27.2118, 39.0617, 50.4708, 61.4858, 72.3735, 84.2673, 95.4679, 107.2217, 119.1158, 130.4545, 141.7029, 153.1814, 164.7778, 176.4101, 188.8527, 200.2005, 211.4545, 222.6291, 233.7142]
model_loss = [63.1330, 60.1968, 59.8505, 59.5648, 59.3204, 59.1074, 58.9203, 58.7542, 58.6059, 58.4731, 58.3534, 58.2450, 58.1469, 58.0576, 57.9761, 57.9017, 57.8332, 57.7705, 57.7126, 57.6594]

plt.figure(figsize=(4, 2.7))

plt.plot(admm_time, admm_loss, ".-", color=colors[0], label=methods[0])
plt.plot(model_time, model_loss, "x-", color=colors[1],  label=methods[1])


plt.title("ADMM vs. Model Average", fontweight="bold")
plt.ylabel("loss")
plt.xlabel("seconds")
plt.xlim(0, 200)
plt.ylim(57, 64)
plt.xticks((0, 100, 200), ('0', '100', '200'))
plt.yticks((57, 58, 59, 60, 61, 62, 63, 64), ('57', '58', '59', '60', '61', '62', '63', '64'))
plt.yscale("log")
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=3, shadow=False)

plt.tight_layout()

plt.savefig("admm_vs_model_avg.pdf")
#plt.show()
