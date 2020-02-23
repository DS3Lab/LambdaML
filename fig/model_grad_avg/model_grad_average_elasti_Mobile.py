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

matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 6

#matplotlib.rcParams['hatch.linewidth'] = 0.1

matplotlib.rcParams['legend.fontsize'] = 6
matplotlib.rcParams['legend.edgecolor'] = 'Black'


hatches = ('', '**', '++', 'oo', '', '*', 'o', '.', 'O')
colors = ['Red', 'Skyblue', 'Orange', 'LightGrey', 'MediumSlateBlue', 'Tomato', 'Palegreen', 'Azure']

# SketchML, Adam, ZipML
methods = ['grad-average, lr=0.1', 'model-average, lr=0.1', 'centralized, lr=0.1']

grad_time = [214 * x for x in range(80)]
grad_loss = [2.079551,
1.701682,
1.58941,
1.454188,
1.350122,
1.247615,
1.151631,
1.091718,
1.03317,
0.963377,
0.908488,
0.8609,
0.809835,
0.806608,
0.75824,
0.720758,
0.693043,
0.671853,
0.619256,
0.619216,
0.592681,
0.574114,
0.544992,
0.521098,
0.52585,
0.505742,
0.490694,
0.469302,
0.472765,
0.462896,
0.451638,
0.423262,
0.410398,
0.408697,
0.40586,
0.408543,
0.392581,
0.36871,
0.360353,
0.348185,
0.343076,
0.349617,
0.320262,
0.333466,
0.306343,
0.322919,
0.310901,
0.299644,
0.275803,
0.290607,
0.285114,
0.263478,
0.278939,
0.259349,
0.238776,
0.256958,
0.260858,
0.263933,
0.247786,
0.266517,
0.251029,
0.24118,
0.235883,
0.207901,
0.233438,
0.214677,
0.209227,
0.196964,
0.207218,
0.188436,
0.176407,
0.199682,
0.200263,
0.185778,
0.180459,
0.157467,
0.169382,
0.180919,
0.16435,
0.179045]

model_time = [101 * x for x in range(160)]
model_loss = [2.28872,
2.117125,
2.039242,
1.948021,
1.909334,
1.905282,
1.79047,
1.775314,
1.677324,
1.633654,
1.595147,
1.564739,
1.582024,
1.485837,
1.620292,
1.385522,
1.393313,
1.413846,
1.363686,
1.225555,
1.241539,
1.208326,
1.181373,
1.06241,
1.096576,
1.061668,
1.043267,
0.995198,
0.979341,
0.925902,
0.982708,
0.883505,
0.894628,
0.869544,
0.854931,
0.796604,
0.801697,
0.789984,
0.76498,
0.782154,
0.746129,
0.703936,
0.72998,
0.663678,
0.678401,
0.6897,
0.676605,
0.610812,
0.630147,
0.623837,
0.615626,
0.611421,
0.679481,
0.617138,
0.601725,
0.542071,
0.567934,
0.559256,
0.557339,
0.598633,
0.629981,
0.50654,
0.506056,
0.484703,
0.577726,
0.483301,
0.526943,
0.478405,
0.496269,
0.454498,
0.48505,
0.441001,
0.528413,
0.411195,
0.445736,
0.434618,
0.559836,
0.478726,
0.471585,
0.402263,
0.407515,
0.38794,
0.438548,
0.392156,
0.427158,
0.367828,
0.389803,
0.368183,
0.555698,
0.388092,
0.39145,
0.348715,
0.392906,
0.341252,
0.372525,
0.359735,
0.371536,
0.334625,
0.350978,
0.319797,
0.36165,
0.328437,
0.367017,
0.307312,
0.289489,
0.305768,
0.332347,
0.305173,
0.399955,
0.294474,
0.285659,
0.292666,
0.309095,
0.299378,
0.284284,
0.277023,
0.290962,
0.281647,
0.28327,
0.260896,
0.36285,
0.256822,
0.249551,
0.262842,
0.422885,
0.257669,
0.2957,
0.250894,
0.226864,
0.235835,
0.265033,
0.233027,
0.245784,
0.239821,
0.230607,
0.228009,
0.348139,
0.214974,
0.268777,
0.22159,
0.21002,
0.248612,
0.25283,
0.298763,
0.225013,
0.199443,
0.205883,
0.206295,
0.453804,
0.201372,
0.19889,
0.18479,
0.184252,
0.192833,
0.215798,
0.187475,
0.208232,
0.18381,
0.228319,
0.170978]

single_time = [816 * x for x in range(19)]
single_loss = [1.784132,
1.402254,
1.181109,
1.02867,
0.907287,
0.817185,
0.743248,
0.685682,
0.646383,
0.605843,
0.572089,
0.541399,
0.523182,
0.492221,
0.464978,
0.454545,
0.43401,
0.415849,
0.401993
]

plt.figure(figsize=(3, 3))

plt.plot(grad_time, grad_loss, ".-", color=colors[0],  label=methods[0])
plt.plot(model_time, model_loss, "x-", color=colors[1],  label=methods[1])
plt.plot(single_time, single_loss, "o-", color=colors[2],  label=methods[2])

plt.title("MobileNet, Cifar10, 10 Workers", fontweight="bold")
plt.ylabel("loss")
plt.xlabel("seconds")
#plt.yscale("log")
#plt.xlim(0, 700)
#plt.ylim(1.5, 2.5)
#plt.xticks((0, 200, 400, 600), ('0', '200', '400', '600'))
#plt.yticks((1.6, 1.8, 2.0, 2.2, 2.4), ('1.6', '1.8', '2.0', '2.2', '2.4'))
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=1, shadow=False)

plt.tight_layout()

plt.savefig("model_grad_average_elasti_mobile.pdf")
#plt.show()
