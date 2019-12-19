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
methods = ['grad-average', 'model-average']

grad_time = [197.63, 398.3974, 613.3907, 816.8052]
grad_loss = [3.996873, 3.963667, 3.931304, 3.912611]

model_time = [11.0693, 24.0635, 37.4493, 50.3778, 63.0248, 76.9139, 91.5786, 104.9062, 118.8242, 133.6197, 146.8182, 160.1805, 173.3149, 186.8998, 199.7228, 213.5177, 226.2654, 240.2631, 253.6133, 266.6198, 279.6308, 293.8732, 307.044, 321.1432, 334.8375, 347.7972, 361.0746, 374.6305, 388.228, 402.7437, 415.9754, 427.9176, 441.3588, 454.589, 468.0321, 481.6141, 494.9695, 508.7619, 522.42, 536.2503, 550.5886, 563.6938, 576.8008, 590.2048, 603.9208, 617.8739, 632.1903, 645.8492, 659.8199, 675.2341, 690.3464, 704.6822, 719.2336, 733.0605, 747.9558, 761.9374, 776.5032, 790.3845, 804.0336, 817.6056, 832.2836, 845.5774, 859.2567, 873.1643]
model_loss = [3.408051, 3.374608, 3.349239, 3.333841, 3.317036, 3.304748, 3.293512, 3.28597, 3.277847, 3.275613, 3.269871, 3.264814, 3.260914, 3.254884, 3.253005, 3.253499, 3.247252, 3.248328, 3.245541, 3.245125, 3.244374, 3.244656, 3.238484, 3.241946, 3.240275, 3.239615, 3.237707, 3.230612, 3.239058, 3.23948, 3.234013, 3.231568, 3.230653, 3.231289, 3.229683, 3.229654, 3.231871, 3.229378, 3.232246, 3.22671, 3.228266, 3.227122, 3.229104, 3.225784, 3.225956, 3.225816, 3.225937, 3.224368, 3.22629, 3.223989, 3.224967, 3.221926, 3.220918, 3.223558, 3.223958, 3.222336, 3.224216, 3.222418, 3.219663, 3.220976, 3.222634, 3.220595, 3.218138, 3.219301]

plt.figure(figsize=(4, 2.7))

plt.plot(grad_time, grad_loss, ".-", color=colors[0],  label=methods[0])
plt.plot(model_time, model_loss, "x-", color=colors[1],  label=methods[1])


plt.title("Grad Average vs. Model Average", fontweight="bold")
plt.ylabel("loss")
plt.xlabel("seconds")
plt.yscale("log")
plt.xlim(0, 1000)
plt.ylim(3.1, 4.5)
plt.xticks((0, 500, 1000), ('0', '500', '1000'))
plt.yticks((3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4), ('3.2', '3.4', '3.6', '3.8', '4', '4.2', '4.4'))
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=3, shadow=False)

plt.tight_layout()

plt.savefig("model_grad_average.pdf")
#plt.show()
