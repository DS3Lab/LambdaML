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
methods = ['ASP', 'BSP']

n_epochs = 40

async_time = [12.4 * x for x in range(n_epochs)]
async_loss = [0.8376694971864870,
0.6562013355168430,
0.6518839678981090,
0.6520907824689690,
0.6590988094156440,
0.6491372314366430,
0.6507322517308320,
0.6483289084651250,
0.6559474766254430,
0.6462554741989480,
0.6469772458076480,
0.6455637920986520,
0.6449286991899660,
0.6435662508010860,
0.6513417742469090,
0.6441836113279520,
0.6441931534897200,
0.643351904370568,
0.6425313245166430,
0.6425399265506050,
0.6456466696479100,
0.6419690657745710,
0.6510137102820660,
0.6449013000184840,
0.6908822628584780,
0.6496237191286950,
0.6407786716114390,
0.6420177519321440,
0.6418354944749310,
0.6418675942854450,
0.6405146826397290,
0.6415414024483070,
0.6405761756680230,
0.7451567053794860,
0.649313972754912,
0.6399110582741820,
0.640630841255188,
0.6401394443078470,
0.6401938525113190,
0.6394357003948910]

sync_time = [11.2 * x for x in range(n_epochs)]
sync_loss = [0.6696668321436100,
             0.6597454466603020,
             0.6546685262159870,
             0.6517127616838980,
             0.6498110023411840,
             0.6484831463206900,
             0.6474824493581600,
             0.6466992009769790,
             0.6460416994311590,
             0.6454697440971030,
             0.64497556198727,
             0.644534398208965,
             0.6441385637630120,
             0.6437760970809240,
             0.6434356516057790,
             0.6431276391852990,
             0.6428403068672530,
             0.6425674178383570,
             0.64231722463261,
             0.6420816074718130,
             0.6418626172976060,
             0.6416590836915100,
             0.6414584111083640,
             0.6412763920697300,
             0.6411023167046630,
             0.6409352963620970,
             0.6407822289250110,
             0.6406334774060680,
             0.6404954167929560,
             0.6403606940399520,
             0.640235112472014,
             0.6401172469962730,
             0.6400038356130770,
             0.6398947401480240,
             0.6397953060540290,
             0.6396952027624300,
             0.6396049396558240,
             0.6395146277817810,
             0.6394303630698810,
             0.6393508396365430]


plt.figure(figsize=(4, 3))

plt.plot(async_time, async_loss, ".-", color=colors[0], label=methods[0])
plt.plot(sync_time, sync_loss, "x-", color=colors[1], label=methods[1])

plt.title("LR, Higgs, 16 workers", fontweight="bold")
plt.ylabel("loss")
plt.xlabel("seconds")
plt.yscale("log")
plt.xlim(0, 500)
plt.ylim(0.63, 0.76)
plt.xticks((0, 100, 200, 300, 400, 500), ('0', '100', '200', '300', '400', '500'))
plt.yticks((0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76), ('0.64', '0.66', '0.68', '0.70', '0.72', '0.74', '0.76'))
plt.legend(bbox_to_anchor=(0, 1), loc=2, ncol=3, shadow=False)

plt.tight_layout()

plt.savefig("sync_async_elasti_LR.pdf")
#plt.show()
