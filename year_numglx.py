#import packages
import math as m
import statistics as stat
import numpy as np
import csv
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (12,8)
# %%
df = pd.read_csv('survey_info_Ysort.txt',  delimiter='\t', header=0, nrows=50)

surveys = df.iloc[:, 0]

year = df.iloc[:, 1]

num_G = df.iloc[:, 2]

max_D = df.iloc[:, 3]
print(max_D)

#%%

x_dat = np.linspace(0, len(num_G), len(num_G))
plt.scatter(year,num_G, label = '# of glxs', s=40, edgecolors='k')

plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yscale('log')

# a, b = np.polyfit(x_dat, num_G, 1)
# plt.plot(x_dat, a*x_dat+b)
#plt.fill_between(x_dat, y1 = (a*x_dat+b)-st_dev, y2 = (a*x_dat+b)+st_dev)
#sns.regplot(x = x_dat, y = num_G, scatter= False, fit_reg = True)

plt.ylabel('Number of Galaxies')
plt.xlabel('Surveys')
plt.grid()
plt.show()
#plt.savefig('num_glx.png', bbox_inches = 'tight', dpi = 500)

#%%

x_dat = np.linspace(0, len(max_D), len(max_D))
plt.scatter(year,max_D, label = 'Distance', s=40, edgecolors='k')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# a, b = np.polyfit(x_dat, max_D, 1)
# plt.plot(x_dat, a*x_dat+b)
plt.ylabel('Maximum Distance (Mpc)')
plt.xlabel('Surveys')
plt.yscale('log')
#plt.ylim(0, max(num_G))
#plt.legend()
plt.grid()
plt.show()
#plt.savefig('maxD.png', bbox_inches = 'tight', dpi = 500)

#%%
min_num = min(max_D)
max_num = max(max_D)
median = stat.median(max_D)
print('min:', min_num, '\n max:', max_num, '\n median:', median)
