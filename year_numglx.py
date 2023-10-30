#import packages
import math as m
import statistics as stat
import numpy as np
import csv
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('C:/Users/hchri/OneDrive/Documents/GitHub/Galaxy-Surveys-Paper/dates.txt',  delimiter='\t', header=0, nrows=30)

surveys = df.iloc[:, 0]

year = df.iloc[:, 1]

num_G = df.iloc[:, 2]

max_D = df.iloc[:, -2]
print(max_D)

#%%


x_dat = np.linspace(0, len(num_G), len(num_G))
plt.scatter(surveys,num_G, label = '# of glxs')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yscale('log')
sns.regplot(x = x_dat, y = num_G, scatter= False, fit_reg = True)
'''
plt.ylabel('log(# galaxies)')
plt.xlabel('Surveys')
plt.savefig('num_glx.png', bbox_inches = 'tight', dpi = 200)
'''
#%%

x_dat = np.linspace(0, len(max_D), len(max_D))
plt.scatter(surveys,max_D, label = 'Distance')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
sns.regplot(x = x_dat, y = max_D, scatter= False, fit_reg = True)
plt.ylabel('Maximum Distance (Mpc) \n & # sample glx')
plt.xlabel('Surveys')
plt.yscale('log')
plt.ylim(0, max(num_G)+10)
plt.legend()
plt.savefig('both.png', bbox_inches = 'tight', dpi = 200)

#%%
min_num = min(max_D)
max_num = max(max_D)
median = stat.median(max_D)
print('min:', min_num, '\n max:', max_num, '\n median:', median)