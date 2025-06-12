#%% Growth model

import numpy as np
import pandas as pd
from model_bioreactor import growth_model, bioreactor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os
import seaborn as sns

df = pd.read_csv('Validation dataset.csv')
df = df.interpolate()
df['x'] = df['x'] * 1000
x_model = [df['x'][0]]
x_data = [df['x'][0]]
mius = []
dt = df['Day'][1] - df['Day'][0] 

for i in range(df.shape[0]-1):
    x = df['x'][i] 
    irradiance = df['Irradiance (W/m2)'][i]
    temperature = df['Temperature (deg C)'][i]
    pH = df['pH'][i]
    DO = df['DO (micro mol/L)'][i] * 32 / 1e6 * 1000
    substrate = df['Nitrate'][i] * 62.01 + df['Phospate'][i] * 94.97
    carbon = df['DIC'][i] * 44 
    
    
    model = growth_model(x_model[i], 
                         irradiance, 
                         temperature, 
                         pH, 
                         DO, 
                         substrate, 
                         carbon)
    
    miu, _, _, _, _, _ = model.solve()

    x_model.append(x_model[i]*(1+miu*dt))
    x_data.append(x)
    mius.append(miu)

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
ax.plot(df['Day'], x_model, label='Model', color = '#003366', linewidth=3)
ax.scatter(df['Day'], x_data, label='Data', color = '#96a1a8')
ax.legend(frameon=False,)
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'Biomass concentration (g $\mathrm{m}^{-3}$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.0, 120, f'Accuracy: {(1-mean_absolute_percentage_error(x_data, x_model))*100:.1f}%')
ax.text(-0.12, 1.05, 'a', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
ax.set_xticks(np.arange(0,22,2))
ax.set_xticklabels(np.arange(0,22,2))

#%% Photosynthesis model

df = pd.read_csv('photosynthesis validation.csv')
df['O2 data'] = df['O2 production (micro mol/mg chl/h)'] * 0.004 / 1e3 * 32 * 24 # from micro mol/mg chl/h to g O2/g biomass/day
x = 1 # g/m3

# Below is not important parameter for this particular use i.e. photosynthesis validation 
S = 1000 
FH2O = 0
Facid = 0
Fbase = 0
O2 = 10
d = 1.5/1000
z = 500/1e6 * 4 / (np.pi * d**2)
T_env = 30
RH = 0.5
wind_speed = 0.1
Q_heat = 0
I_red_light = 0
I_blue_light = 0
level = 0.7
Fs = 0   

###

visible_wavelength_fration = 0.42 # https://en.wikipedia.org/wiki/Sunlight
Fair = 200/1000/60

CO2 = 330 / 1e6 * 44 * 1000 # g/m3

O2_model = []

for i in range(df.shape[0]):
    I_sunlight = df['Irradiance(W/m2)'][i] / visible_wavelength_fration
    T = df['Temperature'][i]
    pH = df['pH'][i] 
    
    FCO2 = df['CO2 ratio'][i]*Fair

    model =  bioreactor(x, 
                S,  
                I_sunlight,
                I_red_light,
                I_blue_light, 
                level,
                Fs,
                FCO2,
                Fair,
                FH2O,
                Facid,
                Fbase,
                T,
                CO2,
                O2,
                pH, 
                d, 
                z,
                T_env,
                RH,
                wind_speed,
                Q_heat)
    
    rO2 = model.rO2p() / model.x_mass
    O2_model.append(rO2)

refs = df['Ref'].unique()
colors = sns.color_palette("tab10", n_colors=len(refs)+1)
markers = ['o', 'x', '+']

df['O2 model'] = O2_model
accuracy = (1-mean_absolute_percentage_error(df['O2 data'], df['O2 model']))
p = min(min(df['O2 data']), min(df['O2 model']))
q = max(max(df['O2 data']), max(df['O2 model']))

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
ax.plot([p,q], [p,q], linewidth=3, color=colors[0])

for i, ref in enumerate(refs):
    ax.scatter(df[df['Ref'] == ref]['O2 data'], df[df['Ref'] == ref]['O2 model'],
               label=ref, 
               color=colors[i+1], 
               alpha=0.8,
               marker=markers[i])

ax.text(p*1.2, q*0.6, f'Accuracy: {accuracy*100:.1f}%')
ax.set_ylabel(r'Predicted O$_2$ production (g$_{\mathrm{O_2}}$ g$_\mathrm{biomass}^{-1}$ day$^{-1}$)')
ax.set_xlabel(r'Actual O$_2$ production (g$_{\mathrm{O_2}}$ g$_\mathrm{biomass}^{-1}$ day$^{-1}$)')
ax.legend(frameon=False)
ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
