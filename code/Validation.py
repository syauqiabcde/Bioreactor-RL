#%% Growth model

import numpy as np
import pandas as pd
from model_bioreactor import growth_model, bioreactor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os

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

svg_path = r'D:\Ahmad Syauqis Document\Paper - RL for microalgae\Paper work\Initial\Figure\SVG'
png_path = r'D:\Ahmad Syauqis Document\Paper - RL for microalgae\Paper work\Initial\Figure\PNG'

png_file = os.path.join(png_path, f"growth_validation.png")
svg_file = os.path.join(svg_path, f"growth_validation.svg")
    
fig.savefig(png_file, dpi=600, bbox_inches='tight')
fig.savefig(svg_file, bbox_inches='tight')
#%% Photosynthesis model

df = pd.read_csv('photosynthesis validation.csv')
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
FCO2 = 0.01*Fair
CO2 = 330 / 1e6 * 44 * 1000 # g/m3

O2_model = []
O2_datas = []
for i in range(df.shape[0]):
    I_sunlight = df['Irradiance(W/m2)'][i] / visible_wavelength_fration
    T = df['Temperature'][i]
    pH = df['pH'][i] 
    O2_data = df['O2 production (micro mol/mg chl/h)'][i] * 0.004 / 1e3 * 32 * 24 # from micro mol/mg chl/h to g O2/g biomass/day
    
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
    O2_datas.append(O2_data)

accuracy = (1-mean_absolute_percentage_error(O2_datas, O2_model))
p = min(min(O2_datas), min(O2_model))
q = max(max(O2_datas), max(O2_model))

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
ax.plot([p,q], [p,q])
ax.scatter(O2_datas, O2_model)
ax.text(0, 2, f'Accuracy: {accuracy*100:.1f}%')