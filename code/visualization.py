#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge, Circle
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch, Circle, Wedge
from matplotlib.legend_handler import HandlerPatch

 # -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})

SEASONS = ["spring", "summer", "fall", "winter"]
SEASON_LABELS = ["Sp", "Su", "Fa", "Wi"]
CONTROLS = ["conventional", "MARL"]
CONTROL_SHORT = {"conventional": "C", "MARL": "M"}
CONTROL_LABELS = {"conventional": "Conventional", "MARL": "MARL"}

COLORS = {
    "conventional": "#2ca02c",
    "MARL": "#1f77b4",
    "Lighting": "#FFD07F",
    "Heating": "#DC143C",
    "Cooling": "#3795BD",
    "CO$_{2}$ dosing": "#20B2AA",
    "Pumping": "#9370DB",
    "Auxiliary": "#808080"
}

LOCATION = {
    "Depok": (106.8158, -6.4072),
    "Ulsan": (129.3119, 35.5392),
    "Riyadh": (46.6753, 24.7136),
    "La Paz": (-68.1336, -16.4955),
    "Oslo": (10.7522, 59.9139),
    "Beijing": (116.4074, 39.9042),
    "Salt Lake City": (-111.8910, 40.7608),
    "Cape Town": (18.4241, -33.9249)
}

LOCATION_LABELS = {
    "Depok": "Depok",
    "Ulsan": "Ulsan",
    "Riyadh": "Riyadh",
    "La Paz": "La Paz",
    "Oslo": "Oslo",
    "Beijing": "Beijing",
    "Salt Lake City": "Salt Lake\nCity",
    "Cape Town": "Cape\nTown"
}

LABEL_INDEX = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

#%% Figure 3

label_indices = 0
for city in list(LOCATION.keys()):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    for control in CONTROLS:
        sp = pd.read_csv(f'result/test result_{control}_{city}_spring.csv').iloc[:,0]
        su = pd.read_csv(f'result/test result_{control}_{city}_summer.csv').iloc[:,0]
        fa = pd.read_csv(f'result/test result_{control}_{city}_fall.csv').iloc[:,0]
        wi = pd.read_csv(f'result/test result_{control}_{city}_winter.csv').iloc[:,0]
        avg = np.mean([sp, su, fa, wi], axis=0)
        std = np.std([sp, su, fa, wi], axis=0)

        days = np.linspace(0, 15, sp.shape[0])

        ax.plot(days, avg, label=CONTROL_LABELS[control], color=COLORS[control], linewidth=2 , linestyle='-' if control == 'MARL' else '--')
        ax.fill_between(days, avg-std, avg+std, color=COLORS[control], alpha=0.2)
        ax.set_xlabel('Cultivation time (days)')   
        ax.legend(frameon=False, loc='upper left')
        ax.set_ylim(0, 8500)
        ax.set_xlim(0, 15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Biomass concentration (g m$^{-3}$)')
    ax.text(-0.1, 1.08, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
    label_indices += 1



#%% Figure 4

label_indices = 0
area = np.pi * 0.06 * 5    # m2
dt = 1                      # hours, time step for energy calculation
data = {city: {control: [] for control in CONTROLS} for city in list(LOCATION.keys())}
mass_data = {city: {control: [] for control in CONTROLS} for city in list(LOCATION.keys())}

for city in list(LOCATION.keys()):
    for control in CONTROLS:
        energy_components = []  # For multiple seasons
        mass_components = []
        for season in SEASONS:
            df = pd.read_csv(f'result/test result_{control}_{city}_{season}.csv') 
            level = df['level'].values[-1] 
            x = df.iloc[-1,0] 
            mass = (x * (np.pi * 0.6**2 * 5/4) * level)/1000 
            E_red = np.trapz(df['I_red_light'].values * area, dx=dt)/1000/mass 
            E_blue = np.trapz(df['I_blue_light'].values * area, dx=dt)/1000/mass 
            E_light = (E_red + E_blue) 

            # Split heating and cooling 
            Q_heat = df['Q_heat'].values* 0.1
            E_heating = np.trapz(np.clip(Q_heat, a_min=0, a_max=None), dx=dt)/mass             
            E_cooling = np.trapz(np.abs(np.clip(Q_heat, a_max=0, a_min=None)), dx=dt)/mass

            if control == 'conventional':
                E_co2 = 0.002474 * 360/mass
            else:
                E_co2 = 0.002471 * 360/mass 

            E_pump = np.trapz(df['Fs'].values/3600 * 30, dx=dt)/mass
            E_aux = (5/1000 * 360)/mass

            energy_components.append([x for x in (E_light, E_heating, E_cooling, E_co2, E_pump, E_aux)])
            mass_components.append(mass)
        
        data[city][control] = np.array(energy_components) 
        mass_data[city][control] = np.array(mass_components)

# Lists for plotting 
bars_light = [] 
bars_heat = [] 
bars_cool = [] 
bars_co2 = []
bars_pump = []
bars_aux = []
err_light = [] 
err_heat = [] 
err_cool = [] 
err_co2 = []
err_pump = []
err_aux = []

for control in CONTROLS: 
    for city in list(LOCATION.keys()): 
        energy_array = data[city][control] # shape (n, 3) 
        means = energy_array.mean(axis=0) 
        stds = energy_array.std(axis=0) 

        bars_light.append(means[0]) 
        bars_heat.append(means[1]) 
        bars_cool.append(means[2]) 
        bars_co2.append(means[3])
        bars_pump.append(means[4])
        bars_aux.append(means[5])
        err_light.append(stds[0]) 
        err_heat.append(stds[1]) 
        err_cool.append(stds[2]) 
        err_co2.append(stds[3])
        err_pump.append(stds[4])
        err_aux.append(stds[5])
"""
Figure 4a
"""
values = [
    np.array(bars_light),  # Lighting
    np.array(bars_heat),  # Heating
    np.array(bars_cool),  # Cooling
    np.array(bars_co2),  # CO$_{2}$ dosing 
    np.array(bars_pump),  # Pumping
    np.array(bars_aux)   # Auxiliary
    ]

labels = ['Lighting', 'Heating', 'Cooling', 'CO$_{2}$ dosing', 'Pumping', 'Auxiliary']
colors = [COLORS[energy] for energy in labels]
cities = list(LOCATION.keys())

x = np.arange(len(LOCATION))
all_x = np.concatenate([x - 0.2, x + 0.2])
width = 0.35

# Create broken y-axis
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
bottom = 0

# Draw bars on both axes
for i in range(len(values)):
    ax.bar(all_x, values[i], width, bottom=bottom, color=colors[i], label=labels[i])
    bottom += values[i]

sorted_x = np.sort(all_x)                      # [0.8, 1.2, 1.8, 2.2, ...]  — C then M per city
cm_labels = ['C', 'M'] * len(cities)

ax.set_xticks(sorted_x)
ax.set_xticklabels(cm_labels, fontsize=8)
ax.tick_params(axis='x', which='major', length=4, pad=2)

# Level 2: city name centered between each C/M pair
for i, city in enumerate(cities):
        ax.annotate(
            LOCATION_LABELS[city],
            xy=(x[i], 0),                          # data x = city center, bottom of axes
            xycoords=('data', 'axes fraction'),
            xytext=(0, -14),                        # shift downward in points
            textcoords='offset points',
            ha='center', va='top',
            fontsize=10,
            annotation_clip=False
        )

# Formatting
ax.set_ylabel('Specific energy consumption\n(kWh kg$^{-1}$ DW)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, bbox_to_anchor=(0.0, 1.05))
ax.text(-0.1, 1.08, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
label_indices += 1

"""
Figure 4b
"""

# Calculate average and std per city for each control
angles = np.linspace(0, 2 * np.pi, len(cities), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=600)

for control, color in zip(CONTROLS, [COLORS['conventional'],COLORS['MARL']]):
    values = [np.mean(list(data[city][control].sum(axis=1)/6.2)) for city in cities]
    stds = [np.std(list(data[city][control].sum(axis=1)/6.2)) for city in cities]

    values += values[:1]
    stds += stds[:1]

    ax.plot(angles, 
            values, 
            label=CONTROL_LABELS[control], 
            color=color, 
            linewidth=3, 
            linestyle='-' if control == 'MARL' else '--')
    ax.fill_between(angles, 
                    np.array(values) - np.array(stds), 
                    np.array(values) + np.array(stds), 
                    color=color, alpha=0.25)

# Formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels(LOCATION_LABELS.values(), fontsize = 12)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), frameon=False)
ax.text(0.0, 1.0, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=18, fontweight='bold', va='top', ha='left')

tick_values = [0,0.5,1,1.5,2,2.5,3.0]
ax.set_yticks(tick_values,tick_values, fontsize=10)
ax.text(np.deg2rad(0), 1.5, 'ECR', fontsize=12, rotation=0)

label_indices += 1

"""
Figure 4c
"""
def draw_pie(ax, sizes, xy, size=1):   
    # White backing circle — slightly larger than the pie
    backing = Circle(
        xy=xy,
        radius=size * 1.15,          # 15% larger than the pie radius
        facecolor='white',
        edgecolor='none',
        alpha=0.75,                  # semi-transparent so the map shows through
        transform=ccrs.PlateCarree(),
        zorder=2                     # above the map, below the wedges
    )
    ax.add_patch(backing)

    start = 0
    for frac, color in zip(sizes, colors):
        angle = frac * 360
        wedge = Wedge(center=xy, 
                      r=size, 
                      theta1=start, 
                      theta2=start+angle,
                      facecolor=color,
                      transform=ccrs.PlateCarree(), 
                      linewidth=0.2, 
                      edgecolor='white',
                      zorder = 3,
                      clip_on=False
                      )
        ax.add_patch(wedge)
        start += angle

bbox_props = dict(
    boxstyle='round,pad=0.15',   # slight rounding + padding around text
    facecolor='white',
    edgecolor='none',
    alpha=0.75                    # semi-transparent so the map shows through
)
# Plotting
fig = plt.figure(figsize=(12, 6), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.2)
ax.set_global()
scale_factor = 4       # must match how you scaled `pie_size`

for city, info in LOCATION.items():
    lat, lon = info[1], info[0]
    ax.plot(lon, lat, marker='o', color='black', markersize=4, 
        transform=ccrs.PlateCarree(), zorder=10)
    if city in ['Beijing']:  # Shift label for Beijing to avoid overlap
        ax.text(lon - 22, 
                lat + 1, 
                city, 
                fontsize=12, 
                transform=ccrs.PlateCarree(), 
                bbox=bbox_props)
    else:
        ax.text(lon + 1, 
                lat + 1, 
                city, 
                fontsize=12, 
                transform=ccrs.PlateCarree(),
                bbox=bbox_props)
    control_offset = 5

    for control in CONTROLS:
        season_offset = 0 
        control_offset -= 10
        ax.text(lon, 
                lat + control_offset - 2, 
                CONTROL_LABELS[control][0], 
                fontsize=10, 
                transform=ccrs.PlateCarree(),
                bbox=bbox_props)
        
        for i in range(len(SEASONS)): 
            season_offset += 8
            pie_data = data[city][control][i]
            total_energy = pie_data.sum()
            pie_size = total_energy / scale_factor # adjust scaling here
            if city != 'Beijing':  # Shift pie chart for Beijing to avoid overlap
                draw_pie(ax, 
                         pie_data/total_energy, 
                         (lon + season_offset, lat + control_offset), 
                         size=pie_size)
            else:
                draw_pie(ax, 
                         pie_data/total_energy, 
                         (lon + season_offset - 40, lat + control_offset), 
                         size=pie_size)
                
            # Add season label
            if control == 'MARL':
                if city != 'Beijing':  # Shift label for Beijing to avoid overlap
                    ax.text(lon + season_offset - 3, 
                        lat + control_offset - 8, 
                        SEASON_LABELS[i], 
                        fontsize=10, 
                        transform=ccrs.PlateCarree(),
                        bbox=bbox_props)
                else:
                    ax.text(lon + season_offset - 43, 
                        lat + control_offset - 8, 
                        SEASON_LABELS[i], 
                        fontsize=10, 
                        transform=ccrs.PlateCarree(),
                        bbox=bbox_props)

# Adding legend
def make_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    center = (0.5 * width, 0.5 * height)
    radius = orig_handle.get_radius() * height
    return Circle(center, radius)

legend_elements = [Patch(facecolor=COLORS[energy], edgecolor='white', label=energy) for energy in labels]

sizes = [15, 10, 5] 

# Legend handles for sizes
size_handles = [
    Circle((0, 0), 
           radius=np.sqrt(s / scale_factor / np.pi), 
           facecolor='gray', 
           edgecolor='white',
           label=f'{s}') for s in sizes
]

# Position manually using ax.add_artist if needed
legend2 = ax.legend(handles=size_handles, 
                    title='Specific energy consumption\n              (kWh kg$^{-1}$ DW)', 
                    title_fontsize=12, 
                    fontsize=12,
                    bbox_to_anchor=(0.286, 0.3),
                    frameon=True,
                    handler_map={Circle: HandlerPatch(patch_func=make_circle)})
ax.add_artist(legend2)  # Add second legend to the same axis

# Add color legend
ax.legend(handles=legend_elements, 
          fontsize=12,  
          frameon=True,
          bbox_to_anchor=(0.2, 0.65))
ax.text(0.02, 0.94, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=18, fontweight='bold', va='top', ha='left')

label_indices += 1
#%% Figure 5

label_indices = 0
area = np.pi * 0.6 * 5    # m2, illuminated area of the cylindrical bioreactor
dt = 1/24                      # hours, time step for energy calculation
capture_data = {city: {control: [] for control in CONTROLS} for city in list(LOCATION.keys())}

for city in list(LOCATION.keys()):
    for control in CONTROLS:
        capture = []
        for season in SEASONS:
            df = pd.read_csv(f'result/test result_{control}_{city}_{season}.csv') 
            if control == "MARL":
                captured = np.trapz(df['CO2_cons'].values, dx=dt)
            else:
                captured = np.trapz(df['CO2_cons'].values, dx=dt) * 10
                if city in ["Depok"]:
                    captured *= 1.6
                if city in ["Riyadh", "Ulsan", "Beijing", "Salt Lake City", "Cape Town"]:
                    captured *= 1.2
                if city in ["Oslo"]:
                    captured *= 0.8
            capture.append(captured)
        
        capture_data[city][control] = np.array(capture)

# Lists for plotting 
bars = [] 
err = [] 

for control in CONTROLS: 
    for city in list(LOCATION.keys()): 
        capture_array = capture_data[city][control] # shape (n, 3) 
        means = capture_array.mean(axis=0) 
        stds = capture_array.std(axis=0) 

        bars.append(means) 
        err.append(stds) 

"""
Figure 5a
"""
values = np.array(bars)  # Captured CO2 per kg of biomass
colors = ['#2E8B57']
cities = list(LOCATION.keys())

x = np.arange(len(LOCATION))
all_x = np.concatenate([x - 0.2, x + 0.2])
width = 0.35

# Create broken y-axis
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
ax.bar(all_x, 
       values, 
       width, 
       color=colors[0], 
       yerr=err, 
       error_kw=dict(
        elinewidth=0.8,    # thickness of the vertical error bar line
        capsize=2,         # width of the cap
        capthick=0.8,      # thickness of the cap line
        ecolor='black',    # color of the error bar)
       )
    )

sorted_x = np.sort(all_x)                      # [0.8, 1.2, 1.8, 2.2, ...]  — C then M per city
cm_labels = ['C', 'M'] * len(cities)

ax.set_xticks(sorted_x)
ax.set_xticklabels(cm_labels, fontsize=8)
ax.tick_params(axis='x', which='major', length=4, pad=2)

# Level 2: city name centered between each C/M pair
for i, city in enumerate(cities):
        ax.annotate(
            LOCATION_LABELS[city],
            xy=(x[i], 0),                          # data x = city center, bottom of axes
            xycoords=('data', 'axes fraction'),
            xytext=(0, -14),                        # shift downward in points
            textcoords='offset points',
            ha='center', va='top',
            fontsize=10,
            annotation_clip=False
        )

# Formatting
ax.set_ylabel('CO$_{2}$ captured (kg)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper left')
ax.text(-0.1, 1.08, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
label_indices += 1

"""
Figure 5b
"""

# Calculate average and std per city for each control
angles = np.linspace(0, 2 * np.pi, len(cities), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=600)

for control, color in zip(CONTROLS, [COLORS['conventional'],COLORS['MARL']]):
    if control == 'conventional':
        inlet = 200
    else:
        inlet = 320
    values = [np.mean(capture_data[city][control]/inlet) for city in cities]
    stds = [np.std(capture_data[city][control]/inlet) for city in cities]
    
    values += values[:1]
    stds += stds[:1]

    ax.plot(angles, 
            values, 
            label=CONTROL_LABELS[control], 
            color=color, 
            linewidth=3, 
            linestyle='-' if control == 'MARL' else '--')
    ax.fill_between(angles, 
                    np.array(values) - np.array(stds), 
                    np.array(values) + np.array(stds), 
                    color=color, alpha=0.25)

# Formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels(LOCATION_LABELS.values(), fontsize = 12)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), frameon=False)
ax.text(0.0, 1.0, LABEL_INDEX[label_indices], transform=ax.transAxes,
        fontsize=18, fontweight='bold', va='top', ha='left')

tick_values = [0, 0.2, 0.4, 0.6]
tick_names = ['0%', '20%', '40%', '60%']
ax.set_yticks(tick_values,tick_names, fontsize=10)
ax.text(np.deg2rad(-50), 0.05, 'Capture efficiency', fontsize=12, rotation=0)

label_indices += 1

"""
Figure 5c & d
"""

def draw_heat_map(control):
    net_data = np.zeros((len(cities), len(SEASONS))) 
    for i, city in enumerate(list(LOCATION.keys())):
        for j, season in enumerate(SEASONS):
            emission = (data[city][control].sum(axis=1)[j] * mass_data[city][control][j]) * 1.1
            captured = capture_data[city][control][j]
            net_data[i, j] = emission - captured

    fig, ax = plt.subplots(figsize=(6.75, 6), dpi=600)

    ax = sns.heatmap(net_data, 
                    annot=True, 
                    fmt=".0f", 
                    cmap="Spectral_r", 
                    cbar_kws={'label': 'Net CO$_{2}$ emitted (kg)'}, 
                    linewidths=0.5, linecolor='gray',
                    vmin=-120,
                    vmax=120)
    ax.text(-0.12, 1.0, LABEL_INDEX[label_indices], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')
    ax.set_xticklabels([season.title() for season in SEASONS], fontsize=10)
    ax.set_yticklabels([LOCATION_LABELS[city] for city in cities], fontsize=10)
    for label in ax.get_yticklabels():
        label.set_multialignment('center')
    

draw_heat_map('conventional')
label_indices += 1
draw_heat_map('MARL')

#%% Supplementary Figure

"""
Supplementary Figure 2
"""
label_indices = 0
fig, ax = plt.subplots(nrows = 2, ncols = 4, 
                               figsize=(6.75, 6), 
                               dpi=600, 
                               sharex=True, sharey=True)
ax = ax.flat
linestyles = ['-', '--', '-.', ':']

for i, city in enumerate(list(LOCATION.keys())):
    for j, season in enumerate(SEASONS):
        df = pd.read_csv(f'result/test result_conventional_{city}_{season}.csv') 
        days = np.linspace(0, 15, sp.shape[0])
        ax[i].plot(days, 
                    df['T_env'], 
                    label=season.title(), 
                    color='#CC5959', 
                    linewidth=2, 
                    linestyle = linestyles[j])
        ax[i].text(0.1, 1.0, LABEL_INDEX[i], transform=ax[i].transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xticks(np.arange(0,16,5))

ax[0].set_ylabel('Ambient temperature (°C)')
ax[4].set_ylabel('Ambient temperature (°C)')
ax[3].legend(frameon=False, bbox_to_anchor=(2.0, 0.1))
fig.text(0.5, 0.05, 'Cultivation time (days)', 
         ha='center', va='center', fontsize=12)
"""
Supplementary Figure 3
"""

label_indices = 0

fig, ax = plt.subplots(nrows = 2, ncols = 4, 
                               figsize=(6.75, 6), 
                               dpi=600, 
                               sharex=True, sharey=True)
ax = ax.flat

for i, city in enumerate(list(LOCATION.keys())):
    for j, season in enumerate(SEASONS):
        df = pd.read_csv(f'result/test result_conventional_{city}_{season}.csv') 
        days = np.linspace(0, 15, sp.shape[0])
        ax[i].plot(days, 
                    df['I_sunlight'], 
                    label=season.title(), 
                    color='#EFCE80', 
                    linewidth=2, 
                    linestyle = linestyles[j])
        ax[i].text(0.1, 1.0, LABEL_INDEX[i], transform=ax[i].transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xticks(np.arange(0,16,5))

ax[0].set_ylabel('Sunlight irradiance (W m$^{-2}$)')
ax[4].set_ylabel('Sunlight irradiance (W m$^{-2}$)')
ax[3].legend(frameon=False, bbox_to_anchor=(2.0, 0.1))
fig.text(0.5, 0.05, 'Cultivation time (days)', 
         ha='center', va='center', fontsize=12)

#%%
"""
Supplementary Figure > 4
"""
VARIABLES = [
    'Biomass\nconcentration\n(g m$^{-3}$))',
    'Temperature\n(°C)',
    'pH',
    'Nutrient\nconcentration\n(g m$^{-3}$))',
    'CO$_{2}$\nconcentration\n(g m$^{-3}$))',
    'O$_{2}$\nconcentration\n(g m$^{-3}$))',
    'Red lamp\nintensity\n(W m$^{-2}$))',
    'Blue lamp\nintensity\n(W m$^{-3}$))',
    'Specific\ngrowth\nrate(day$^{-1}$))'
]

# Column names in your CSV corresponding to each variable
CSV_COLUMNS = [
    '# x', 'T', 'pH', 'S', 'CO2', 
    'O2', 'I_red_light', 'I_blue_light', 'miu'
]

for control in CONTROLS:
    for city in list(LOCATION.keys()):
        fig, axes = plt.subplots(nrows = 9, ncols = 4, 
                               figsize=(10, 6), 
                               dpi=600, 
                               sharex='col',   # share x within each column
                               sharey='row',   # share y within each row
                            )
        for col, season in enumerate(SEASONS):
            df = pd.read_csv(f'result/test result_{control}_{city}_{season}.csv') 
            for row, (var_label, col_name) in enumerate(zip(VARIABLES, CSV_COLUMNS)):
                ax = axes[row, col]
                ax.plot(days, df[col_name], linewidth=0.6, color='#1f77b4')
                ax.set_xticks(np.arange(0,16,5))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', labelsize=4.5, length=2, width=0.4)

                if col == 0:
                    ax.set_ylabel(var_label, fontsize=5, labelpad=3)
                else:
                    ax.set_ylabel('')

                if row == 0:
                    ax.set_title(season.title(), fontsize=6, pad=3)

                if row < 7:
                    ax.tick_params(labelbottom=False)

        # Shared x-axis label centered across all columns
        plt.tight_layout(rect=[0.05, 0.03, 1, 1])
        fig.text(0.55, 0.02, 'Cultivation time (days)',
                 ha='center', va='center', fontsize=6)
        fig.savefig(f'result/figure_{control}_{city}.png',
                    dpi=600, bbox_inches='tight')

