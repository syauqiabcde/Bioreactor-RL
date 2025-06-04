import numpy as np
import pandas as pd
from tensorflow import keras
from model_bioreactor import bioreactor
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import itertools
import tensorflow as tf
from utils import ControlAgent
from datetime import datetime, timedelta

#########################
#     General config    #
#########################
simulation_time = 15        # days 
d = 0.6                     # diameter  
z = 5                       # height

CO2, pH, x, S, T, level, O2, FH2O = 10, 8.5, 50.0, 30.0, 25.0, 0.7, 5.0, 0.0
Fgas = 200/1000/60     # 200 L per minute total gas going in to m3/s
agent = ControlAgent()
day = 0

#########################
#   Get weather data    #
#########################
location = {
            'Depok': (-6.4072, 106.8158),   # Depok, Indonesia, represents tropical climate
            'Ulsan': (35.5392, 129.3119),   # Ulsan, S. Korea, represents sub-tropical climate
            'Riyadh': (23.333, 45.333) ,    # Riyadh, Saudi Arabia, represents arid and dry climate
            'La Paz': (-16.4955, -68.1336), # La Paz, Bolivia, reprents high-altitde climate
            'Oslo': (59.9133, 10.739)       # Oslo, Norway, represents low light, colde climate
            }      

season = {
    'winter': ['2024-01-01'],
    'spring': ['2024-04-01'],
    'summer': ['2024-07-01'],
    'fall': ['2024-10-01']
    }

for key in season:
    start_date = datetime.strptime(season[key][0], "%Y-%m-%d")
    end_date = start_date + timedelta(days=simulation_time)
    season[key].append(end_date.strftime("%Y-%m-%d"))

city = 'La Paz'
time = 'fall'

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": location[city][0],
	"longitude": location[city][1],
	"hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"],
    "start_date": season[time][0],
    "end_date": season[time][1]
}
responses = openmeteo.weather_api(url, params=params)

response = responses[0]
hourly = response.Hourly()
temperature = hourly.Variables(0).ValuesAsNumpy()
humidity = hourly.Variables(1).ValuesAsNumpy()
wind = hourly.Variables(2).ValuesAsNumpy()
irradiation = hourly.Variables(3).ValuesAsNumpy()

#########################
#       Test loop       #
#########################
control_mode = 'HMARL' # MARL, HMARL, or conventional

result = []

print('Start')

for steps in range(simulation_time * 24):
    T_env = temperature[steps]
    RH = humidity[steps]
    wind_speed = wind[steps]
    I_sunlight = irradiation[steps]
    obs = [I_sunlight, CO2, pH, x, S, T, T_env]

    actions = agent.take_action(obs, mode=control_mode)

    I_red_light, I_blue_light, CO2_ratio, Facid, Fbase, Fs, Q_heat = list(itertools.chain.from_iterable(actions))
    FCO2 = CO2_ratio * Fgas
    Fair = (1 - CO2_ratio) * Fgas

    bio = bioreactor(x, 
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
                    RH/100,
                    wind_speed,
                    Q_heat)
    
    x, S, level, T, CO2, O2, pH, miu = bio.solve()

    if steps % 24 == 0:
        day += 1
        print(f'Day {day}/{simulation_time}: Algae concentration {x:.2f} g/m3')

    result.append([x, 
                  S,  
                  I_sunlight,
                  I_red_light,
                  I_blue_light, 
                  level,
                  Fs,
                  CO2_ratio,
                  Facid,
                  Fbase,
                  T,
                  CO2,
                  O2,
                  pH, 
                  T_env,
                  RH,
                  wind_speed,
                  Q_heat,
                  miu[0],
                  miu[1],
                  miu[2],
                  miu[3],
                  miu[4],
                  miu[5]])

cols = 'x,S,I_sunlight,I_red_light,I_blue_light,level,Fs,CO2_ratio,Facid,Fbase,T,CO2,O2,pH,T_env,RH,wind_speed,Q_heat,miu,miu_light,miu_carbon,miu_pH,miu_S,miu_T'    
results = np.array(result)
np.savetxt(f'test result_{control_mode}_{city}_{time}.csv', results, header=cols, delimiter=',', fmt='%f')
print('Finished')
print(f'S: {S}')
    
