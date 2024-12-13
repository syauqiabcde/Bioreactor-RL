import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import math
from scipy import constants

class growth_model():
    def __init__(self, x, irradiance, temperature, pH, DO, substrate, carbon):
        self.x = x          
        self.irradiance = irradiance
        self.T = temperature
        self.pH = pH
        self.DO = DO
        self.substrate = substrate
        self.carbon = carbon

        self.KI = 100.83        # in W/m2
        self.Ki = 111.5         # in W/m2
        self.KIi = 13.9136      # in W/m2
        self.KC = 0.0396        # in g/m3
        self.Kc = 7920          # in g/m3
        self.KS = 125           # in g/m3
        self.Ks = 200           # in g/m3
        self.KN = 5.314         # in g/m3
        self.KP = 0.028         # in g/m3
        self.TL = 15            # in deg C
        self.TU = 37            # in deg C
        self.Topt = 29          # in deg C
        self.DO_max = 47.9      # in g/m3
        self.pH_min = 8         
        self.pH_max = 10.5
        self.miu_max = 3.24     # in d-1
        self.x_star = 8000      # in g m-3

    def growth_carbon(self):
        fC = self.carbon/(self.carbon+self.KC)
        return fC

    def growth_light(self):
        fI = self.irradiance/(self.irradiance+self.KIi)
        return fI

    def growth_temperature(self):
        if self.T >= self.Topt:
            fT = np.exp(-2.3*((self.T-self.Topt)/(self.TU-self.Topt))**2)
        else:
            fT = np.exp(-2.3*((self.T-self.Topt)/(self.TL-self.Topt))**2)
        return fT

    def growth_O2(self):
        fO2 = 1- self.DO/self.DO_max
        return fO2
    
    def growth_pH(self):
        if self.pH_min <= self.pH <= self.pH_max:
            fpH = (self.pH-self.pH_max)/(self.pH_min-self.pH_max)*np.exp(1-((self.pH-self.pH_max)/(self.pH_min-self.pH)))
        else:
            fpH = 0
        return fpH
    
    def growth_substrate(self):
        fS = self.substrate/(self.KS+self.substrate+self.substrate**2/self.Ks)
        # fS = 0.15
        return fS

    def inhibitory(self):
        finh = 1 - (self.x / self.x_star)
        return finh
    
    def solve(self):
        fC = self.growth_carbon()
        fI = self.growth_light()
        fT = self.growth_temperature()
        fO2 = self.growth_O2()
        fpH = self.growth_pH()
        fS = self.growth_substrate()
        finh = self.inhibitory()

        miu = self.miu_max*fC*fI*fT*fO2*fpH*fS*finh
        return miu

class bioreactor():
    def __init__(self, 
                x, 
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
                Q_heat):

        # Constants
        self.m = 0.05               # substrate requirement for maintaining biomass daily activities
        self.Yxs =  0.6             # yield coefficient of mass of substrate consumed per mass of new cells produced
        self.tau = 0.289            # Cell absorption coefficient in m2/g
        self.g = constants.g        # gravity acceleration
        self.R = constants.physical_constants['molar gas constant'][0] #Ideal Gas Constant
        self.db = 0.001             # Bubble diameter in m
        self.PE = 0.03              # photosynthesis efficiency
        self.absorp = 0.03          # Absorptivity of bioreactor material i.e. glass
        self.glass_k = 0.85         # thermal conductivity of bioreactor material in W/m.K
        self.thick = 3e-3           # bioreactor thickness in m
        self.pH_acid = 3            # HCl injection pH
        self.pH_base = 11           # NaOH injection pH
        self.TL = 15                # in deg C
        self.TU = 37                # in deg C
        self.Topt = 29              # in deg C
        self.pH_min = 8             #  
        self.pH_max = 10.5          
        self.transmittance = 0.9    # glass transmittance

        # Molecular weight
        self.M_H2O = 18.015         # in g/mol       
        self.M_O2 = 31.998          # in g/mol
        self.M_CO2 = 44.01          # in g/mol
        self.M_Air = 28.96          # in g/mol

        # Photobioreactor specification
        self.level = level
        self.ID = d
        self.OD = self.ID + 2*self.thick
        self.A = np.pi*self.ID**2/4
        self.z = z
        self.V = self.A*self.z                   # Design volume
        self.V_act = self.A*(self.z*self.level)  # actual liquid volume
        
        # initial condition
        self.x = x                 # biomass concentration (g/m3)
        self.S = S                  # Substrate concentration (g/m3) 
        self.I_sun = I_sunlight     # Sunlight Irradiance (W/m2)
        self.I_red = I_red_light      # Irradiance of red lamp (W/m2)   
        self.I_blue = I_blue_light    # irradicnce of blue lamp (W/m2)
        self.I0 = self.I_sun + self.I_red + self.I_blue  # total irradiance hitting surface of biorea ctor
        self.Iz = 0                 # Initializing the light intensity hitting the biomass
        self.T = T                  # Temperature of the cell culture (deg C)
        self.CO2 = CO2              # Dissolved CO2 concentration (g/m3)
        self.O2 = O2                # Dissolved O2 concentration (g/m3)
        self.Q_heat = Q_heat        # heat supplied from heater (kW)
        self.level = level          # level of the liquid inside photobioreactor
        self.Fs = Fs                # Flow of substrate in (kg/s)
        self.FCO2_in = FCO2         # Volumetric flow rate of CO2 (m3/s)
        self.Fair_in = Fair         # Volumetric flow rate of air (m3/s)
        self.FH2O_in = FH2O         # Volumetric flow rate of water (m3/s)
        self.Facid_in = Facid       # Volumetric flow rate of acid (m3/s)
        self.Fbase_in = Fbase       # Volumetric flow rate of base (m3/s)
        self.pH = pH                # pH of the bioreactor
        self.H = 10**(-self.pH)
        self.OH = 10**(-(14-self.pH))
        self.lambert_beer()

        self.x_mass = self.x * self.V_act  # biomass dry weight (g)
        self.S_mass = self.S * self.V_act   # substrate dry weight (g)
        self.CO2_mass = self.CO2 * self.V_act   # Dissolved CO2 mass (g)
        self.O2_mass = self.O2 * self.V_act   # Dissolved O2 mass (g)

        # Environment condition
        self.T_env = T_env          # Environment temperature
        self.RH = RH                # Environment humidity
        self.wind_speed = wind_speed

        # Temporal increment
        self.dt_s = 1                                                # temporal increment in s
        self.dt = self.dt_s / (24 * 60 * 60)                         # temporal increment in day
        self.dt_h = self.dt_s / (60 * 60)

    def Henry(self, component):                  # From NIST
        kH = {'CO2': 0.035,
             'O2': 0.0013}
        
        kH_dT = {'CO2': 2400,
             'O2': 1500}
        
        
        H = kH[component]*np.exp(kH_dT[component] * (1/(self.T+273.15) - 1/298.15))
        return H                                    # in mol/(L.atm)

    def Diffusivity(self, component):      # in m2/s https://pubs.acs.org/doi/10.1021/acs.jced.3c00778
        D0 = {'CO2': 19.798e-9,            # Diffusivity of CO2 and O2  water
              'O2': 2.033e-8,              # Diffusivity of water to air
              'Air': 4.13e-5}              
        
        m = {'CO2': 2.01489,
             'O2': 1.8944929,
             'Air': 0.403231}

        D = D0[component] * ((self.T+273.15)/227-1)**m[component]
        return D                           # in m2/s
    
    def viscosity(self, component):
        A = {'H2O':0.0002,
             'CO2': 0,
             'O2': 0,
             'Air': -4.09812e-8}
        
        B = {'H2O':- 0.0269,
             'CO2': 5.4983e-5,
             'O2': 5.5337e-5,
             'Air': 4.98553e-5}
        
        C = {'H2O':1.4879,
             'CO2': 1.3249e-2,
             'O2': 1.9557e-2,
             'Air': 1.76061e-2}
        
        miu = A[component]*self.T**2 + B[component]*self.T + C[component]     # in cP
        return miu/1000                                                       # in Pa.s
    
    def density(self, component):
        A = {'H2O':-0.0035,
             'CO2': 0,
             'O2': 1.25527e-5,
             'Air': 1.07321e-5,
             'Acid': -0.0035,
             'Base': 0}
        
        B = {'H2O':-0.0949,
             'CO2': -5.5381e-3,
             'O2': -5.01445e-3,
             'Air': -4.49998e-3,
             'Acid': -0.0949,
             'Base': -0.7695}
        
        C = {'H2O':1001.3,
             'CO2': 1.9507,
             'O2': 1.42559,
             'Air': 1.28621,
             'Acid': 1001.3,
             'Base': 1026.3}
        
        rho = A[component]*self.T**2 + B[component]*self.T + C[component]
        return rho                                  # in kg/m3
    
    def heat_capacity(self, component):            # heat capacity in kJ/kg C
        A = {'H2O': 1.60425e-5,
            'Air': 3.1325e-8, 
            'CO2': 0,
            'Acid': 1.60425e-5,
            'Base': 1.60425e-5}

        B = {'H2O': - 8.89887e-4,
            'Air': 2.0440e-4,
            'CO2': 6.77361e-4,
            'Acid': - 8.89887e-4,
            'Base':- 8.89887e-4}

        C = {'H2O': 4.32557,
            'Air': 1.007,
            'CO2': 8.57773e-1,
            'Acid': 4.32557,
            'Base':4.32557}

        Cp =  A[component] * self.T**2 + B[component] * self.T + C[component]    # Cp in kJ/kg.C
        return Cp*1000                                                           # Cp in J/kg.C

    def thermal_conductivity(self, component):              # in W/m.K
        A = {'H2O': -7.3635e-6,
            'Air': -3.4022e-8}

        B = {'H2O': 1.8466e-3,
            'Air': 7.3490e-5}

        C = {'H2O': 5.6930e-1,
            'Air': 2.4099e-2}

        k = A[component] * self.T**2 + B[component] * self.T + C[component] 
        return k                                # in W/m.K

    def update_inventory(self):                                 # update inventory in case of inlet flow
        self.V_act += (self.FH2O_in + self.Facid_in + self.Fbase_in - self.NH2O_evap()/self.density('H2O')) * self.dt_s
        self.level = min(max(self.V_act / self.V, 0), 1)
        self.V_act = self.V * self.level                                        # recalculating in case the previous level is not between 0 - 100%
    
    def latent_heat(self):
        return (-2.4884*self.T + 2527) * 1000    # heat of evaporation of water in J/kg
    
    def lambert_beer(self):             # in W/m2
        self.Iz = (self.I_sun * 0.3456 + self.I_red * 0.819 + self.I_blue * 0.668)* self.transmittance * (np.exp(-self.tau*self.OD/2*self.x))         # light intensity at d/2 will be used for further calculation assumed as the average intensity that hit the algae

    def energy_balance(self):
        R_cond = np.log(self.OD/self.ID) / (2 * np.pi * self.glass_k * self.z)

        beta = -4.81220e-2 * self.T**2 + 1.22987e1 * self.T - 3.32344e1
        Gr = self.g * beta * (np.abs(self.T_env - self.T) + 273.15) * self.z**3 / ((self.viscosity('H2O') * self.density('H2O'))**2)
        Pr = self.viscosity('H2O') * self.heat_capacity('H2O') / self.thermal_conductivity('H2O')
        Ra = Gr * Pr
        Nu = (0.825 + 0.387*Ra**(1/6)/(1+(0.492/Pr)**(9/16))**(8/27))**2
        h = Nu * self.thermal_conductivity('H2O') / self.z
        R_conv_int = 1/(h * (self.ID * np.pi * self.z))

        Re = self.density('Air') * self.OD * self.wind_speed / self.viscosity('Air')
        Pr = self.viscosity('Air') * self.heat_capacity('Air') / self.thermal_conductivity('Air')
        Nu = 0.023 * Re**0.8 * Pr**0.3
        h = Nu * self.thermal_conductivity('Air') / self.OD
        R_conv_out = 1/(h * (self.OD * np.pi * self.z))

        U = 1/(R_cond + R_conv_int + R_conv_out)                                # deg C/w

        Q_air = self.Fair_in * self.density('Air') * self.heat_capacity('Air') * (self.T_env - self.T)          # in W
        Q_CO2 = self.FCO2_in * self.density('CO2') * self.heat_capacity('CO2') * (self.T_env - self.T)       # in W
        Q_H2O = self.FH2O_in * self.density('H2O') * self.heat_capacity('H2O') * (self.T_env - self.T)
        Q_acid = self.Facid_in * self.density('Acid') * self.heat_capacity('Acid') * (self.T_env- self.T)
        Q_base = self.Fbase_in * self.density('Base') * self.heat_capacity('Base') * (self.T_env - self.T)

        Q_light = (1- self.PE) * self.I0 * (self.OD * np.pi * self.z) * self.absorp
        Q_env = U * (self.T_env - self.T)
        Q_evap = self.NH2O_evap() * self.latent_heat() 
        Q_mass = Q_air + Q_CO2 + Q_H2O + Q_acid + Q_base   
        Q_tot = Q_light + Q_env + (self.Q_heat * 1000) - Q_evap + Q_mass                            # in W
        self.T += (1/(self.density('H2O') * self.V_act * self.heat_capacity('H2O')) * Q_tot) * self.dt_s # in deg C

    def Ka(self):                           # H2CO3 dissociation constant
        return -8.729424e-11*self.T**2 + 9.56905e-9 * self.T + 2.59902e-7   # 10.1021/ja01250a059
     
    def pH_balance(self):
        K_w = 1e-14
        H_acid = 10**(-self.pH_acid)                                                    # in mol/L
        OH_base = 10**(-1*(14-self.pH_base))                                            # in mol/L

        conc_H_in = H_acid * self.Facid_in * 1000                                       # in mol/s
        conc_OH_in = OH_base * self.Fbase_in * 1000                                     # in mol/s
        conc_CO2_in = (self.NCO2() * 1000 - self.rCO2() / (24 * 60 *60)) / self.M_CO2   # in mol/s

        conc_H_CO2 = conc_CO2_in * self.Ka() / (self.Ka() + self.H)                     # in mol/s

        tot_H =  self.H * self.V_act * 1000 + (conc_H_in + conc_H_CO2) * self.dt_s      # in mol
        tot_OH = self.OH * self.V_act * 1000 + conc_OH_in * self.dt_s                   # in mol

        if tot_H > tot_OH:
            excess_H = tot_H - tot_OH
            self.H = excess_H / (self.V_act * 1000)
        elif tot_OH > tot_H:
            excess_OH = tot_OH - tot_H
            self.H = K_w / (excess_OH / (self.V_act * 1000))
        elif tot_H == tot_OH:
            self.H = 1e-7                               # if tot_H = tot_OH then the pH should be neutral i.e. [H+] = 1e-7

        self.pH = min(max(-np.log10(self.H), 0), 14)    # constraint to make the pH always between 0 and 14

    def bubble_velocity(self):                          # bubble velocity using stoke's law
        Ftotal = self.Fair_in + self.FCO2_in
        XCO2 = (self.FCO2_in + 400/1e6*self.Fair_in) / (Ftotal)
        XAir = 1-XCO2

        density = self.density('CO2') * XCO2 + self.density('Air') * XAir
        u = self.db**2*(self.density('H2O')-density)*self.g/(18*self.viscosity('H2O')) 
        return u
    
    def NO2(self):
        H = self.Henry('O2')                      # in mol/(L.atm)
        Ftotal = self.FCO2_in + self.Fair_in
        P_O2 = self.Fair_in * 0.21 / Ftotal     # in atm
        C_star = H * P_O2 * 1000 * self.M_O2    # in g/m3
        D = self.Diffusivity('O2') 

        u = self.bubble_velocity()

        kL = 2*np.sqrt(D*u/(np.pi*self.db))
        NO2 = (kL * (C_star - self.O2)) * self.A / 1000
        return NO2                                # in kg/s
    
    def NCO2(self):
        H = self.Henry('CO2')                       # in mol/(L.atm)
        Ftotal = self.FCO2_in + self.Fair_in
        P_CO2 = (self.FCO2_in + 400/1e6*self.Fair_in) / (Ftotal)   # in atm,  400/1e6 is the concentration of CO2 in air (400 ppm)
        C_star = H * P_CO2 * 1000 * self.M_CO2      # in g/m3
        D = self.Diffusivity('CO2')

        u = self.bubble_velocity()  

        kL = 2*np.sqrt(D*u/(np.pi*self.db))
        NCO2 = (kL * (C_star - self.CO2)) * self.A / 1000
        return NCO2                                 # in kg/s
    
    def rO2p(self):
        miuO2 = 2.7         # in gO2 / (g biomass * d)
        KL = 19.53          # in W/m2
        KCO2 = 0.005        # in g/m3
        KPR = 0.16          # in g/m3
        pH_opt = (self.pH_max + self.pH_min)/2

        a = (self.Topt - self.TL) * (self.T - self.Topt)
        b = (self.Topt - self.TU) * (self.Topt + self.TL - 2*self.T)
        c = (pH_opt - self.pH_min) * (self.pH - pH_opt)
        d = (pH_opt - self.pH_max) * (pH_opt + self.pH_min - 2*self.pH)

        light_function = self.Iz/(KL + self.Iz)
        CO2_function = self.CO2/(self.CO2+KCO2*(1+self.O2/KPR))
        temperature_function = ((self.T - self.TU) * (self.T - self.TL)**2) / ((self.Topt - self.TL) * (a - b))
        pH_function = ((self.pH - self.pH_max) * (self.pH - self.pH_min)**2) / ((pH_opt - self.pH_min) * (c - d))
        rO2p = miuO2 * self.x_mass * light_function * CO2_function * max(temperature_function,0) * max(pH_function,0) 
        return rO2p                         # in gO2/day

    def rCO2(self):
        rO2p = self.rO2p()
        return rO2p*self.M_CO2/self.M_O2    # in gCO2/day

    def rH2O(self):
        rO2p = self.rO2p()
        return rO2p*self.M_H2O/self.M_O2    # in gH2O/day

    def rO2(self):
        kresp = 0.266           # in d-1
        KMO2 = 1.81e-1          # in g/m3
        kinh = 0.026229         # in d-1
        zeta = 1.175e-1         # in g/m3

        rO2p = self.rO2p() / self.V_act                     # in g/(m3.day)
        rO2r = kresp * self.O2/(KMO2 + self.O2) * self.x    # in g/(m3.day)
        rO2inh = kinh * np.exp(zeta * self.O2) * self.x     # in g/(m3.day)
        rO2 = rO2p - rO2r - rO2inh
        return rO2                                          # in g/(m3.day)
    
    def NH2O_evap(self):
        Psat = (10**(4.6543-1435.264/(self.T+273.15-64.848)))*1e5   # in Pa
        C_star = (Psat/((self.T + 273.15) * self.R))                # in mol/m3
        C = self.RH * C_star                                        # in mol/m3

        Ftotal = self.FCO2_in + self.Fair_in                        # in m3/s
        v = self.bubble_velocity()

        Re = v * self.ID * self.density('Air') / self.viscosity('Air')
        Sc = self.viscosity('Air') / (self.Diffusivity('Air') * self.density('Air'))
        Sh = 0.145*Re**0.69*Sc**0.87
        k = Sh * self.Diffusivity('Air') / self.ID                  # in m/s
        NH2O = k*(C_star - C) * self.A  * self.M_H2O /1000          # in kg/s
        return NH2O                                                 # in kg/s

    def solve(self):
        self.update_inventory()
        self.x = self.x_mass / self.V_act                           # updating the biomass concentration due to dillution of additional water
        self.S = self.S_mass / self.V_act
        self.O2 = self.O2_mass / self.V_act
        self.CO2 = self.CO2_mass / self.V_act
        self.S = self.S_mass / self.V_act
        self.lambert_beer()                                         # updating the light intensity hitting the biomass
        self.energy_balance()                                       # updating the temperature
        self.pH_balance()                                           # updating the pH
        self.O2_mass += (self.NO2() * 1000 * self.dt_s)             # updating the dissolved O2 mass after the flow air in
        self.CO2_mass += (self.NCO2() * 1000 * self.dt_s)           # updating the dissolved CO2 mass after the flow CO2 in    
        self.O2 = self.O2_mass / self.V_act
        self.CO2 = self.CO2_mass / self.V_act
        self.S += self.Fs/self.V_act * self.dt_s                    # updating the substrate concentration after substrate flow in
        
        growth = growth_model(x= self.x,
                          irradiance=self.Iz,
                          temperature=self.T,
                          pH = self.pH,
                          DO = self.O2,
                          substrate = self.S,
                          carbon = self.CO2)
        miu = growth.solve()

        self.S += (-self.m*self.x) * self.dt_h - (self.Yxs*miu*self.x) * self.dt
        self.x += (self.x*miu) * self.dt
        self.x_mass = self.x * self.V_act
        self.V_act += (-self.rH2O()/1000 * self.dt)/self.density('H2O')
        self.level = min(max(self.V_act / self.V, 0), 1)
        self.V_act = self.level * self.V
        self.x = self.x_mass / self.V_act
        self.O2 += (self.rO2()) * self.dt
        self.CO2_mass += (- self.rCO2()/1000) * self.dt
        self.CO2 = self.CO2_mass / self.V_act
        
        return max(self.x,0), max(self.S,0), self.level, self.T, max(self.CO2,0), max(self.O2,0), self.pH, miu
    

x = 10 
S = 10
I_sunlight = 1000
I_red_light = 0
I_blue_light = 0
level = 0.7
Fs = 0
FCO2 = 10
Fair = 2
FH2O = 0
Facid = 0
Fbase = 0
T = 30
CO2 = 105
O2 = 10
pH = 8.5 
d = 0.5
z = 2
T_env = 30
RH = 0
Q_heat= 1
wind_speed = 0.1

for i in range(1):
    bio = bioreactor(x = x,  
                S = S,  
                I_sunlight = I_sunlight,
                I_red_light = I_red_light,
                I_blue_light = I_blue_light, 
                level = level,
                Fs = Fs,
                FCO2 = FCO2,
                Fair = Fair,
                FH2O= FH2O,
                Facid = Facid,
                Fbase = Fbase,
                T = T,
                CO2 = CO2,
                O2 = O2,
                pH = pH, 
                d = d, 
                z = z,
                T_env = T_env,
                RH = RH,
                wind_speed= wind_speed,
                Q_heat = Q_heat)
    
    x, S, level, T, CO2, O2, pH, miu = bio.solve()
    print('t: ', i)
    print('x: ', x)
    print('miu:', miu)
    print('S: ', S)
    print('level :', level)
    print('T :', T)
    print('CO2 :', CO2)
    print('O2 :', O2)
    print('pH :', pH)
    print('')