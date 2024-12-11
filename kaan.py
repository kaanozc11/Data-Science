# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:45:30 2024

@author: Kaan
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

vehicle = pd.read_csv("co2.csv")

vehicle.dtypes



vehicle = vehicle.fillna({
    "Cylinders": vehicle["Cylinders"].mode(),
    "Fuel Consumption City (L/100 km)": vehicle["Fuel Consumption City (L/100 km)"].mean(),
    "Fuel Consumption Hwy (L/100 km)" : vehicle["Fuel Consumption Hwy (L/100 km)"].median()
    
    
    
})


pd.set_option("display.max_columns", 20)
myCors = vehicle[["Engine Size(L)", "Cylinders", "CO2 Emissions(g/km)", "Fuel Consumption Comb (L/100 km)"]].corr()
myCors

scaler = MinMaxScaler()
vehicle[["Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"]] = scaler.fit_transform(vehicle[["Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"]])



new_df = vehicle.loc[vehicle["Cylinders"]==4]
new_df2 = vehicle.loc[vehicle["CO2 Emissions(g/km)"] > 0.25]


birlesim = vehicle.loc[
    (vehicle["Engine Size(L)"] > 1.6) & (vehicle["Fuel Consumption Comb (L/100 km)"] > 0.25)
]

