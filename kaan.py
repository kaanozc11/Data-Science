# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:45:30 2024

@author: Kaan Özcan
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

vehicle = pd.read_csv("co2.csv")

vehicle.dtypes


# If we have more than one missing data columns/attriibutes, how we can input

vehicle = vehicle.fillna({
    "Cylinders": vehicle["Cylinders"].mode(),
    "Fuel Consumption City (L/100 km)": vehicle["Fuel Consumption City (L/100 km)"].mean(),
    "Fuel Consumption Hwy (L/100 km)" : vehicle["Fuel Consumption Hwy (L/100 km)"].median()
    
    
    
})
# Please find two or more numeric attributes considering your interest. Then
#find the Pearson Cor. matrix. Then interpret the results.

pd.set_option("display.max_columns", 20)
myCors = vehicle[["Engine Size(L)", "Cylinders", "CO2 Emissions(g/km)", "Fuel Consumption Comb (L/100 km)"]].corr()
myCors

# Use min-max data normalization for the numeric attributes in the dataset.
scaler = MinMaxScaler()
vehicle[["Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"]] = scaler.fit_transform(vehicle[["Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"]])


# Assign to a new dataset "chest-pain == 4" patients
new_df = vehicle.loc[vehicle["Cylinders"]==4]
new_df2 = vehicle.loc[vehicle["CO2 Emissions(g/km)"] > 0.25]

# Assign to a new dataset sex==1 and angina==0 and max-heart-rate > 120
#heart-disease =="absence"

birlesim = vehicle.loc[
    (vehicle["Engine Size(L)"] > 1.6) & (vehicle["Fuel Consumption Comb (L/100 km)"] > 0.25)
]


