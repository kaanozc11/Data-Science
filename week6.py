# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:02:52 2024

@author: test2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


heart = pd.read_csv("heart.csv")
heart.dtypes

heart.describe()


heart["chest-pain"] = heart["chest-pain"].astype("category")


heart.iloc[0:5,0] #only numeric indexes required
heart.loc[0:5,"age"].values
heart.loc[0:5,["age","thal"]]



heart["serum-chol"] = heart["serum-chol"].fillna(heart["serum-chol"].mean())
print(heart.info())

#if we have more than one mising data columns/attributes, how we can input

heart.loc[264,"serum-chol"]
print(heart.info())


pd.set_option("display.max_column", 20)
myCorr = heart[["age","max-heart-rate","oldpeak","slope"]].corr()
myCorr

sns.boxplot(y="heart-disease",
            x = "age",
            data=heart,
            palette ="rainbow" )


#Please find two or more numeric attributes considering your interest. Then
#find the Pearson cor. matrix. Then interpret the result.8




plt.scatter(x="slope",y="oldpeak",data=heart)
plt.xlabel("slope")
plt.ylabel("oldpeak")

plt.scatter(x="major-vessels", y="thal", data=heart)

plt.scatter(x="rest-bp", y="serum-chol", data=heart)
plt.xlabel("Rest-BP")
plt.ylabel("SerumChol")

#chest pain 4 olanları verisetine ata

heart["chest-pain"] = heart["chest-pain"].astype("int64")

heart.iloc[heart["chest-pain"] == 4]


d2 = heart.loc(pd.,snull(heart["chest-pain"]))

#sex = 1, angina = 0, max-heart-rate > 120, heart-disease = "absence"
