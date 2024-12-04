import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


southpark = pd.read_csv("All-seasons.csv")
southpark.dtypes


southpark["Character"] = southpark["Character"].astype("category")
southpark["Episode"] = pd.to_numeric(southpark["Episode"], errors="coerce")

southpark["Episode"] = southpark["Episode"].astype("float") 

southpark["Season"] = pd.to_numeric(southpark["Season"], errors="coerce")

southpark["Season"] = southpark["Season"].astype("float") 

pd.set_option("display.max_column", 20)
myCorr = southpark[["Season", "Episode", "Character"]].corr()
myCorr


southpark = southpark.fillna({
    "Season": southpark["Season"].mode(),
    "Episode": southpark["Episode"].mode(),
    
})

print(southpark.info())
print(southpark)


southpark.loc[southpark["Episode"] == 10]
