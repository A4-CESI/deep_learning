import pandas as pd

df = pd.read_csv("../data/diabetes_binary_health_indicators_BRFSS2015.csv")

print(df.head()) ,
print(df.info())