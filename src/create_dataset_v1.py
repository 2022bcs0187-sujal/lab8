import pandas as pd

# load dataset
df = pd.read_csv("data/housing.csv")

# take first 5000 rows
df_v1 = df.head(5000)

# overwrite file
df_v1.to_csv("data/housing.csv", index=False)

print("Dataset Version 1 created with 5000 rows")