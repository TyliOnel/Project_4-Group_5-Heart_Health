import pandas as pd

# Import CSV
df = pd.read_csv('Resources/Heart_Disease (1).csv')
print(df.head())

print(df.isnull().sum())

