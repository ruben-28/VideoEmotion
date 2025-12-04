import os 
import csv
import pandas as pd

# Charger le CSV
df = pd.read_csv("C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions\\emotions_analysis.csv")

# Exporter en Excel
df.to_excel("C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions\\emotions_analysis.xlsx", index=False)