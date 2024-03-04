import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_path=r"C:\Users\SMIT\Downloads\weather_game.csv"

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
s2 = pd.Series([5, 6, 7, 8, 9, 10], index=(range(6)))
df = pd.DataFrame(s1, columns=['s1'])
df['s2'] = s2

