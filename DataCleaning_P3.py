import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load your dataset from the CSV file
file_path = r'weather_game.csv'

# Read the CSV file with a different encoding (Latin-1)
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Handling Missing Values
data['Temperature'].fillna(data['Temperature'].mean(), inplace=True)

data['Outlook'] = data['Outlook'].astype('category').cat.codes
data['Wind'] = data['Wind'].map({'weak': 0, 'strong': 1})
data['Play'] = data['Play'].map({'no': 0, 'yes': 1})

# Save the cleaned data to a new CSV file
cleaned_file_path = r'cleaned_weather_data.csv'
data.to_csv(cleaned_file_path, index=False)

# Define numerical columns for normalization
numerical_columns = ['Outlook','Temperature', 'Humidity','Wind','Play']

# Min-Max Normalization
min_max_scaler = MinMaxScaler()
data[numerical_columns] = min_max_scaler.fit_transform(data[numerical_columns])

# Z-score Normalization
z_score_scaler = StandardScaler()
data[numerical_columns] = z_score_scaler.fit_transform(data[numerical_columns])

# Data Exploration after cleaning and normalization
print("\n\nAfter Cleaning and Normalization\n\n")
print("First 5 rows of the dataset:")
print(data.head())

# Summary statistics of the dataset
print("\nSummary statistics of the dataset after cleaning and normalization:")
print(data.describe())

# Missing values check after cleaning and normalization
missing_values = data.isnull().sum()
print("\nMissing values after cleaning and normalization:")
print(missing_values)
