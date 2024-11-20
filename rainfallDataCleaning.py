import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# INPUT DATA
df_weather = pd.read_csv('01_initialData\\weather.csv', delimiter=';')
df_rainfallhistoricReport = pd.read_csv('rainfallhistoric.csv', delimiter=',')

df_weather = df_weather.dropna(axis=1, how='all') # delete completely null columns
df_weather = df_weather.dropna(axis=0, how='all') # delete completely null lines
df_weather.columns = ['date', 't_mean', 't_min', 't_max','rainfall'] # rename colums

# data type
df_weather['date'] = pd.to_datetime(df_weather['date'], format='%d/%m/%Y') # change date into datetime format
for col in df_weather.columns[1:]:
    df_weather[col] = df_weather[col].str.replace(',', '.').astype(float) # change object format to float format

df_weather['year'] = df_weather['date'].dt.year
df_weather['month'] = df_weather['date'].dt.month
df_weather['day'] = df_weather['date'].dt.day
df_weather['dayOfYear'] = pd.to_datetime(df_weather[['year', 'month', 'day']], errors='coerce').dt.dayofyear

# INTEGRATION OF THE UMBRIA RAINFALL HISTORIC DATA
"""
To correct potentiel outlier values, we are integrated data with the historic rainfall from sensor 3 to 5 km from our parcel, provided by the Umbria region.
(https://dati.regione.umbria.it/dataset/sir_precipitazioni_storico)

"""
colDrop = ['ID_TIPOLOGIA_SENSORE', 'ID_SENSORE_DETTAGLIO', 'STRUMENTO', 'TIPO_STRUMENTO', 'UNITA_MISURA', 'ID_STAZIONE'] # delete useless columns
for col in df_rainfallhistoricReport.columns:
    if col in colDrop:
        df_rainfallhistoricReport = df_rainfallhistoricReport.drop(col, axis=1)
        
df_rainfallhistoricReport.columns = ['station_name', 'municipality', 'latitude', 'longitude','year', 'month', 'day', 'rainfall'] # change the name of columns
df_rainfallhistoricReport = df_rainfallhistoricReport[df_rainfallhistoricReport['year'] >= 2009] # select only data from 2009
df_rainfallhistoricReport = df_rainfallhistoricReport[df_rainfallhistoricReport['latitude'] == 42.94416] # select only data from the closest station
df_rainfallhistoricReport = df_rainfallhistoricReport[df_rainfallhistoricReport['longitude'] == 12.63886] 
maxRainfall = df_rainfallhistoricReport['rainfall'].max() # max rainfall value

# plot the rainfall per day for each year
"""
plt.figure(figsize=(14, 8))
for year in df_weather['year'].unique():
    yearly_data = df_weather[df_weather['year'] == year]
    plt.plot(yearly_data['dayOfYear'], yearly_data['rainfall'], label=year)
    
plt.title('Rainfall per day for each year', fontsize=16)
plt.xlabel('Day of the year', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.legend(title='year', fontsize=10, loc='upper right')
plt.ylim(0, maxRainfall+10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(True)
plt.show()
"""

# plot the frequency distribution for each rainfall value
"""
rainfall_freq = df_weather['rainfall'].value_counts().sort_index()
plt.figure(figsize=(14, 8))
plt.plot(rainfall_freq.index, rainfall_freq.values, linestyle='-', markersize=6, linewidth=2)
plt.title('rainfall value frequencies', fontsize=16)
plt.xlabel('Rainfall (mm)', fontsize=12)
plt.xlim(0, maxRainfall)
plt.ylim(0, 250)
plt.grid(True)
plt.show()
"""

# OUTLIERS IDENTIFICATION
"""
The dataset follow a long tail distribution. the relevant techniques for outliers detection have to be adapted.
"""
df_weather = pd.merge(df_weather, # merge the weather dataframe with the historic report dataframe
                      df_rainfallhistoricReport[['year', 'month', 'day', 'rainfall']], 
                      on=['year', 'month', 'day'], 
                      how='left', 
                      suffixes=('_obs', '_histo'))

df_weather['rainfall_obs'] = df_weather['rainfall_obs'].fillna(df_weather['rainfall_histo']) # Replace NaN by values from the report

data = df_weather[df_weather['rainfall_obs'] != 0] # do not take into account 0 values for the following operations

meanDistance = round(np.abs(data['rainfall_histo'] - data['rainfall_obs']).mean(), 3) # compute the mean distance between rainfall_obs and rainfall_histo

# Robust Zscore as a function of median and median
# median absolute deviation (MAD) defined as z-score = |x – median(x)| / mad(x)
median = np.median(data['rainfall_obs'])
mad = np.median(np.abs(data['rainfall_obs'] - median))
mad = max(mad, 1e-9)  # if MAD is 0, corrected with 1e-9 to avoid division by 0
modified_z_scores = (data['rainfall_obs'] - median) / mad
z_outliers = data[np.abs(modified_z_scores) > 30]

# percentiles
upper_threshold = np.percentile(data['rainfall_obs'], 99)
p_outliers = data[data['rainfall_obs'] > upper_threshold]

# combine p_outliers and z_outliers and delete duplicate
combined_outliers = pd.concat([z_outliers, p_outliers]).drop_duplicates(subset=['year', 'month', 'day'])

# compare rainfall_obs and rainfall_histo taking into acount the average distance between the two
df_weather['is_outlier'] = False
for idx in combined_outliers.index:
    obs_value = df_weather.loc[idx, 'rainfall_obs']
    histo_value = df_weather.loc[idx, 'rainfall_histo']
    
    if obs_value > histo_value + meanDistance:
        df_weather.loc[idx, 'is_outlier'] = True

outliers = df_weather[df_weather['is_outlier']]
"""
According to this outlier detection, there are 21 outliers. But exploring these outliers, we can see that some of them are false positive
Those techniques are styles not really relevant because the distribution is not gaussian

The difference between rainfall_obs and rainfall_histo follows a normal distribution which can be easily studied.
"""
# distance outlier identification
df_weather['rainfall_difference'] = df_weather['rainfall_histo'] - df_weather['rainfall_obs'] # compute the distance
df_weather = df_weather[np.isfinite(df_weather['rainfall_difference'])] # eliminate infinite values
meanDistance = round(df_weather['rainfall_difference'].mean(), 3)

# density computation with gaussian_kde
kde = gaussian_kde(df_weather['rainfall_difference'])
x = np.linspace(min(df_weather['rainfall_difference']), max(df_weather['rainfall_difference']), 1000)
y = kde(x)

# plot the distance densities
"""
plt.figure(figsize=(10, 6))
plt.plot(x, y, color="skyblue", lw=2)  # Tracer la courbe
plt.fill_between(x, y, color="skyblue", alpha=0.5)  # Remplir sous la courbe
plt.title('Distribution des différences absolues (rainfall_histo - rainfall_obs)')
plt.xlabel('Différence absolue (mm)')
plt.xlim(-20, 20)
plt.ylabel('Densité')
plt.grid(True)
plt.show()
"""

# z-score 
mean = df_weather['rainfall_difference'].mean()
std = df_weather['rainfall_difference'].std()
z_score = np.absolute(df_weather['rainfall_difference'] - mean) / std
threshold = 2
distance_outliers = df_weather[abs(z_score) > threshold]
print(distance_outliers)

df_weather = df_weather[['year', 'month', 'day', 'dayOfYear', 'rainfall_obs']].rename(columns={'rainfall_obs': 'rainfall'}) # delate useless columns

# export the outliers based on the distance analysis in a csv file
with open('outliers.csv', 'w', newline='') as f:
    distance_outliers.to_csv(f, index=True)