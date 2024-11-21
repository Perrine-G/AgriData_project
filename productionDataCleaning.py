import pandas as pd
import matplotlib.pyplot as plt
import locale
locale.setlocale(locale.LC_TIME, 'it_IT.utf8')

# INPUT DATA
df_production = pd.read_csv('01_initialData\\production.csv', delimiter=';')

df_production = df_production.dropna(axis=1, how='all') # delete completely null columns
df_production = df_production.dropna(axis=0, how='all') # delete completely null lines

df_production.columns = ['year', 'variety', 'parcel', 'ha','rowSpacing', 'plantSpacing', 'plantPerHa', 'picking', 'kg', 'yield'] # rename colums

df_production['ha'] = df_production['ha'].str.replace(',', '.').astype(float) # change object format to float format
df_production['rowSpacing'] = df_production['rowSpacing'].str.replace(',', '.').astype(float)
df_production['plantSpacing'] = df_production['plantSpacing'].str.replace(',', '.').astype(float)
df_production['plantPerHa'] = df_production['plantPerHa'].str.replace(' ', '').astype(int) # change object format to int format

df_production['full_date'] = df_production['picking'] + '-' + df_production['year'].astype(str) # concatenation picking date + year
df_production['pickingDate'] = pd.to_datetime(df_production['full_date'], format='%d-%b-%Y', errors='coerce') # change object format to date format
df_production.drop(columns=['picking', 'full_date'], inplace=True) # delete useless columns

# plot yield for each parcel for each year
df_parcel = df_production['parcel'].unique()

fig, axes = plt.subplots(3, 2, figsize=(12, 18))
fig.tight_layout(pad=5.0)
axes = axes.flatten()

for i, parcel in enumerate(df_parcel):
    ax = axes[i]
    df_parcel_data = df_production[df_production['parcel'] == parcel]
    
    ax.bar(df_parcel_data['year'], df_parcel_data['yield'], color='skyblue', edgecolor='black', zorder=3)
    ax.set_title(parcel)
    ax.set_xlabel("Année")
    ax.set_ylabel("Yield")
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
plt.show()

# plot yield per parcel per year
df_parcel = df_production['parcel'].unique()

plt.figure(figsize=(12, 8))

for parcel in df_parcel:
    df_parcel_data = df_production[df_production['parcel'] == parcel]
    plt.plot(df_parcel_data['year'], df_parcel_data['yield'], label=parcel, marker='o')

plt.title("Yield par parcel along the years")
plt.xlabel("year")
plt.ylabel("yield")
plt.legend(title="Parcels", loc="upper right")
plt.grid(True, linestyle='--', color='gray', alpha=0.3)
plt.show()


# plot the proportional size of each parcel
df_parcel_surface = df_production.drop_duplicates(subset='parcel')[['parcel', 'ha']]
plt.figure(figsize=(8, 8))
plt.pie(df_parcel_surface['ha'], labels=df_parcel_surface['parcel'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title("proportional size of the parcels")
plt.show()