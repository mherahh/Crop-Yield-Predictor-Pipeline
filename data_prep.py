import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestRegressor

#%% 1. Data loading
path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")
df = pd.read_csv(f"{path}/yield_df.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(df.head())

#%% 2. Data Cleaning
print(df.isnull().sum())
print(df.duplicated().sum())
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['unnamed:_0'],errors='ignore')



#%% 3. Exploratory Data analysis

print(df.describe()) #hg/ha_yield ranges from 50 - 501412, mean is 77053

print(df['area'].nunique())

print(df['year'].unique()) # Data is from 1990 - 2013, 2003 is missing

print(df['item'].value_counts()) # 10 crops

# 3.1  Yield over the years by each crop

yearly = df.groupby(["year", "item"])["hg/ha_yield"].mean().reset_index()

plt.figure(figsize=(13, 6))
sns.lineplot(data=yearly, x="year", y="hg/ha_yield", hue="item")
plt.title("Average yield over time, by crop")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 3.2 Each crop highest yield producer country

crop_list = df['item'].unique()

for crop in crop_list:
    crop_df = df[df['item'] == crop]
    counts = crop_df['area'].value_counts()
    reliable = counts[counts >= 5].index  # 5+ records to qualify
    crop_df = crop_df[crop_df['area'].isin(reliable)]

    top10 = (crop_df.groupby('area')['hg/ha_yield']
            .mean().sort_values(ascending=False).head(10)
    )
    plt.figure(figsize=(10,5))
    plt.bar(top10.index, top10.values)

    plt.xticks(rotation=45)
    plt.title(f"Top 10 countries for {crop} yield")
    plt.xlabel('Country')
    plt.ylabel("Yield (hg/ha)")
    plt.tight_layout()
    plt.show()

# 3.3 Distribution of yield
sns.histplot(df['hg/ha_yield'], bins=30, kde=True)
plt.title("Distribution of yield")
plt.show() # heavily right skewed

# 3.4 Factor vs Yield Relationship
factors = ["avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes"]

for factor in factors:
    fig, ax = plt.subplots(figsize=(12, 6))   # full-size, one factor per figure

    for crop in sorted(df["item"].unique()):
        crop_df = df[df["item"] == crop]
        binned = (crop_df.groupby(pd.cut(crop_df[factor], bins=15))["hg/ha_yield"]
                         .median().dropna())
        binned.index = [iv.mid for iv in binned.index]
        ax.plot(binned.index, binned.values, marker="o",
                linewidth=2, markersize=4, label=crop)

    ax.set_yscale("log")                       # <-- the fix that un-crushes low crops
    clean = factor.replace("_", " ").title()
    ax.set_xlabel(clean)
    ax.set_ylabel("Median Yield (hg/ha, log scale)")
    ax.set_title(f"Yield vs {clean}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="Crop", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show() # No clear pattern is shown

# 3.5 Correlation Matrix
num_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes','avg_temp']
plt.figure(figsize=(10,5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation between numeric factors")
plt.tight_layout()
plt.show()

# 3.6 Yield distribution by crop
plt.figure(figsize=(12, 5))
order = df.groupby("item")["hg/ha_yield"].median().sort_values().index
sns.boxplot(data=df, x="item", y="hg/ha_yield", order=order)
plt.xticks(rotation=45, ha="right")
plt.title("Yield varies enormously by crop")
plt.tight_layout()
plt.show()

print(df['area'].value_counts(ascending=True))
sizes = df.groupby(['item', 'area']).size()
print("smallest crop-country groups:")
print(sizes.sort_values().head(15))
print("\nhow many have fewer than 5 records:", (sizes < 5).sum())


# 3.7 Outlier detection
def outlier_report(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    return {
        "Feature": col,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Total Outliers": len(outliers),
        "Outlier %": round(len(outliers) / len(df) * 100, 2)
        }

results= []
for col in num_cols:
    results.append(outlier_report(df, col))

print(pd.DataFrame(results))

#%% 4. Preprocessing

# 4.1 Data Split
from sklearn.model_selection import train_test_split

train_df = df[df['year'] <= 2001]
test_df = df[df['year'] >= 2002]

X_train = train_df.drop(columns=['hg/ha_yield'])
y_train = train_df['hg/ha_yield']

X_test = test_df.drop(columns=['hg/ha_yield'])
y_test = test_df['hg/ha_yield']

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train years:", sorted(train_df['year'].unique()))
print("Test years:", sorted(test_df['year'].unique()))

# 4.2 One hot encoding for categorical cols
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['area', 'item']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #sparse output?
encoder.fit(X_train[cat_cols])

train_encoded = encoder.transform(X_train[cat_cols])
test_encoded = encoder.transform(X_test[cat_cols])

train_encoded = pd.DataFrame(train_encoded,
                             columns=encoder.get_feature_names_out(cat_cols),
                             index=X_train.index)
test_encoded  = pd.DataFrame(test_encoded,
                             columns=encoder.get_feature_names_out(cat_cols),
                             index=X_test.index)

X_train = X_train.drop(columns=cat_cols).join(train_encoded) # drop og columns, join the binary ones
X_test  = X_test.drop(columns=cat_cols).join(test_encoded)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 4.3 Scaling

from sklearn.preprocessing import RobustScaler
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = RobustScaler()

num_cols = ['year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test_scaled[num_cols])
print(X_train_scaled[num_cols].head())

joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "prepared_data.pkl")