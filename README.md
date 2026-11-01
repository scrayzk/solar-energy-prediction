#1. Imports
#seaborn draws the charts (matplotlib is installed with it and only used to show them). scikit-learn provides the models and #metrics. 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style(style="whitegrid")

from google.colab import files
uploaded = files.upload()

gen1 = pd.read_csv("Plant_1_Generation_Data.csv")
gen2 = pd.read_csv("Plant_2_Generation_Data.csv")
w1 = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
w2 = pd.read_csv("Plant_2_Weather_Sensor_Data.csv")

gen1["DATE_TIME"] = pd.to_datetime(gen1["DATE_TIME"], format="%d-%m-%Y %H:%M")
for df in (gen2, w1, w2):
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

gen1.head()
#w1.head()

sns.histplot(gen1["AC_POWER"])

gen = pd.concat([gen1, gen2], ignore_index=True)
weather = pd.concat([w1, w2], ignore_index=True)

print("Missing values in generation data:\n", gen.isna().sum(), "\n")
print("Missing values in weather data:\n", weather.isna().sum(), "\n")

# One row per plant per timestamp: average AC power across inverters
power = gen.groupby(["DATE_TIME", "PLANT_ID"], as_index=False)["AC_POWER"].mean()

# Merge with weather (inner join keeps only timestamps that exist in both)
weather = weather[["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
data = power.merge(weather, on=["DATE_TIME", "PLANT_ID"], how="inner")

print("Merged dataset shape:", data.shape)
data.head()

print("Plant-timestamps in generation data:", len(power))
print("Weather rows:", len(weather))
print("After merge: ", len(data))

sns.histplot(data["AC_POWER"])
plt.show()


data["PLANT"] = data["PLANT_ID"].map({4135001: 1, 4136001: 2})
data["HOUR"] = data["DATE_TIME"].dt.hour + data["DATE_TIME"].dt.minute / 60
data["TEMP_DIFF"] = data["MODULE_TEMPERATURE"] - data["AMBIENT_TEMPERATURE"]

data = data[data["IRRADIATION"] > 0].reset_index(drop=True)

features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TEMP_DIFF", "HOUR", "PLANT"]
target = "AC_POWER"

data[features + [target]].describe().round(2)

data.shape


fig, axes = plt.subplots(3, 1, figsize=(15, 20))

sns.heatmap(data[features + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation heatmap")

sns.scatterplot(data=data, x="IRRADIATION", y="AC_POWER", hue="PLANT", alpha=0.4, ax=axes[1])
axes[1].set_title("AC power vs. irradiation")

sns.lineplot(data=data, x="HOUR", y="AC_POWER", ax=axes[2])
axes[2].set_title("Average AC power by hour of day")

plt.tight_layout()
plt.show()

days = np.sort(data["DATE_TIME"].dt.normalize().unique())
cutoff = days[int(len(days) * 0.8)]

train = data[data["DATE_TIME"] < cutoff]
test = data[data["DATE_TIME"] >= cutoff]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print(f"Train: {len(train)} rows | Test: {len(test)} rows | Test starts on {str(cutoff)[:10]}")


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

rows, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })

results = pd.DataFrame(rows).set_index("Model").sort_values("RMSE").round(3)
results


best_name = results.index[0]
best_model = models[best_name]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results["RMSE"].plot(kind="bar", ax=axes[0])
axes[0].set_title("Model comparison (lower RMSE is better)")

axes[1].scatter(y_test, predictions[best_name],s=12)
axes[1].plot([0, y_test.max()], [0, y_test.max()], color="red")
axes[1].set_xlabel("Actual AC power")
axes[1].set_ylabel("Predicted AC power")
axes[1].set_title(f"{best_name}: predicted vs. actual")

corrs= data[features + [target]].corr()[target].drop(target)
corrs.plot(kind="bar", ax=axes[2])
axes[2].set_title("Correlation with AC power")

plt.tight_layout()
plt.show()

best = results.loc[best_name]
top_feature = corrs.abs().idxmax()

print(
    "Solar power output fluctuates with weather conditions, complicating grid planning and increasing "+"\n"+
    "reliance on fossil-fuel backup generation. This project builds a supervised regression pipeline that "+"\n"+
    "predicts solar power generation from weather measurements, using 34 days of 15-minute interval data "+"\n"+
    "from two grid-connected solar plants in India. Generation and weather records were merged, cleaned, "+"\n"+
    f"and enriched with engineered features, then used to train and compare {len(models)} regression models "+"\n"+
    f"evaluated with MAE, RMSE, and R². The best-performing model was {best_name} "+"\n"+
    f"(MAE = {best['MAE']:.2f} kW, RMSE = {best['RMSE']:.2f} kW, R² = {best['R2']:.3f}), "+"\n"+
    f"and the main driver of solar output was {top_feature}. The results illustrate how better forecasting "+"\n"+
    "can support renewable-energy integration and reduce dependence on backup fossil-fuel generation."+"\n"
)
