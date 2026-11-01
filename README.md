# Solar Energy Production Prediction

**Course:** CS 316 – Introduction to AI and Data Science (Section 1282)
**Group members:** Rumah Juweria, Ruqayyah Raja, Rayhana Basha

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yscrayzk/solar-energy-prediction/blob/main/solar_energy_prediction.ipynb)

## Project overview

Solar power output fluctuates with weather conditions, which complicates grid planning and increases reliance on fossil-fuel backup generation. This project builds a supervised regression pipeline that predicts solar power generation (AC power) from weather measurements. We merge and clean generation and weather data from two solar plants, explore the relationships between weather and power, engineer features, and compare three regression models using MAE, RMSE and R².

## Dataset

[Solar Power Generation Data (Kaggle)](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data): 34 days (15 May – 17 June 2020) of 15-minute interval data from two grid-connected solar plants in India. Each plant has:

- a **generation file** with inverter-level DC/AC power and yield (22 inverters per plant)
- a **weather sensor file** with ambient temperature, module temperature and irradiation

No external meteorological data was used.

## Repository structure

```
├── solar_energy_prediction.ipynb     # Main notebook (all code, comments and outputs)
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── requirements.txt                  # Python libraries needed
└── README.md
```

## How to run the code

### Option A: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this README (or open `solar_energy_prediction.ipynb` and choose *Open in Colab*).
2. In the data-loading cell, make sure `scrayzk/solar-energy-prediction` in the `git clone` line is this repository's address. This downloads the code and the `data/` folder into Colab automatically.
3. Run all cells: **Runtime → Run all**. The notebook takes about a minute to run.

### Option B: Run locally

1. Clone the repository:
   ```
   git clone https://github.com/scrayzk/solar-energy-prediction.git
   cd solar-energy-prediction
   ```
2. Install the libraries (Python 3.9+ recommended):
   ```
   pip install -r requirements.txt
   ```
3. Start Jupyter and open the notebook:
   ```
   jupyter notebook solar_energy_prediction.ipynb
   ```
4. Skip the cell that runs `git clone` (the data is already in `data/`) and run the remaining cells from top to bottom.

## Methodology

1. **Data collection and cleaning:** The four CSV files were loaded and checked for missing values (none found). Because each plant has about 22 inverters reporting at every timestamp, AC power was averaged across inverters to give one value per plant per timestamp. Generation (6,417 records) and weather (6,441 records) data were merged on timestamp and plant ID with an inner join, giving 6,416 records.
2. **Filtering:** Only records with positive irradiation (daytime) were kept, since night-time records have zero power and are trivially easy to predict. This left 3,616 records.
3. **Feature engineering:** `PLANT` (plant 1 or 2), `HOUR` (time of day as a decimal) and `TEMP_DIFF` (module minus ambient temperature).
4. **Exploratory data analysis:** correlation heatmap, power vs. irradiation, and average daily power curve.
5. **Train/test split:** chronological. The first 27 days (15 May – 10 June, 2,862 records) were used for training and the last 7 days (11 – 17 June, 754 records) for testing, so models are evaluated on unseen days.
6. **Models:** Linear Regression (baseline), Decision Tree (max depth 8) and Random Forest (100 trees), all with `random_state=42`.
7. **Evaluation:** MAE, RMSE and R² on the test set.

## Results

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| **Random Forest** | **35.1** | **77.9** | **0.935** |
| Decision Tree | 37.5 | 85.4 | 0.922 |
| Linear Regression | 98.0 | 125.8 | 0.831 |

- The **Random Forest** performed best on all three metrics.
- **Irradiation** is the main driver of solar output (correlation with AC power r = 0.91). Module temperature and temperature difference are also highly correlated with power, but largely because they rise with sunlight.
- Linear Regression performed worst, indicating a non-linear relationship between weather and power.

## Limitations

- Only 34 days from a single season; the test set covers just 7 days.
- No hyperparameter tuning or cross-validation.
- Models use weather measured at the same time as the power output; a real forecasting system would rely on forecast weather.
- Cloud cover and wind speed were not available.
- A few unexplained outages (mainly Plant 2) cause large errors, and a few near-zero-irradiation night-time records remain after filtering.

## Libraries used

`pandas`, `numpy`, `seaborn`, `matplotlib` and `scikit-learn`.
