# Project goal

## Что хотим предсказать

Предсказываем, будет ли месяц "проблемным" для авиакомпании в конкретном аэропорту.

Одна строка датасета представляет собой агрегированную статистику:

- авиакомпания;
- аэропорт;
- месяц;
- количество рейсов и задержек.

На данном датасете решаем задачу бинарной классификации.


## Target

Месяц считается "проблемным", если доля задержанных рейсов составляет больше **20%** от общего количества прибывших рейсов.

Formula:

`bad_month = (arr_del15 / arr_flights) > 20%`

Classes:

- `0` — нормальный месяц (≤ 20% задержанных рейсов)
- `1` — проблемный месяц (> 20% задержанных рейсов)


## Выбор порога target

Было рассмотрено несколько значений порога:

| Delay rate threshold | Normal month | Problem month | Distribution |
|---|---:|---:|---:|
| 15% | 71,363 | 100,303 | 41.6% / 58.4% |
| 20% | 106,623 | 65,043 | 62.1% / 37.9% |
| 25% | 133,801 | 37,865 | 77.9% / 22.1% |

Порог **20%** выбран как компромисс между:

- бизнес-смыслом (значительный уровень задержек);
- достаточным количеством объектов обоих классов для обучения модели.


# Shape of data

Shape: (171666, 21)


# Columns

- year: The year of the data (int64)
- month: The month of the data (int64)
- carrier: Carrier code (object)
- carrier_name: Carrier name (object)
- airport: Airport code (object)
- airport_name: Airport name (object)
- arr_flights: Number of arriving flights (float64)
- arr_del15: Number of flights delayed by 15 minutes or more (float64)
- carrier_ct: Number of delays caused by carrier (float64)
- weather_ct: Number of delays caused by weather (float64)
- nas_ct: Number of delays caused by NAS (National Airspace System) (float64)
- security_ct: Number of delays caused by security issues (float64)
- late_aircraft_ct: Number of delays caused by late aircraft arrival (float64)
- arr_cancelled: Number of cancelled flights (float64)
- arr_diverted: Number of diverted flights (float64)
- arr_delay: Total arrival delay time (float64)
- carrier_delay: Total delay time caused by carrier (float64)
- weather_delay: Total delay time caused by weather (float64)
- nas_delay: Total delay time caused by NAS (float64)
- security_delay: Total delay time caused by security (float64)
- late_aircraft_delay: Total delay time caused by late aircraft arrival (float64)


# Data quality check

## Missing values

Missing values were found in several numerical columns.

Important columns for target creation:

| Column | Missing values | Percentage |
|---|---:|---:|
| arr_flights | 240 | 0.14% |
| arr_del15 | 443 | 0.26% |

Since `arr_flights` and `arr_del15` are required to calculate the target variable, rows with missing values in these columns cannot be used.

Decision:

- Remove rows with missing values in `arr_flights` or `arr_del15`.
- The amount of removed data is less than 1%, therefore it should not significantly affect the dataset distribution.


## Duplicates

No duplicated rows found.


# EDA plan

## Target analysis

- Calculate distribution of `bad_month`
- Check class balance
- Analyze percentage of delayed flights distribution
- Validate selected threshold

## Seasonal analysis

The probability of a problematic month varies significantly depending on the month.

The highest delay rates are observed during:
- June (55.1%)
- July (53.5%)
- December (50.1%)

The lowest delay rates are observed during:
- September (20.1%)
- October (24.9%)
- November (25.8%)

This suggests a seasonal component in flight delays. Summer months and the holiday period in December have a higher probability of problematic operations.

## Yearly analysis

The share of problematic months changes over time.

A noticeable decrease can be observed in 2020, which may be related to the COVID-19 pandemic and the significant reduction in air traffic.

After 2020, the share of problematic months increases sharply, indicating a deterioration of operational stability during the recovery period.

## Airport workload analysis

The number of arriving flights has a highly skewed distribution.

Statistics:
- Median: 101 flights per month
- Mean: 363 flights per month
- Maximum: 21,977 flights per month

The large difference between mean and median indicates the presence of major airport hubs with significantly higher traffic volumes.

## Airport workload analysis

A positive relationship can be observed between airport workload and probability of a problematic month.

Airports with the highest flight volumes have a higher share of problematic months:

- Very Low traffic: 35.3%
- Very High traffic: 41.8%

This suggests that airport workload may be an informative feature for the classification model.

## Delay causes analysis

Delay causes differ significantly between normal and problematic months.

The biggest contributors to problematic months are:

1. Late aircraft arrival
2. National Airspace System (NAS) delays
3. Carrier-related delays

Compared with normal months, problematic months have approximately twice as many delays caused by late aircraft arrival and NAS issues.

This suggests that operational factors and network congestion play a larger role in severe delay periods than external factors such as weather.

## Numerical features correlation analysis

No strong linear correlation was found between numerical features and the target variable.

The highest correlation with `bad_month` was observed for:
- `arr_diverted`: 0.049
- `arr_cancelled`: 0.023

This indicates that numerical features alone are not sufficient for prediction.

Strong correlations were observed between traffic volume and operational disruption counts:
- `arr_flights` and `arr_diverted`: 0.64
- `arr_flights` and `arr_cancelled`: 0.42

Categorical features such as airline and airport are expected to provide more predictive information.

## Airline analysis

Questions:

- Which airlines have the highest percentage of delayed flights?
- Which airlines have the most problematic months?
- Does carrier influence delay probability?


## Airport analysis

Questions:

- Which airports have the highest delay rate?
- Are there airports with consistently bad performance?


## Time analysis

Questions:

- Are there seasonal patterns?
- Does delay probability change depending on month/year?
- Are there long-term trends?


## Delay causes analysis

Questions:

- Which factors contribute most to delays?
- Are carrier, weather, NAS or late aircraft delays the main contributors?


# Potential data leakage

Features that should not be used for prediction:

- arr_del15
- arr_delay
- carrier_delay
- weather_delay
- nas_delay
- security_delay
- late_aircraft_delay

Reason:

These values are known only after delays happen and contain direct information about the target.


# Feature Engineering ideas

## Existing features

Potentially useful:

1. year
2. month
3. carrier
4. airport
5. arr_flights


## New features

Ideas:

1. delay_rate = arr_del15 / arr_flights (only for EDA, not for model)
2. average delays per flight
3. number of flights per carrier/airport
4. season
5. airport workload category
6. carrier performance statistics