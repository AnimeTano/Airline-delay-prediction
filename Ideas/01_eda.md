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

Found missing values in:

- arr_flights
- arr_del15
- delay cause columns
- cancellation/diversion columns

Need to investigate whether missing values represent missing information or zero values.


## Duplicates

No duplicated rows found.


# EDA plan

## Target analysis

- Calculate distribution of `bad_month`
- Check class balance
- Analyze percentage of delayed flights distribution
- Validate selected threshold


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