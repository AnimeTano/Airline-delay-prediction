import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv("./data/raw/Airline_Delay_Cause.csv")

# Сортируем по времени
data = data.sort_values(['carrier', 'airport', 'year', 'month'])

# Создаём лаги для arr_del15 и arr_delay (за 1, 3, 6 месяцев)
for col in ['arr_del15', 'arr_delay']:
    for lag in [1, 3, 6]:
        data[f'{col}_lag{lag}'] = data.groupby(['carrier', 'airport'])[col].shift(lag)

# Создаем скользящее среднее, как признаки для временных рядов
for window in [3, 6]:
    data[f'arr_del15_rolling{window}'] = (
        data.groupby(['carrier', 'airport'])['arr_del15'].transform(lambda x: x.rolling(window, min_periods = 1).mean())
    )

# Заполняем пропуски (первые лаги) нулями или средним
data = data.fillna(0)

print(data.columns)

# Устраняем возможное data_leakage
leak_features = [
    'arr_del15', 'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
    'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct'
]

X = data.drop(columns = leak_features)
y = ((data['arr_del15'] / (data['arr_flights'])) * 100) > 20.0

# Добавляем новые признаки
X = X.copy() # уже реализована deepcopy в pd

# Кодируем признаки таким образом, чтобы модель понимала цикличность времен года. Кодируем месяца как углы на окружности
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

def get_season(x):
    if (x in [12, 1, 2]):
        return "Winter"
    elif (x in [3, 4, 5]):
        return 'Spring'
    elif (x in [6, 7, 8]):
        return "Summer"
    else:
        return "Autumn"

X['season'] = X['month'].apply(get_season)

# Создаем маски для разделение дата сета
## В силу того, что год может привести к data leakage, будем разделять данные по годам
train_mask = (X['year'] < 2021)
val_mask = (X['year'] >= 2021) & (X['year'] <= 2022)
test_mask = (X['year'] > 2022)

X_train = X[train_mask].drop(columns = ['year'], axis = 1)
y_train = y[train_mask]

X_val = X[val_mask].drop(columns = ['year'], axis = 1)
y_val = y[val_mask]

X_test = X[test_mask].drop(columns = ['year'], axis = 1)
y_test = y[test_mask]

# Сохраняем результаты и получаем размеры

for name, df in [('X_train', X_train), ('y_train', y_train), ('X_val', X_val), ('y_val', y_val), ('X_test', X_test), ('y_test', y_test)]:
    df.to_csv(f'data/processed/{name}.csv', index=False)

print("Данные подготовлены и сохранены")
print(f"Sizes: train {X_train.shape}, valid {X_val.shape}, test {X_test.shape}")