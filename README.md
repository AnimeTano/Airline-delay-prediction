# Airline Delay Prediction

## Project goal

Прогнозирование того, прибудет ли рейс с задержкой на 3 и более часов.

Такие задержки могут привести к обязательной компенсации пассажирам в соответствии с правилами ЕС о правах авиапассажиров. Раннее прогнозирование значительных задержек может помочь авиакомпаниям:
- оптимизировать расписание рейсов;
- превентивно информировать пассажиров;
- сократить эксплуатационные расходы;
- оценить потенциальные компенсационные обязательства.

## Бизнес-проблема

Длительные задержки рейсов могут привести к выплате пассажирам компенсационных выплат в соответствии с правилами ЕС.
Этот проект прогнозирует, будет ли рейс задержан на 3 или более часа, используя исторические данные о полетах и ​​модели машинного обучения.

## Dataset

- Kaggle Airline Delay Dataset

## Tech stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Project structure

...

## EDA

...

## Feature engineering

...

## Modeling

| Model | ROC-AUC |
|------|------|
| Logistic Regression | 0.0 |
| Random Forest | 0.0 |
| XGBoost | 0.0 |

## Results

## Run
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay) and put it into:

data/raw/

pip install -r requirements.txt

python src/train.py