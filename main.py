import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Lectura de datos y transformacion de la fecha
mydata = pd.read_csv("shib_data.csv")
mydata['date'] = pd.to_datetime(mydata.snapped_at, format='%Y-%m-%d')

# Reordeenacion y refactorización de los datos
# Separo las fechas para normalizar
mydataDate = mydata[['date']]
mydata = mydata[['price', 'market_cap', 'total_volume']]

# Normalizacion
scaler = MinMaxScaler()
scaler = scaler.fit(mydata)
mydata = scaler.transform(mydata)

# Union de las fechas para separar los conjuntos de entrenamiento y test por fecha
mydata = pd.DataFrame(data=mydata, columns=['price', 'market_cap', 'total_volume'])
mydata = pd.concat([mydataDate, mydata], axis=1)

# Los datos de salida seran los resultados del dia siguiente
mydata['next_day_price'] = mydata[['price']]
for i in mydata.index-1:
  mydata['next_day_price'][i] = mydata['price'][i+1]

# Se crean los conjuntos de entrenamiento y test con sus entradas y salidas
train, test = mydata.loc[mydata.date <= '2021-10-31'], mydata.loc[mydata.date > '2021-10-31']
trainInputs, trainOutputs = train[['price', 'market_cap', 'total_volume']], train[['next_day_price']]
testInputs, testOutputs = test[['price', 'market_cap', 'total_volume']], test[['next_day_price']]

# Implementacion del modelo
model = tf.keras.Sequential()


# Graficas
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(figsize=(30,12))
sns.lineplot(x='date', y='price', data = mydata.tail(30), palette='tab8', linewidth=2.5, ax=ax)