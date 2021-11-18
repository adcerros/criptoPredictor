import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Lectura de datos y transformacion de la fecha
initialData = pd.read_csv("shib_data.csv")
initialData['date'] = pd.to_datetime(initialData.snapped_at, format='%Y-%m-%d')

# Reordeenacion y refactorización de los datos
# Separo las fechas para normalizar y añadirlas posteriormente
dataDates = initialData[['date']]
# Se redefine el conjunto de datos
finalData = initialData[['price', 'market_cap', 'total_volume']]
dataPrices = finalData[['price']]

# Normalizacion de los datos
scaler = MinMaxScaler()
scaler = scaler.fit(finalData)
finalData = scaler.transform(finalData)
# Normalizacion de los precios para almacenar maximos y minimos y denormalizar las salidas posteriormente
priceScaler = MinMaxScaler()
priceScaler = scaler.fit(dataPrices)


finalData = pd.DataFrame(data=finalData, columns=['price', 'market_cap', 'total_volume'])


# Se crean los conjuntos de entrenamiento y test con sus entradas y salidas
batchSize = 5
numberOfBatches = int(np.floor(len(finalData) / batchSize))
trainIn, trainOut = [], []
testIn, testOut = [], []
testDates = []
for i in range(numberOfBatches):
  if(i < numberOfBatches*(8/9)): 
    trainIn.append(finalData[(i*batchSize) : (i*batchSize) + batchSize - 1])
    trainOut.append(finalData.iloc[(i*batchSize) + batchSize - 1][0])
  else:
    testIn.append(finalData[(i*batchSize) : (i*batchSize) + batchSize - 1])
    testOut.append(finalData.iloc[(i*batchSize) + batchSize - 1][0])
    testDates.append(dataDates.iloc[(i*batchSize) + batchSize - 1][0])


# Transformacion de las listas a np.array
trainIn=np.array(trainIn)
trainOut=np.array(trainOut)
testIn=np.array(testIn)
testOut=np.array(testOut)

# Implementacion del modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.LSTM(units=25))
model.add(tf.keras.layers.Dense(units=50))
model.add(tf.keras.layers.Dense(units=50))
model.add(tf.keras.layers.Dense(units=25))
model.add(tf.keras.layers.Dense(units=10))
model.add(tf.keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer = tf.keras.optimizers.SGD(learning_rate=0.025, momentum = 0.5))

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
history = model.fit(trainIn, trainOut, epochs=1000, batch_size=10, validation_split=0.2, verbose=1, shuffle=True, callbacks=[callback])

# Graficas
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(figsize=(30,12))
plt.plot(history.history['loss'], label='Training loss', color='orange')
plt.plot(history.history['val_loss'], label='Validation loss', color='red')


scores = model.evaluate(trainIn, trainOut)
result = model.predict(testIn)

result = priceScaler.inverse_transform(np.array(result).reshape(-1,1))
testOut = priceScaler.inverse_transform(testOut.reshape(-1,1))

print(model.metrics_names[0], scores)
absErrors = []
for i in range(len(result)):
  absErrors.append(abs(testOut[i] - result[i]))
  print("Date: " + str(testDates[i]) + "\tExpected: " + str(testOut[i][0]) + " \tPredicted: " + str(result[i][0]) + " \tAbsDiference: " + str(absErrors[i][0]))
print()
fig, ax = plt.subplots(figsize=(30,12))
plt.plot(testOut, label='Expected values', color='orange')
plt.plot(result, label='Predicted values',color='red')
plt.plot(absErrors, label='Abs Errors',color='blue')

