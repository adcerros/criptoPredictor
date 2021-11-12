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
myPrices = mydata[['price']]

# Normalizacion
scaler = MinMaxScaler()
scaler = scaler.fit(mydata)
mydata = scaler.transform(mydata)
priceScaler = MinMaxScaler()
priceScaler = scaler.fit(myPrices)

# Union de las fechas para separar los conjuntos de entrenamiento y test por fecha
mydata = pd.DataFrame(data=mydata, columns=['price', 'market_cap', 'total_volume'])


# Se crean los conjuntos de entrenamiento y test con sus entradas y salidas
batchSize = 5
numberOfBatches = int(np.floor(len(mydata) / batchSize))
trainIn, trainOut = [], []
testIn, testOut = [], []
for i in range(numberOfBatches):
  if(i < numberOfBatches*(8/9)): 
    trainIn.append(mydata[(i*batchSize) : (i*batchSize) + batchSize-1])
    trainOut.append(mydata.iloc[(i*batchSize) + batchSize - 1][0])
  else:
    testIn.append(mydata[(i*batchSize) : (i*batchSize) + batchSize-1])
    testOut.append(mydata.iloc[(i*batchSize) + batchSize - 1][0])

trainIn=np.array(trainIn)
trainOut=np.array(trainOut)
testIn=np.array(testIn)
testOut=np.array(testOut)

# Implementacion del modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=25))
model.add(tf.keras.layers.Dense(units=50))
model.add(tf.keras.layers.Dense(units=50))
model.add(tf.keras.layers.Dense(units=25))
model.add(tf.keras.layers.Dense(units=10))
model.add(tf.keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer = tf.keras.optimizers.SGD(learning_rate=0.01))

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
history = model.fit(trainIn, trainOut, epochs=1000, batch_size=10, validation_split=0.2, verbose=1, shuffle=True, callbacks=[callback])

# Graficas
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(figsize=(30,12))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')

scores = model.evaluate(trainIn, trainOut)
result = model.predict(testIn)

result = priceScaler.inverse_transform(np.array(result).reshape(-1,1))
testOut = priceScaler.inverse_transform(testOut.reshape(-1,1))

print(model.metrics_names[0], scores)
absErrors = []
for i in range(len(result)):
  absErrors.append(abs(testOut[i] - result[i]))
  print("Expected: " + str(testOut[i][0]) + " \tPredicted: " + str(result[i][0]) + " \tAbsDiference: " + str(absErrors[i][0]))
print()
fig, ax = plt.subplots(figsize=(30,12))
plt.plot(testOut, label='Expected values', color='orange')
plt.plot(result, label='Predicted values',color='red')
plt.plot(absErrors, label='Abs Errors',color='blue')