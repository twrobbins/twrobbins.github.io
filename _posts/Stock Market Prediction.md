```python
# import libraries
import http.client
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Flatten, Dense, Dropout


# pull 5 years worth of amazon stock info directly from iex api
TOKEN = 'Tsk_c85274ff81f940e29ce16dbbffc7c4c8'

# use production data for final analysis
production_TOKEN = 'pk_6939d57e996d48b784f1b25bcd91f1f0'

# use sandbox for testing
url = 'sandbox.iexapis.com'

# use production for final analysis
production_url = 'cloud.iexapis'

# define http connection
conn = http.client.HTTPSConnection(url)

# make connection to api and get last 5 years stock price
conn.request("GET", "/stable/stock/AMZN/chart?token=" + TOKEN + "&range=5y")

# get response from web page
res = conn.getresponse()
print(res)

# read data in to variable
data = res.read()

# decode data to utf-8
data.decode("utf-8");
```

    <http.client.HTTPResponse object at 0x000002072AB45CC8>
    


```python
# convert to json format
data = json.loads(data)

#print(data)
type(data)
```




    list




```python
# data is of type list - convert to pandas df
df = pd.DataFrame(data)

# sort values by date
df = df.sort_values('date')

# remove last date as this is what we will try to predict
df = df.iloc[:-1]


print(type(df))
print(df.head)
print(df.columns)


```

    <class 'pandas.core.frame.DataFrame'>
    <bound method NDFrame.head of             date   uClose    uOpen    uHigh     uLow  uVolume    close  \
    0     2015-11-16   673.73   667.23   651.04   641.03  7536269   674.85   
    1     2015-11-17   670.70   672.73   660.38   651.00  4510795   649.70   
    2     2015-11-18   695.94   666.73   666.44   654.80  4575998   694.58   
    3     2015-11-19   688.28   684.28   693.68   682.00  4744610   663.29   
    4     2015-11-20   685.74   674.28   700.33   673.72  3965415   677.07   
    ...          ...      ...      ...      ...      ...      ...      ...   
    1253  2020-11-06  3342.70  3375.96  3447.00  3233.00  4745814  3438.99   
    1254  2020-11-09  3262.50  3249.69  3380.00  3228.66  7274445  3260.12   
    1255  2020-11-10  3149.28  3143.46  3234.00  3138.42  6874965  3038.10   
    1256  2020-11-11  3273.63  3207.55  3162.22  3083.00  4502256  3180.75   
    1257  2020-11-12  3127.04  3287.09  3194.22  3162.97  4521475  3133.54   
    
             open     high      low   volume currency  change  changePercent  \
    0      668.58   656.69   637.44  7794882             0.00         0.0000   
    1      679.51   656.63   667.00  4396346            -4.57        -0.7234   
    2      648.21   667.52   655.59  4628026            20.58         3.2122   
    3      686.91   688.82   684.00  4909583            -2.32        -0.3520   
    4      697.90   692.79   662.90  3914091             7.23         1.1043   
    ...       ...      ...      ...      ...      ...     ...            ...   
    1253  3446.96  3370.00  3363.00  4864272           -10.80        -0.3300   
    1254  3361.15  3295.00  3246.50  7331207          -175.66        -5.2541   
    1255  3154.08  3130.00  3163.99  6763148          -111.86        -3.5516   
    1256  3150.90  3257.37  3054.00  4578114           105.62         3.3950   
    1257  3176.46  3304.95  3104.88  4559131           -27.56        -0.8909   
    
               label  changeOverTime  
    0     Nov 16, 15        0.000000  
    1     Nov 17, 15       -0.007141  
    2     Nov 18, 15        0.024485  
    3     Nov 19, 15        0.021718  
    4     Nov 20, 15        0.033419  
    ...          ...             ...  
    1253   Nov 6, 20        4.158069  
    1254   Nov 9, 20        3.942654  
    1255  Nov 10, 20        3.830111  
    1256  Nov 11, 20        4.025327  
    1257  Nov 12, 20        3.931258  
    
    [1258 rows x 16 columns]>
    Index(['date', 'uClose', 'uOpen', 'uHigh', 'uLow', 'uVolume', 'close', 'open',
           'high', 'low', 'volume', 'currency', 'change', 'changePercent', 'label',
           'changeOverTime'],
          dtype='object')
    


```python
# save df as csv file
df.to_csv('stock_dataset.csv')

```


```python
# check for missing values
df.isna().any()
```




    date              False
    uClose            False
    uOpen             False
    uHigh             False
    uLow              False
    uVolume           False
    close             False
    open              False
    high              False
    low               False
    volume            False
    currency          False
    change            False
    changePercent     False
    label             False
    changeOverTime    False
    dtype: bool




```python
#plt.scatter(df['close'], df['volume'])
```


```python
# check correlation between close price and volume
df['close'].corr(df['volume'])
```




    0.15362494923178294




```python
# set figure size to large
plt.figure(figsize=(16,8))

# set titles
plt.title('Close Price History', fontsize=30)

# plot close price
plt.plot(df['close'])
               
# set labels
plt.xlabel('Days', fontsize=20)
plt.ylabel('Close Price', fontsize=20)

plt.show()

```


![png](output_7_0.png)



```python
#Create a new dataframe with only the 'Close' column
data = df.filter(['close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on - first 80% (last 20% used for validation)
training_data_len = int(np.ceil( len(dataset) * .8 ))


print(training_data_len)
print(dataset.shape)
print(len(dataset))
```

    1007
    (1258, 1)
    1258
    


```python
#Scale the data
from sklearn.preprocessing import MinMaxScaler

# use min/max scaler to scale data from 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
# fit the scaler to the data
scaled_data = scaler.fit_transform(dataset)

scaled_data
```




    array([[0.05751591],
           [0.04966929],
           [0.06367153],
           ...,
           [0.7948334 ],
           [0.8393392 ],
           [0.82461001]])




```python
scaled_data.shape
```




    (1258, 1)




```python
#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

print(train_data.shape)

# create empty lists
x_train = []
y_train = []

# loop through training data starting at day 60 since need to look 60 days back
for i in range(60, len(train_data)):
    #Split the data into x_train and y_train data sets
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into 3-dimensional array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

```

    (1007, 1)
    (947, 60, 1)
    


```python
# create the validation data
validation_data = scaled_data[int(training_data_len):, :]

print(validation_data.shape)

# create empty lists
x_valid = []
y_valid= []

# loop through training data starting at day 60 since need to look 60 days back
for i in range(60, len(validation_data)):
    #Split the data into x_train and y_train data sets
    x_valid.append(validation_data[i-60:i, 0])
    y_valid.append(validation_data[i, 0])

        
# Convert the x_train and y_train to numpy arrays 
x_valid, y_valid = np.array(x_valid), np.array(y_valid)

#Reshape the data into 3-dimensional array
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))


x_valid.shape
```

    (251, 1)
    




    (191, 60, 1)




```python
#Build the LSTM model

# set model type to sequential
model = Sequential()
# add input layer
model.add(LSTM(units = 100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
# add 1st layers
model.add(LSTM(units = 100, return_sequences=True))
model.add(Dropout(rate = 0.2))
# add 2nd layers
model.add(LSTM(units = 100, return_sequences=True))
model.add(Dropout(rate = 0.2))
# add 3rd layers
model.add(LSTM(units = 100, return_sequences= False))
model.add(Dropout(rate = 0.2))
# add output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# metrics=['acc']

model.summary()


```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 60, 100)           40800     
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 60, 100)           80400     
    _________________________________________________________________
    dropout (Dropout)            (None, 60, 100)           0         
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 60, 100)           80400     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 100)           0         
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 101       
    =================================================================
    Total params: 282,101
    Trainable params: 282,101
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#fit the model with bacth size of 20 for 100 epochs
history = model.fit(x_train,
          y_train, 
          batch_size=20, 
          epochs=100, 
          validation_data = [x_valid, y_valid])
```

    Train on 947 samples, validate on 191 samples
    Epoch 1/100
    947/947 [==============================] - 20s 21ms/sample - loss: 0.0060 - val_loss: 0.0037
    Epoch 2/100
    947/947 [==============================] - 8s 9ms/sample - loss: 0.0012 - val_loss: 0.0079
    Epoch 3/100
    947/947 [==============================] - 8s 9ms/sample - loss: 0.0011 - val_loss: 0.0044
    Epoch 4/100
    920/947 [============================>.] - ETA: 0s - loss: 0.0013


```python
#Create the testing data set
#Create a new array containing scaled values for test data
test_data = scaled_data[training_data_len - 60: , : ]

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len : , : ]

# loop through test dataset starting 60 days out
for i in range(60, len(test_data)) :
    x_test.append(test_data[i-60 : i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)
print(x_test.shape)

# Reshape the x_test data (rows = number of samples, columns = time steps, features = 1 ('close' price))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# confirm shape of array
print(x_test.shape)

# Get the models predicted price values 
predictions = model.predict(x_test)
print(predictions.shape)

# reverse the scaling done on the predictions to get actual price
predictions = scaler.inverse_transform(predictions)

```


```python
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse
```


```python
# Plot the data

# define training and validation sets for plotting
train = data[:training_data_len]
valid = data[training_data_len:]

# define predictions for validation set
valid['predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))

# set title and labels
plt.title('LSTM Model', fontsize=30)
plt.xlabel('Days', fontsize=20)
plt.ylabel('Close Price', fontsize=20)

# plot training data, validation data and predictions
plt.plot(train['close'])
plt.plot(valid[['close', 'predictions']])
plt.legend(['Training', 'Validation', 'Predictions'], loc='lower right')
plt.show()
```


```python
# Plot the data

# define training and validation sets for plotting
valid = data[training_data_len:]
valid['predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))

# set title and labels
plt.title('Model Prediction on Validation Set', fontsize=30)
plt.xlabel('Days', fontsize=20)
plt.ylabel('Close Price', fontsize=20)

# plot training data, validation data and predictions     
plt.plot(valid[['close', 'predictions']])
plt.legend(['Validation Data', 'Predictions'], loc='lower right')
plt.show()
```


```python
# pull validation and loss info from model history
loss = history.history['loss']
val_loss = history.history['val_loss']

# number of epochs
epochs = range(1, len(loss) + 1)

# increase figure size
plt.figure(figsize=(16,8))

# plto epochs against training abd validaiton losses
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')

# set labels, titles, and legend
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('Training and Validation Loss', fontsize=30)
plt.legend()

plt.show()
```


```python
np.where(val_loss == min(val_loss))
```


```python
#Show the valid and predicted prices
valid
```


```python
# create new dataframe to hold test data
new_df = data.filter(['close'])

# look at values from days ago
last_60_days = new_df[-60:].values

# scale data between 0 1nd 1
last_60_days_scaled = scaler.transform(last_60_days)

# initialize empty list to store test data
X_test = []

# append scaled data to list
X_test.append(last_60_days_scaled)

# convert to numpy array
X_test = np.array(X_test)

# reshape numpy array into 3 dimension for model prediction
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict price on test data
pred_price = model.predict(X_test)

# reverse scaling down on price (0-1) to actual price
pred_price = scaler.inverse_transform(pred_price)

# predict next day price
print(pred_price)

```


```python
# data is of type list - convert to pandas df
final = pd.DataFrame(new_df)

# set final to last date (date we are predicting)
final = final.iloc[-1]

print('Predicted Price: ', pred_price)
print('Actual Price: ', final['close'])

```

