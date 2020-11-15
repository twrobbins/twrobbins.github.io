---
title:  "Stock Market Prediction using LSTMs"
category: posts
date: 2020-11-15
excerpt: # "Explanation of, and code for, a small Python tool for sampling from arbitrary distributions."
---


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
    947/947 [==============================] - 8s 8ms/sample - loss: 0.0013 - val_loss: 0.0209
    Epoch 5/100
    947/947 [==============================] - 9s 10ms/sample - loss: 8.8416e-04 - val_loss: 0.0083
    Epoch 6/100
    947/947 [==============================] - 9s 10ms/sample - loss: 8.4497e-04 - val_loss: 0.0147
    Epoch 7/100
    947/947 [==============================] - 9s 9ms/sample - loss: 9.7683e-04 - val_loss: 0.0140
    Epoch 8/100
    947/947 [==============================] - 9s 10ms/sample - loss: 9.8646e-04 - val_loss: 0.0211
    Epoch 9/100
    947/947 [==============================] - 8s 8ms/sample - loss: 0.0011 - val_loss: 0.0117
    Epoch 10/100
    947/947 [==============================] - 9s 9ms/sample - loss: 7.4686e-04 - val_loss: 0.0113
    Epoch 11/100
    947/947 [==============================] - 8s 8ms/sample - loss: 8.1362e-04 - val_loss: 0.0050
    Epoch 12/100
    947/947 [==============================] - 8s 8ms/sample - loss: 8.4127e-04 - val_loss: 0.0103
    Epoch 13/100
    947/947 [==============================] - 7s 8ms/sample - loss: 7.9220e-04 - val_loss: 0.0157
    Epoch 14/100
    947/947 [==============================] - 8s 8ms/sample - loss: 7.6653e-04 - val_loss: 0.0069
    Epoch 15/100
    947/947 [==============================] - 8s 9ms/sample - loss: 7.0729e-04 - val_loss: 0.0092
    Epoch 16/100
    947/947 [==============================] - 8s 9ms/sample - loss: 6.9992e-04 - val_loss: 0.0046
    Epoch 17/100
    947/947 [==============================] - 8s 8ms/sample - loss: 6.7202e-04 - val_loss: 0.0099
    Epoch 18/100
    947/947 [==============================] - 9s 9ms/sample - loss: 7.2842e-04 - val_loss: 0.0112
    Epoch 19/100
    947/947 [==============================] - 8s 9ms/sample - loss: 5.9371e-04 - val_loss: 0.0136
    Epoch 20/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.9450e-04 - val_loss: 0.0129
    Epoch 21/100
    947/947 [==============================] - 8s 8ms/sample - loss: 6.8016e-04 - val_loss: 0.0034
    Epoch 22/100
    947/947 [==============================] - 8s 9ms/sample - loss: 6.4839e-04 - val_loss: 0.0030
    Epoch 23/100
    947/947 [==============================] - 9s 9ms/sample - loss: 6.4723e-04 - val_loss: 0.0081
    Epoch 24/100
    947/947 [==============================] - 8s 9ms/sample - loss: 6.1321e-04 - val_loss: 0.0054
    Epoch 25/100
    947/947 [==============================] - 10s 10ms/sample - loss: 5.3872e-04 - val_loss: 0.0024
    Epoch 26/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.4037e-04 - val_loss: 0.0033
    Epoch 27/100
    947/947 [==============================] - 8s 8ms/sample - loss: 6.3042e-04 - val_loss: 0.0042
    Epoch 28/100
    947/947 [==============================] - 8s 9ms/sample - loss: 7.1209e-04 - val_loss: 0.0039
    Epoch 29/100
    947/947 [==============================] - 8s 8ms/sample - loss: 6.9452e-04 - val_loss: 0.0052
    Epoch 30/100
    947/947 [==============================] - 8s 9ms/sample - loss: 5.6304e-04 - val_loss: 0.0032
    Epoch 31/100
    947/947 [==============================] - 8s 8ms/sample - loss: 4.7539e-04 - val_loss: 0.0039
    Epoch 32/100
    947/947 [==============================] - 8s 8ms/sample - loss: 4.6555e-04 - val_loss: 0.0046
    Epoch 33/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.1242e-04 - val_loss: 0.0067
    Epoch 34/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.2162e-04 - val_loss: 0.0070
    Epoch 35/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.7327e-04 - val_loss: 0.0029
    Epoch 36/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.9616e-04 - val_loss: 0.0049
    Epoch 37/100
    947/947 [==============================] - 9s 9ms/sample - loss: 5.3179e-04 - val_loss: 0.0026
    Epoch 38/100
    947/947 [==============================] - 9s 10ms/sample - loss: 4.3335e-04 - val_loss: 0.0023
    Epoch 39/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.5289e-04 - val_loss: 0.0022
    Epoch 40/100
    947/947 [==============================] - 8s 9ms/sample - loss: 5.2819e-04 - val_loss: 0.0044
    Epoch 41/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.7020e-04 - val_loss: 0.0025
    Epoch 42/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.9542e-04 - val_loss: 0.0031
    Epoch 43/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.9600e-04 - val_loss: 0.0030
    Epoch 44/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.0832e-04 - val_loss: 0.0023
    Epoch 45/100
    947/947 [==============================] - 8s 8ms/sample - loss: 5.1038e-04 - val_loss: 0.0053
    Epoch 46/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.0401e-04 - val_loss: 0.0033
    Epoch 47/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.9002e-04 - val_loss: 0.0045
    Epoch 48/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.8333e-04 - val_loss: 0.0040
    Epoch 49/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.0078e-04 - val_loss: 0.0068
    Epoch 50/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.5346e-04 - val_loss: 0.0055
    Epoch 51/100
    947/947 [==============================] - 9s 9ms/sample - loss: 4.1450e-04 - val_loss: 0.0043
    Epoch 52/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.8437e-04 - val_loss: 0.0067
    Epoch 53/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.2252e-04 - val_loss: 0.0024
    Epoch 54/100
    947/947 [==============================] - 9s 9ms/sample - loss: 4.0227e-04 - val_loss: 0.0041
    Epoch 55/100
    947/947 [==============================] - 9s 10ms/sample - loss: 3.2827e-04 - val_loss: 0.0031
    Epoch 56/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1468e-04 - val_loss: 0.0032
    Epoch 57/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.8628e-04 - val_loss: 0.0013
    Epoch 58/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.6025e-04 - val_loss: 0.0022
    Epoch 59/100
    947/947 [==============================] - 8s 9ms/sample - loss: 4.1576e-04 - val_loss: 0.0026
    Epoch 60/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.6075e-04 - val_loss: 0.0019
    Epoch 61/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.7549e-04 - val_loss: 0.0011
    Epoch 62/100
    947/947 [==============================] - 9s 9ms/sample - loss: 4.3230e-04 - val_loss: 0.0023
    Epoch 63/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.1173e-04 - val_loss: 0.0036
    Epoch 64/100
    947/947 [==============================] - 9s 10ms/sample - loss: 3.5084e-04 - val_loss: 0.0026
    Epoch 65/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.5735e-04 - val_loss: 0.0041
    Epoch 66/100
    947/947 [==============================] - 9s 10ms/sample - loss: 4.3803e-04 - val_loss: 0.0035
    Epoch 67/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.8863e-04 - val_loss: 0.0046
    Epoch 68/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.4988e-04 - val_loss: 0.0070
    Epoch 69/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.2481e-04 - val_loss: 0.0049
    Epoch 70/100
    947/947 [==============================] - 9s 10ms/sample - loss: 3.7401e-04 - val_loss: 0.0032
    Epoch 71/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.0224e-04 - val_loss: 0.0027
    Epoch 72/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.2879e-04 - val_loss: 0.0024
    Epoch 73/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.8326e-04 - val_loss: 0.0017
    Epoch 74/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.2003e-04 - val_loss: 0.0042
    Epoch 75/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.2650e-04 - val_loss: 0.0022
    Epoch 76/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.3282e-04 - val_loss: 0.0055
    Epoch 77/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.2926e-04 - val_loss: 0.0025
    Epoch 78/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.3287e-04 - val_loss: 0.0012
    Epoch 79/100
    947/947 [==============================] - 8s 9ms/sample - loss: 2.8060e-04 - val_loss: 0.0016
    Epoch 80/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.6117e-04 - val_loss: 0.0058
    Epoch 81/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.2890e-04 - val_loss: 0.0018
    Epoch 82/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.0286e-04 - val_loss: 0.0020
    Epoch 83/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.0773e-04 - val_loss: 0.00281339
    Epoch 84/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.3255e-04 - val_loss: 0.0010
    Epoch 85/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.0312e-04 - val_loss: 0.0016
    Epoch 86/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1248e-04 - val_loss: 0.0026
    Epoch 87/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.1951e-04 - val_loss: 0.0026
    Epoch 88/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.8209e-04 - val_loss: 0.0047
    Epoch 89/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.1429e-04 - val_loss: 0.0030
    Epoch 90/100
    947/947 [==============================] - 9s 10ms/sample - loss: 2.8929e-04 - val_loss: 0.0025
    Epoch 91/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.0371e-04 - val_loss: 0.0041
    Epoch 92/100
    947/947 [==============================] - 8s 9ms/sample - loss: 3.1481e-04 - val_loss: 0.0033
    Epoch 93/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1258e-04 - val_loss: 0.0019
    Epoch 94/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1296e-04 - val_loss: 0.0023
    Epoch 95/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.2285e-04 - val_loss: 0.0012
    Epoch 96/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1647e-04 - val_loss: 0.0045
    Epoch 97/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.1076e-04 - val_loss: 0.0028
    Epoch 98/100
    947/947 [==============================] - 8s 8ms/sample - loss: 3.3025e-04 - val_loss: 0.0018
    Epoch 99/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.8272e-04 - val_loss: 0.0032
    Epoch 100/100
    947/947 [==============================] - 9s 9ms/sample - loss: 3.3997e-04 - val_loss: 0.0027
    


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

    (251, 60)
    (251, 60, 1)
    (251, 1)
    


```python
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse
```




    92.85583733425672




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

    C:\Users\Tim\anaconda3\envs\minimal\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


![png](output_17_1.png)



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

    C:\Users\Tim\anaconda3\envs\minimal\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    


![png](output_18_1.png)



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


![png](output_19_0.png)



```python
np.where(val_loss == min(val_loss))
```




    (array([83], dtype=int64),)




```python
#Show the valid and predicted prices
valid
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>close</th>
      <th>predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1007</th>
      <td>1788.88</td>
      <td>1825.926025</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1826.17</td>
      <td>1819.333984</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>1833.07</td>
      <td>1823.096802</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>1765.37</td>
      <td>1831.908936</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>1753.94</td>
      <td>1808.410889</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>3438.99</td>
      <td>3197.238037</td>
    </tr>
    <tr>
      <th>1254</th>
      <td>3260.12</td>
      <td>3226.234863</td>
    </tr>
    <tr>
      <th>1255</th>
      <td>3038.10</td>
      <td>3091.545898</td>
    </tr>
    <tr>
      <th>1256</th>
      <td>3180.75</td>
      <td>2901.893066</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>3133.54</td>
      <td>2948.676025</td>
    </tr>
  </tbody>
</table>
<p>251 rows Ã 2 columns</p>
</div>




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

    [[2970.7285]]
    


```python
# data is of type list - convert to pandas df
final = pd.DataFrame(new_df)

# set final to last date (date we are predicting)
final = final.iloc[-1]

print('Predicted Price: ', pred_price)
print('Actual Price: ', final['close'])

```

    Predicted Price:  [[2970.7285]]
    Actual Price:  3133.54
    
