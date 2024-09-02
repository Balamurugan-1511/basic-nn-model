# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![image](https://github.com/user-attachments/assets/4bd0e5aa-bbcc-43e9-afb5-eb54ea57dc6f)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Bala murugan P
### Register Number:212222230017
```python

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('MyMLData').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()

df=df.astype({'input':'float64'})
df=df.astype({'output':'float64'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

X=df.iloc[:,:-1].values
y=df[['output']].values

X.shape
Y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)


scaler=MinMaxScaler()

scaler.fit(X_train)

X_train1=scaler.fit_transform(X_train)

Ai_Brain = Sequential(
    [
        Dense(units =3 , activation= 'relu',input_shape=[1]),
        Dense(units= 3, activation= 'relu'),
        Dense(units=1)
    ]
)

Ai_Brain.summary()

Ai_Brain.compile(optimizer='rmsprop', loss='mean_squared_error')

Ai_Brain.fit(X_train1,y_train,epochs=10)

loss_df = pd.DataFrame(Ai_Brain.history.history)

loss_df.plot()

X_test1=scaler.fit_transform(X_test)

Ai_Brain.evaluate(X_test1,y_test)

X_n1=[[3]]

X_n1=scaler.fit_transform(X_n1)

Ai_Brain.predict(X_n1)

x_n1=[[20]]

X_n1=scaler.fit_transform(X_n1)

Ai_Brain.predict(X_n1)




```
## Dataset Information

![Screenshot 2024-09-02 110332](https://github.com/user-attachments/assets/ac2494b5-4582-4de2-9957-aeaef02ec19d)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-09-02 110149](https://github.com/user-attachments/assets/d6dfae5c-1c2d-4c52-88be-55d068d4ce19)


### Test Data Root Mean Squared Error

![Screenshot 2024-09-02 110116](https://github.com/user-attachments/assets/92a66f9c-13a3-4deb-be46-0eafab3c8cfe)

### New Sample Data Prediction

![Screenshot 2024-09-02 105912](https://github.com/user-attachments/assets/00db397a-49ec-4906-b675-e62a1598ab4c)



## RESULT

Thus the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
