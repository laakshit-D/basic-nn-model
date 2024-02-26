# Developing a Neural Network Regression Model

## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

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
### Name: Laakshit D
### Register Number: 212222230071
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('dldata').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

df.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
x=[]
y=[]
for i in range(60):
  num = i+1
  x.append(num)
  y.append(num*12) 
df=pd.DataFrame({'INPUT': x, 'OUTPUT': y})
df.head()

inp=df[["INPUT"]].values
out=df[["OUTPUT"]].values
Input_train,Input_test,Output_train,Output_test=train_test_split(inp,out,test_size=0.33)
Scaler=MinMaxScaler()
Scaler.fit(Input_train)
Scaler.fit(Input_test)
Input_train=Scaler.transform(Input_train)
Input_test=Scaler.transform(Input_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([Dense(5,activation='relu'),
                  Dense(10,activation='relu'),
                  Dense(1)])
model.compile(loss="mse",optimizer="rmsprop")
history=model.fit(Input_train,Output_train, epochs=1000,batch_size=32)

prediction_test=int(input("Enter the value to predict:"))
preds=model.predict(Scaler.transform([[prediction_test]]))
print("The prediction for the given input "+str(prediction_test)+" is:"+str(int(np.round(preds))))

model.evaluate(Input_test,Output_test)

import matplotlib.pyplot as plt
plt.suptitle("   Harish Ravishankar")
plt.title("Error VS Iteration")
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.plot(pd.DataFrame(history.history))
plt.legend(['train'] )
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

worksheet = gc.open('dldata').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT':'float'})
dataset1 = dataset1.astype({'OUTPUT':'float'})

dataset1.head()

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)


ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=X_train1,y=y_train,epochs=50)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

```

## Dataset Information

![image](https://github.com/Harishspice/basic-nn-model/assets/117935868/e3984d1d-5bdf-4ce3-be69-f0d969797aec)

## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/Harishspice/Neural-Regression-Model/assets/117935868/8439765e-1f72-4ce7-bf81-c0a8147fbbf0)

### Test Data Root Mean Squared Error

![image](https://github.com/Harishspice/Neural-Regression-Model/assets/117935868/2e3bd2c0-bbc1-44c0-9160-215a25106e1e)

### New Sample Data Prediction

![image](https://github.com/Harishspice/Neural-Regression-Model/assets/117935868/433ee1d6-6112-4bdb-8e7d-7acf4d2fd526)
