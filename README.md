# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries

2.Load and check dataset

3.Encode categorical data

4.Split into features and target

5.Train Decision Tree Regressor

6.Predict and evaluate

7.Predict new value

8.Visualize tree


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Balasuriya M
RegisterNumber:  212224240021
*/
```
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Salary.csv']))
data.head()
print(data.head())          # View first 5 rows
print(data.info())          # Dataset info
print(data.isnull().sum())  # Check for null values
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())  # View updated dataset
x = data[["Position", "Level"]]  # Features
y = data["Salary"]               # Target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)
print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/3b0d4a94-7e54-4539-9005-1fdede334905)

![image](https://github.com/user-attachments/assets/6f0b7572-7595-4df1-b58b-5a534a171ce4)

![image](https://github.com/user-attachments/assets/18825212-7306-4c15-8580-f7f03981d8fc)

![image](https://github.com/user-attachments/assets/dc02433e-a473-49fa-9239-d745b1323684)

![image](https://github.com/user-attachments/assets/63200413-d73f-4083-b124-7ae51ad846bd)

![image](https://github.com/user-attachments/assets/68569903-d3d8-4775-ba9b-e73a20572e7b)

![image](https://github.com/user-attachments/assets/edf8ddd7-6b95-415e-99b2-b52d5c074ee0)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
