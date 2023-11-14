# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.
2. Print the untouched data and perform basic operations (df.head,df.info,df.describe()).
3. Split the data.
4. Train the model with Decision Trees Classifier using the data. 
5. Calculate the Accuracy.
6. Predict the outcome with a new data.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: B VIJAY KUMAR
RegisterNumber:  212222230173
*/
```
##### Import Pandas and read the file
```
import pandas as pd
df=pd.read_csv("Employee (1).csv")
```
##### Perform Basic Operations
```
df.head()
df.info()
df.describe()
```
##### Calculate the Number of Null values in the target Variable
```
df.isnull().sum()
```
##### Perform Label Encoding for the Salary and Department Columns
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])
df['Departments '] = le.fit_transform(df['Departments '])
```
##### Split The data into X and Y and allot them to Training and Testing sets
```
x = df[['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments ', 'salary' ] ]

y = df['left']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
```
##### Implement Decision Tree Classifier
```
from sklearn.tree import DecisionTreeClassifier
dk = DecisionTreeClassifier(criterion = "entropy")
dk.fit(x_train, y_train)
y_pred = dk.predict(x_test)
```
##### Calculate the Accuracy
```
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
```
##### Predict the Output with new data
```
dk.predict([[0.5,0.8,8,230,4,0,0,1,2]])
```
## Output:
##### HEAD 

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/fd285207-0e4a-4040-a281-486b447ad2c6)

##### DATATYPE OF EACH FEATURE
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/f151680e-4f11-420e-a817-a3c150fce649)

##### DESCRIBE
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/e2f53ff4-458b-49e1-8a76-c8f786fe0df6)

##### NULL COUNTS IN EACH FEATURE
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/b285e74f-84df-4970-aa47-4a246edf57ec)

##### X-DATA
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/c96df992-9edf-43ff-ba59-05a2a241c959)

##### ACCURACY
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/6c4868c9-2a68-4a79-97eb-e5d0adcb8612)

##### PREDICTIONS
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119657657/8a5080d9-9d35-4915-a51e-5eab0fa22360)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
