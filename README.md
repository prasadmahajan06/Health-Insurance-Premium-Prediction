# Health-Insurance-Premium-Prediction
Health Insurance Premium Prediction Using Linear Regression

🏥 ***Health Insurance Premium Prediction using Linear Regression***
This project builds a Linear Regression model to predict health insurance premiums based on customer attributes such as age, BMI, smoking habits, number of dependents, and region.
The goal is to understand how different features affect insurance charges and to create a reliable predictive system for insurance companies and healthcare analytics.

🚀 ***Project Overview***
- The objective of this project is to:
- Explore customer demographic and medical factors
- Understand their relationship with insurance premiums
- Build a regression model to predict insurance costs
- Evaluate model performance using key regression metrics
- Provide insights that can help insurance companies design fair and data-driven premium strategies
- This project uses a widely available Medical Insurance Charges Dataset for educational and analytical purposes.

📂 ***Dataset Description***
The dataset commonly includes the following features:

- **Feature**      **-Description**
- age	         -Age of the individual
- sex	         -Male/Female
- bmi	         -Body Mass Index
- children	   -Number of dependents
- smoker	     -Whether the person is a smoker
- region	     -Residential region (northwest, southeast, etc.)
- charges	     -Target variable – medical insurance premium

🗂️**Data Source**
- Primary Source: Health Insurance Premium Prediction <a href ="">Download Dataset Here</a>

🧰 ***Technologies & Libraries Used***
- Python 3.x
- Pandas – data analysis
- NumPy – numerical operations
- Matplotlib / Seaborn – visualizations
- Scikit-Learn – Linear Regression model

🛠️ Steps Performed in the Project
✔ 1. Data Exploration
- Summary statistics
- Distribution analysis (age, BMI, charges)
- Relationship analysis (smoker vs charges, BMI vs charges, etc.)

✔ 2. Data Preprocessing
- Handling missing values
- Encoding categorical features (sex, smoker, region)
- Detecting/skipping outliers

✔ 3. Model Building
- Train/test split
- Applying Linear Regression
- Fitting data to the model
- Checking model coefficients (important for interpretation)

✔ 4. Model Evaluation
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Residual analysis

Visualizing predictions vs. actual values
✔ 5. Making Predictions
- Predict insurance charges for new individuals using model input features

⭐ ***Linear Regression Model Code***

### Mounting Drive
from google.colab import drive

drive.mount('/content/drive')

### Improrting Libraries
import pandas as pd # data processing

import numpy as np # numerical computing

import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

from sklearn.linear_model import LinearRegression # Import Linear Regression Model

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score # Import evaluation metrics for Linear Regression model

filepath = '/content/drive/MyDrive/mediclaim.csv'

df = pd.read_csv(filepath)

df.head()

df.info()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace = True)

#### Convert the categorical data "smoker" to numeric value using label encoder
df['smoker'] = df['smoker'].astype(str).str.strip().str.lower()

df['smoker'] = df['smoker'].map({'yes':1, 'no':0})

df['sex'] = df['sex'].map({'male':1, 'female':0})

df.head(20)

df.describe()

### No outlier in age column.
ax = sns.boxplot(df['age'])

ax.set_title('Dispersion of Age')

plt.show(ax)

### '''To ensure there are no outliers for bmi by box plot. But, there are few bmi values above 47 which can be considered as outliers'''
ax = sns.boxplot(df['bmi'])

ax.set_title("Dispersion of bmi")

plt.show(ax)

### Dispersion of Charges
ax = sns.boxplot(df['charges'])

ax.set_title("Dispersion of Charges")

plt.show(ax)

### '''The scatter plot is not representing, when Age is increasing bmi is also increasing.Few data points of bmi is high at younger age compare to other data points. Those data points can be consider as outliers'''
ax = sns.scatterplot(x = 'age', y = 'bmi', data=df)

ax.set_title('Age vs BMI')

plt.show()

### To understand the relationship between the Age and charges with respect to bmi.
#Scatter plot clearly states that, when age is increasing charges also increasing but has three different

#groups of charges irrespective of bmi. Hence, BMI is not influencing the charges with Age.

plt.figure(figsize=(14,9))

ax = sns.scatterplot(x='age',y='charges',hue = 'bmi',size = 'bmi', data=df)

ax = ax.set_title("Age vs Charges by BMI")

plt.xlabel("Age")

plt.ylabel("Charges")

plt.show(ax)

### Scatter plot clearly states that, Age with sex are not influencing the Charges.
plt.figure(figsize=(10,7))

ax = sns.scatterplot(x='age',y='charges', hue='sex',style = 'sex',data=df)

ax.set_title("Age vs charges by Sex")

plt.show(ax)

### Both Age and smoker are highly influencing the charges. Smoker yes
plt.figure(figsize=(10,7))

ax = sns.scatterplot(x='age',y='charges', hue=df['smoker'],style = df['smoker'],size = df['smoker'], data=df)

ax.set_title("Age vs Charges by Smoker")

plt.xlabel("Smoker (Yes - 1, No - 0)")

plt.ylabel("Charges")

plt.show(ax)

### To understand the relationship of each independent variable with dependent variable.

#Age has positive side (30%) relationship against expenses

#bmi has positive side (20%) relationship against expenses

#Children has almost no relationship against expenses

#Smoker has strong positive relationship (78%) against expenses

#sex has no relationship against expenses

df.corr(numeric_only=True)

### Swarm plot shows how smoker feature is influencing the expeneses compare with smoker and non-smoker
ax = sns.swarmplot(x='smoker',y='charges', hue= 'smoker', data=df)

ax.set_title("Smoker vs Charges")

plt.xlabel("Smoker (Yes - 1, No - 0)")

plt.ylabel("Charges")

plt.show(ax)

#### These three features have relationship with charges.
x = df[['age','bmi','smoker']]

y = df['charges']

#train_test_split() to split the dataset into train and test set at random.

#test size data set should be 30% data

X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Creating an linear regression model object

model = LinearRegression()

#Training the model using training data set

model.fit(X_train, Y_train)

#X_train_predict = model.predict(X_train)

#X_test_predict = model.predict(X_test)

### Printing intercept and co-efficients
print("Intercept value:", model.intercept_)

print("Coefficient values:", model.coef_)

coef_df = pd.DataFrame(list(zip(X_train.columns,model.coef_)), columns = ['Features','Predicted Coeff'])

coef_df

#print("Features train data:\n",X_train.smoker)

### Predicting the Y value from the train set and test set.
Y_train_predict = model.predict(X_train)

Y_train_predict[0:5]

Y_test_predict = model.predict(X_test)

### Plot to see the actual expenses and predicted expenses from Train data set
ax = sns.scatterplot(x=Y_train, y=Y_train_predict)

ax.set_title("Actual Expenses vs Predicted Expenses")

plt.xlabel("Actual Expenses")

plt.ylabel("Predicted Expenses")

plt.show()

#### Train and predict the Y_train for the feature 'smoker'
smoker_model = LinearRegression()

smoker_model.fit(X_train[['smoker']], Y_train)

print("intercept:",smoker_model.intercept_, "coeff:", smoker_model.coef_)

#print("Train - Mean squared error:", np.mean((Y_train - model.predict(X_train)) ** 2))

smoker_df = pd.DataFrame(list(zip(Y_train, smoker_model.predict(X_train[['smoker']]))), columns = ['Actual Expenses','Predicted Expenses'])

smoker_df.head()

#X_train['smoker'].shape

### MSE for Train data set
print("MSE:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_train,smoker_model.predict(X_train[['smoker']]))))

### R-Squared value for Train data set
print("R-squared value:",round(r2_score(Y_train, Y_train_predict),3))

print("R-squared value only for smoker:", round(r2_score(Y_train,smoker_model.predict(X_train[['smoker']]))),3)

### Mean absolute error for Train data set
print("Mean absolute error:",mean_absolute_error(Y_train, Y_train_predict))

print("Mean absolute Error only for Smoker:", mean_absolute_error(Y_train,smoker_model.predict(X_train[['smoker']])))

print("MSE for Test data set")

print("MSE:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_test,smoker_model.predict(X_test[['smoker']]))))



📊 ***Example Insights (Replace with your findings)***
- Smokers have significantly higher insurance charges than non-smokers
- BMI is positively correlated with insurance cost
- Premium increases consistently with age
- Sex does not have significant impact on cost/charges
