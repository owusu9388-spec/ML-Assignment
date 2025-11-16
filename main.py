import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from logger import setup_logger



logger = setup_logger("")
logger.warning("Starting the Script")

# Loading data 
data_df = pd.read_csv("heart.csv")
logger.success(data_df.head())  

'''
Following the rules of Machine Learning and Algorithm Development, we will split the data into training, validation, and test sets.
1. Data discretization
2. Data cleaning 
3. Data integration
4. Data transformation
5. Data reduction
'''


# Discretizing and Cleaning the data
'''Check the shape of the data'''
logger.info(f"Data Shape: {data_df.shape}")
logger.info(f"Data Info: {data_df.info()}")
logger.info(f"Missing Values: {data_df.isnull().sum()}\n")
logger.info(f"Duplicate Rows: {data_df.duplicated().sum()}")
print(f"*"*100)

# Identifying unnecessary values
for i in data_df.select_dtypes(include="object").columns:
    logger.success(f"Garbage values in '{i}': {data_df[i].value_counts()}")
    logger.info(f"Unique values in '{i}': {data_df[i].unique()}")

# Exploratory Data Analysis (EDA) 
logger.success(f"Exploratory data: \n{data_df.describe().T}")

# Histogram to understand the distribution of data 
"""for i in data_df.select_dtypes(include="number").columns:
    sns.histplot(data=data_df, x=i)
    plt.show()
    

# Identifying outliers 
for i in data_df.select_dtypes(include="number").columns:
    sns.boxplot(data=data_df, x=i)
    plt.show()



# scatterplot to understand the relationship
for i in data_df.columns:
    sns.scatterplot(data=data_df, x=i)
    plt.show()
    """

# Correlation with heatmap to interpret the relation and multicolliniarity
s=data_df.select_dtypes(include="number").corr
plt.figure(figsize=(15, 15))
sns.heatmap(s, annot=True)

# Missing Value Treatment
"""Choose the method of inputting missing value
like mean, median, mode, or KNNinputer """


for i in []:
    data_df[i].fillna(data_df[i].median(), inplace=True)

from sklearn.immpute import KNNI
impute=KNNImpuuter()
for i in data_df.select_dtypes(include="number").columns:
    df[i]=impute.fit_transform(data_df[i])
data_df.isnull().sum()
"""The KNN imputer imputes all missing values.
How? So it would impute the average value of its nearest neighbour and it would fill in the missing value with the average of the nearest neighbour"""