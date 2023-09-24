# Fraud_Detection_ML_Model
 
This repository contains a fraud transaction detection model implemented in Python using the pandas library. 
The model aims to identify fraudulent transactions in a given dataset and provide insights for fraud prevention. 
This Model has an Accuracy of <b>99.8 %</b>
 
## Dataset 
 
The fraud transaction detection model uses a dataset containing the following columns: 
 
-  step : The time step when the transaction occurred. 
-  type : The type of transaction. 
-  amount : The transaction amount. 
-  nameOrig : The name of the origin account. 
-  oldbalanceOrg : The initial balance of the origin account before the transaction. 
-  newbalanceOrig : The new balance of the origin account after the transaction. 
-  nameDest : The name of the destination account. 
-  oldbalanceDest : The initial balance of the destination account before the transaction. 
-  newbalanceDest : The new balance of the destination account after the transaction. 
-  isFraud : Indicates whether the transaction is fraudulent (1 for fraud, 0 for non-fraud). 
-  isFlaggedFraud : Indicates whether the transaction is flagged as potential fraud (1 for flagged, 0 for not flagged). 
 
 
## Installation 
 
1. Clone the repository:
git clone <repository_url>
2. Install the required dependencies:
pip install pandas
    pip install scikit-learn
    pip install matplotlib
    pip install seaborn
3. Download the dataset: 
   - [Dataset Name] - [Download Link] 
 
## Model Usage 
 
1. Import the necessary libraries in your Python script:
python
import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns
2. Load the dataset into a pandas DataFrame:
python
df = pd.read_csv('dataset.csv')
3. Perform data preprocessing and feature engineering:
python
# Perform necessary data cleaning steps
    # Perform feature engineering to extract relevant features
4. Split the dataset into training and testing sets:
python
X = df.drop('fraud_label', axis=1)
    y = df['fraud_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
5. Train the fraud transaction detection model:
python
model = RandomForestClassifier()
    model.fit(X_train, y_train)
6. Evaluate the model performance:
python
y_pred = model.predict(X_test)
    # Perform necessary evaluation metrics (accuracy, precision, recall, etc.)
7. Visualize the results:
python
# Create visualizations to analyze the fraud detection performance
    # Use matplotlib and seaborn for plotting
8. Experiment with different parameters and techniques to improve the model's performance. 
 
