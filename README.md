# Crypto Currency Prediction Model
## Business Understanding
### Objective
The primary objective of this project is to develop a machine learning model capable of predicting the future prices of cryptocurrencies, specifically Bitcoin (BTC-USD) and Ethereum (ETH-USD).

### Importance
Accurate price predictions can offer substantial advantages for investors, portfolio managers, risk managers, and cryptocurrency exchanges.

### Challenges
The high volatility of the cryptocurrency market influenced by various factors makes the task of price prediction challenging.

### Project Relevance
This project aims to provide better financial tools in the crypto space, potentially serving as a foundation for more sophisticated trading algorithms and investment strategies.

### Stakeholders
Cryptocurrency Traders and Investors
Financial Analysts
Fintech Companies
Data Scientists
## Data Understanding
### Import Relevant Packages
```
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import List, Union, Optional, Any
from scipy.stats import randint
from sklearn.model_selection import (
                                        cross_val_score,
                                        GridSearchCV,
                                        RandomizedSearchCV
                                    )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
                                make_scorer,
                                accuracy_score,
                                precision_score,
                                recall_score,
                                f1_score,
                                classification_report,
                                confusion_matrix
                            )
```

### Define Functions
Several functions are defined to fetch data, add technical indicators, and calculate various statistical measures.

### Access the Data
Data is fetched and updated for both BTC and ETH using the defined functions.

### Explore the Data
Data exploration includes statistical summaries and visualizations.

## Data Preparation
### Create New Columns
New columns are created to indicate whether the price will increase by a certain percentage.

### Create Test and Training Data
Data is split into training and testing sets.

## Categorical Analysis
### Random Forest
A Random Forest classifier is used for the initial model, which is then optimized using Grid Search.

## Model Optimization and Evaluation
### Trying to Optimize Random Forest
Parameters of the Random Forest are tuned to reduce overfitting and improve model performance.

### Executing Grid Search on Random Forest
Grid Search is performed to find the best hyperparameters for the Random Forest model.

### Create New Features and Further Optimization
A pipeline is created to add technical indicators and impute missing values. Randomized Search is used to optimize the model further.

## Model Persistence
### Save the Model
The final model is saved using pickle for later use.
```
with open('model/cryptopredictionmodel.pkl', 'wb') as file:
    pickle.dump(random_search, file)
```
## Instructions for Running the Streamlit App
### Prerequisites
Python 3.8 or above
pip (Python package installer)
### Setup
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
4. Install the required dependencies:
```
pip install -r requirements.txt
```
### Running the App
Navigate to the app subfolder:
```
cd app
```
### Start the Streamlit app:
```
streamlit run webapp.py
```

The app should now be running on your local server, typically at http://localhost:8501.
