import requests
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Any
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import pickle


def fetch_ohlc_data_since_last_entry(pair: str, interval: int, csv_filepath: str):
    """
    Fetches OHLC data from an API and merges it with existing data in a CSV file.

    Parameters
    ----------
    pair : str
        The trading pair to fetch data for.
    interval : int
        The time interval for the data.
    csv_filepath : str
        The filepath of the CSV file to read and write data.

    Returns
    -------
    pd.DataFrame
        The combined DataFrame containing the merged data.

    Notes
    -----
    - The function reads the existing CSV file and checks if it has the expected number of columns.
    - If the CSV file does not exist or has unexpected number of columns, it starts from scratch.
    - The function fetches new data from the API using the provided pair, interval, and last entry time.
    - The new data is merged with the existing data and duplicates are removed.
    - The combined DataFrame is saved back to the CSV file.
    """
    column_names = ['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'number_of_trades']
    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'number_of_trades']
    last_entry_time = 0  # Initialize last_entry_time

    # Attempt to read the existing CSV file
    try:
        df = pd.read_csv(csv_filepath, names=column_names, header=None)
        # Check if the CSV has the expected number of columns to avoid treating header as data
        if len(df.columns) == len(column_names):
            df.columns = column_names
        else:
            raise ValueError("CSV file has unexpected number of columns.")

        # Ensure 'time' is in datetime format
        last_entry_time = df['time'].astype('int64')
        last_entry_time = last_entry_time.max()
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError) as e:
        df = pd.DataFrame(columns=column_names)

    # Fetch new data from the API
    params = {
        'pair': pair,
        'interval': interval,
        'since': last_entry_time
    }
    resp = requests.get('https://api.kraken.com/0/public/OHLC', params=params)

    if resp.status_code == 200:
        data = resp.json()
        ohlc_data = data['result'][list(data['result'].keys())[0]]
        new_df = pd.DataFrame(ohlc_data, columns=column_names)
        new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Concatenate the old and new data
        combined_df = pd.concat([df, new_df], ignore_index=True)

        # Remove duplicates, keeping the last occurrence
        combined_df.drop_duplicates(subset=['time'], keep='last', inplace=True)

        # Save the combined DataFrame back to the CSV file
        combined_df.to_csv(csv_filepath, index=False, header=False)

        combined_df['time'] = pd.to_datetime(combined_df['time'], unit='s')

        return combined_df
    else:
        return df   

def add_technical_indicators(X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs: Any):
    """
    Add technical indicators as features to the input DataFrame.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        The input DataFrame or array.
    y : Optional[Union[pd.Series, np.ndarray]], optional
        The target variable, by default None.
    **kwargs : Any
        Additional keyword arguments for calculating the technical indicators.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new features.
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X).copy()
    else:
        X = X.copy()

    # Calculate each technical indicator using the provided periods
    X[f'SMA_{kwargs["sma_period"]}'] = calculate_sma(X['close'], kwargs['sma_period'])
    X[f'EMA_{kwargs["ema_period"]}'] = calculate_ema(X['close'], kwargs['ema_period'])
    X[f'RSI_{kwargs["rsi_period"]}'] = calculate_rsi(X['close'], kwargs['rsi_period'])
    macd, signal = calculate_macd(X['close'], kwargs['macd_fast_period'], kwargs['macd_slow_period'], kwargs['macd_signal_period'])
    X['MACD'] = macd
    X['MACD_Signal'] = signal
    upper_band, lower_band = calculate_bollinger_bands(X['close'], kwargs['bollinger_period'], kwargs['bollinger_std_dev'])
    X['Bollinger_Upper'] = upper_band
    X['Bollinger_Lower'] = lower_band

    # Return the DataFrame with the new features
    return X

def calculate_sma(data: pd.Series, period: int):
    """
    Calculate the Simple Moving Average (SMA) of a given data series.

    Parameters
    ----------
    data : pd.Series
        The data series for which to calculate the SMA.
    period : int
        The period over which to calculate the SMA.

    Returns
    -------
    pd.Series
        The SMA of the data series.
    """
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int):
    """
    Calculate the Exponential Moving Average (EMA) of a given data series.

    Parameters
    ----------
    data : pd.Series
        The input data series.
    period : int
        The period for calculating the EMA.

    Returns
    -------
    pd.Series
        The EMA of the input data series.
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int):
    """
    Calculate the Relative Strength Index (RSI) for a given data series.

    Parameters:
    data (pd.Series): The data series for which to calculate RSI.
    period (int): The number of periods to use for the RSI calculation.

    Returns:
    pd.Series: The RSI values for the given data series.
    """
    delta = data.diff(1)  # Calculate the difference between consecutive values
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Calculate the average gain over the specified period
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Calculate the average loss over the specified period
    rs = gain / loss  # Calculate the relative strength
    rsi = 100 - (100 / (1 + rs))  # Calculate the RSI
    return rsi

def calculate_macd(data: List[float], fast_period: int, slow_period: int, signal_period: int):
    """
    Calculate the MACD (Moving Average Convergence Divergence) line and signal line.

    Parameters:
    - data (List[float]): The input data.
    - fast_period (int): The period for the fast EMA (Exponential Moving Average).
    - slow_period (int): The period for the slow EMA (Exponential Moving Average).
    - signal_period (int): The period for the signal line EMA (Exponential Moving Average).

    Returns:
    - Tuple[List[float], List[float]]: A tuple containing the MACD line and signal line.

    """
    fast_ema = calculate_ema(data, fast_period)  # Calculate the fast EMA
    slow_ema = calculate_ema(data, slow_period)  # Calculate the slow EMA
    macd_line = fast_ema - slow_ema  # Calculate the MACD line
    signal_line = calculate_ema(macd_line, signal_period)  # Calculate the signal line
    return macd_line, signal_line

def calculate_bollinger_bands(data: pd.Series, period: int, std_dev: int):
    """
    Calculate the upper and lower Bollinger Bands.

    Parameters:
    data (pd.Series): The input data.
    period (int): The period for calculating the moving average and standard deviation.
    std_dev (int): The number of standard deviations to use for calculating the bands.

    Returns:
    Tuple[pd.Series, pd.Series]: A tuple containing the upper and lower Bollinger Bands.
    """
    sma = data.rolling(window=period).mean()  # Calculate the simple moving average
    rolling_std = data.rolling(window=period).std()  # Calculate the rolling standard deviation
    upper_band = sma + (rolling_std * std_dev)  # Calculate the upper Bollinger Band
    lower_band = sma - (rolling_std * std_dev)  # Calculate the lower Bollinger Band
    return upper_band, lower_band

def train_model():
    """
    Trains a model to predict cryptocurrency price movements.

    Returns:
        model_score (float): The best score achieved by the model.
        best_model_params_json (Dict[str, Any]): The best parameters found by RandomizedSearchCV.
        train_accuracy (float): The accuracy of the model on the training data.
        test_accuracy (float): The accuracy of the model on the test/validation data.
    """
    # Update BTC data
    btc_df = fetch_ohlc_data_since_last_entry('XBTUSD', 1440, 'data/XBTUSD_1440.csv')

    # Update ETH data
    eth_df = fetch_ohlc_data_since_last_entry('ETHUSD', 1440, 'data/ETHUSD_1440.csv')

    # Now both btc_df and eth_df have identical 'time' columns and their own data
    # Rename the columns to indicate their source, except for the 'time' column
    btc_df = btc_df.rename(columns=lambda x: x if x == 'time' else f"btc_{x}")
    eth_df = eth_df.rename(columns=lambda x: x if x == 'time' else f"eth_{x}")

    # Merge the dataframes on the 'time' column
    merged_df = pd.merge(btc_df, eth_df, on='time')

    # Now, split the merged dataframe back into two original dataframes
    # by selecting the columns based on the prefix you added
    btc_columns = [col for col in merged_df if col.startswith('btc_')] + ['time']
    eth_columns = [col for col in merged_df if col.startswith('eth_')] + ['time']

    btc_df_original = merged_df[btc_columns].rename(columns=lambda x: x.replace('btc_', ''))
    eth_df_original = merged_df[eth_columns].rename(columns=lambda x: x.replace('eth_', ''))

    # Set 'time' as the index for both DataFrames
    btc_df_original.set_index('time', inplace=True)
    eth_df_original.set_index('time', inplace=True)

    # Join the two DataFrames on the 'time' index
    # Use an inner join to only keep rows that appear in both DataFrames
    combined_df = btc_df_original.join(eth_df_original, lsuffix='_btc', rsuffix='_eth', how='inner')

    # Calculate the correlation matrix for the joined DataFrame
    correlation_matrix = combined_df.corr()

    # If you want to see the correlation between the 'close' prices of BTC and ETH, for example
    high_correlation = correlation_matrix.loc['high_btc', 'high_eth']
    close_correlation = correlation_matrix.loc['close_btc', 'close_eth']

    # For btc_df
    btc_df_original['next_high'] = btc_df_original['high'].shift(-1)
    btc_df_original['next_high_2perc'] = (btc_df_original['next_high'] - btc_df_original['close']) / btc_df_original['close'] > 0.02

    # For eth_df
    eth_df_original['next_high'] = eth_df_original['high'].shift(-1)
    eth_df_original['next_high_2perc'] = (eth_df_original['next_high'] - eth_df_original['close']) / eth_df_original['close'] > 0.02

    # Drop rows with missing values in the training set
    train_df = btc_df_original.dropna(subset=['next_high','next_high_2perc'])
    X_train = train_df[['open', 'high', 'low', 'close', 'vwap', 'volume']]
    y_train = train_df['next_high_2perc']

    # Drop rows with missing values in the testing/validation set
    test_df = eth_df_original.dropna(subset=['next_high','next_high_2perc'])
    X_test = test_df[['open', 'high', 'low', 'close', 'vwap', 'volume']]
    y_test = test_df['next_high_2perc']

    # Define the pipeline with the FunctionTransformer and imputation
    pipeline_with_transformer = Pipeline([
        ('preprocess', FunctionTransformer(add_technical_indicators, validate=False)),  # No kw_args here
        ('impute', SimpleImputer(strategy='mean')),  # Impute missing values
        ('model', RandomForestClassifier(random_state=42))
    ])

    # Define the parameter distributions for RandomizedSearchCV
    param_distributions = {
        'preprocess__kw_args': [
            {
                'sma_period': randint(5, 15).rvs(),
                'ema_period': randint(5, 15).rvs(),
                'rsi_period': randint(10, 20).rvs(),
                'macd_fast_period': randint(8, 17).rvs(),
                'macd_slow_period': randint(19, 35).rvs(),
                'macd_signal_period': randint(5, 15).rvs(),
                'bollinger_period': randint(10, 30).rvs(),
                'bollinger_std_dev': randint(1, 3).rvs()
            }
            for _ in range(100)  # Generate 5 random parameter sets
        ],
        'model__n_estimators': randint(100, 1000),
        'model__max_depth': randint(3, 10),
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 20),
        'model__max_features': [1.0, 'sqrt', 'log2', None]
    }
    # Define a custom scorer
    scorer = make_scorer(accuracy_score)

    # Prepare the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline_with_transformer,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=5,        # 5-fold cross-validation
        verbose=1,   # Controls the verbosity: the higher, the more messages
        random_state=42,
        n_jobs=1,   # Number of jobs to run in parallel (-1 means using all processors)
        scoring=scorer  # Use the custom scorer
    )

    # Run the random search
    random_search.fit(X_train, y_train)

    # Output the best parameters and score found by RandomizedSearchCV
    best_model_params_json = random_search.best_params_
    model_score = random_search.best_score_

    # Extract the best pipeline
    best_pipeline = random_search.best_estimator_

    # Apply transformations to X_test
    X_test_transformed = best_pipeline.named_steps['preprocess'].transform(X_test)

    # Now X_test_transformed contains the technical indicators with the best parameters

    # Predict on the training data
    train_predictions = random_search.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)


    # Evaluate the best model on the test/validation data
    y_pred = random_search.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Classification report (includes precision, recall, and F1-score)
    class_report = classification_report(y_test, y_pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    with open('../model/cryptopredictionmodel.pkl', 'wb') as file:
        pickle.dump(random_search, file)

    return model_score, best_model_params_json, train_accuracy, test_accuracy

def predict(data: pd.DataFrame, model: BaseEstimator):
    """
    Predict outcomes based on the input data using the provided model.

    Parameters:
    - data (pd.DataFrame): The input data for making predictions. 
                            It should be in the same format that the model was trained on.
    - model (BaseEstimator): The trained model used to make predictions.

    Returns:
    - pd.Series: The predictions made by the model.
    """
    prediction = model.predict(data)
    return prediction
