import streamlit as st
import json
import pickle
from training import train_model, fetch_ohlc_data_since_last_entry, add_technical_indicators, predict

# Load your trained model (this should be done outside the main function to avoid reloading on each interaction)
trained_model = None

def load_model():
    global trained_model
    with open('../model/cryptopredictionmodel.pkl', 'rb') as file:
        trained_model = pickle.load(file)

def predict_latest_entry(pair, interval):
    global best_model_params_json
    # Ensure the model is loaded
    if trained_model is None:
        load_model()

    # Fetch the latest OHLC data
    ohlc_data = fetch_ohlc_data_since_last_entry(pair, interval, f'data/{pair}_{interval}.csv')

    # Convert JSON string to dictionary
    best_model_params = json.loads(best_model_params_json)

    # Extract preprocess__kw_args
    preprocess_kwargs = best_model_params["preprocess__kw_args"]

    # Add technical indicators based on the best parameters
    ohlc_data_with_indicators = add_technical_indicators(ohlc_data, **preprocess_kwargs)

    ohlc_data_with_indicators_new= ohlc_data_with_indicators.drop(columns=['number_of_trades', 'time'])

    # Predict using the trained model
    prediction = predict(ohlc_data_with_indicators_new, trained_model)


    return prediction

def main():
    global best_model_params_json
    st.title('Cryptocurrency Prediction Model Training')

    # Initialize session state variables if they don't exist
    if 'best_model_params_json' not in st.session_state:
        st.session_state['best_model_params_json'] = None

    if st.button('Retrain Model'):
        with st.spinner('Training in progress...'):
            model_score, best_model_params_json, train_accuracy, test_accuracy = train_model()
            load_model()  # Reload the model after retraining
            st.success('Training completed!')
            st.session_state['best_model_params_json'] = json.dumps(best_model_params_json)
            st.write(f"Best Score: {model_score}")
            st.json(best_model_params_json)
            st.write(f"Training accuracy: {train_accuracy}")
            st.write(f"Test accuracy: {test_accuracy}")

    # User inputs for the trading pair and interval
    pair = st.text_input('Enter the trading pair (e.g., XBTUSD)', 'LTCUSD')

    # Button for making a prediction
    if st.button('Predict Latest Entry'):
        best_model_params_json = st.session_state['best_model_params_json']
        if best_model_params_json is not None:
            with st.spinner('Fetching data and making prediction...'):
                latest_prediction = predict_latest_entry(pair, 1440)
                st.success('Prediction completed!')
                st.write('Will the next high be 2 per cent higher than the last close?  {}'.format(latest_prediction[0]))
        else:
            st.error('Please retrain the model to get the best parameters before making a prediction.')

if __name__ == "__main__":
    main()