import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.datasets import fetch_california_housing, load_iris
import streamlit as st
import numpy as np

# Caching the models to avoid reloading
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Fragment for saving models (if needed in other parts of the app)
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load datasets
@st.cache_data
def load_datasets():
    boston = fetch_california_housing()
    iris = load_iris()
    return boston, iris

boston, iris = load_datasets()

# Input fragment
def get_user_input(model_type):
    if model_type == 'Linear Regression':
        st.sidebar.write("Provide inputs for House Price Prediction (California Housing)")
        MedInc = st.sidebar.slider('Median Income', 0.0, 15.0, 3.0)
        HouseAge = st.sidebar.slider('House Age', 1.0, 52.0, 20.0)
        AveRooms = st.sidebar.slider('Average Rooms', 1.0, 20.0, 6.0)
        AveBedrms = st.sidebar.slider('Average Bedrooms', 0.5, 5.0, 1.0)
        Population = st.sidebar.slider('Population', 1.0, 35682.0, 1000.0)
        AveOccup = st.sidebar.slider('Average Occupants per Household', 0.5, 10.0, 3.0)
        Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0)
        Longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -120.0)
        return np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    elif model_type in ['Logistic Regression', 'Naive Bayes', 'Decision Tree']:
        st.sidebar.write("Provide inputs for Iris Classification")
        sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 5.0, 3.0)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)
        return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    elif model_type == 'Apriori':
        st.sidebar.write("Provide inputs for Apriori Association Rule Mining")
        product1 = st.sidebar.selectbox('Product1', [1, 0])
        product2 = st.sidebar.selectbox('Product2', [1, 0])
        product3 = st.sidebar.selectbox('Product3', [1, 0])
        product4 = st.sidebar.selectbox('Product4', [1, 0])
        product5 = st.sidebar.selectbox('Product5', [1, 0])
        return pd.DataFrame({'Product1': [product1], 'Product2': [product2], 'Product3': [product3], 
                             'Product4': [product4], 'Product5': [product5]})

# Fragment for model prediction
def load_model_and_predict():
    model_choice = st.sidebar.selectbox('Select Model', 
                                        ['Linear Regression', 'Logistic Regression', 
                                         'Naive Bayes', 'Apriori', 'Decision Tree'])

    placeholder = st.empty()

    if model_choice == 'Linear Regression':
        lin_reg_model = load_model('linear_reg_model.pkl')
        user_input = get_user_input('Linear Regression')
        prediction = lin_reg_model.predict(user_input)
        with placeholder.container():
            st.write(f"House Price Prediction: {prediction[0]:.2f}")

    elif model_choice == 'Logistic Regression':
        log_reg_model = load_model('logistic_reg_model.pkl')
        user_input = get_user_input('Logistic Regression')
        prediction = log_reg_model.predict(user_input)
        with placeholder.container():
            st.write(f"Iris Classification Prediction: {iris.target_names[prediction][0]}")

    elif model_choice == 'Naive Bayes':
        naive_bayes_model = load_model('naive_bayes_model.pkl')
        user_input = get_user_input('Naive Bayes')
        prediction = naive_bayes_model.predict(user_input)
        with placeholder.container():
            st.write(f"Naive Bayes Prediction: {prediction[0]}")

    elif model_choice == 'Apriori':
        st.write("Apriori Model - Association Rule Mining")
        user_input = get_user_input('Apriori')
        st.write("Your Transaction Input:")
        st.write(user_input)

        global transactions
        transactions = pd.DataFrame()  # Initialize if not already defined
        user_input = pd.DataFrame(user_input)  # Convert the list to a DataFrame
        transactions = pd.concat([transactions, user_input], ignore_index=True)
        transactions.fillna(0, inplace=True)
        transactions = transactions.astype(int)
        frequent_itemsets = apriori(transactions, min_support=0.5, use_colnames=True)
        if frequent_itemsets.empty:
            st.write("No frequent itemsets found with the current minimum support.")
        else:
            st.write("Frequent Itemsets:")
            st.write(frequent_itemsets)

            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            st.write("Association Rules:")
            st.write(rules)

    elif model_choice == 'Decision Tree':
        dec_tree_model = load_model('decision_tree_model.pkl')
        user_input = get_user_input('Decision Tree')
        prediction = dec_tree_model.predict(user_input)
        with placeholder.container():
            st.write(f"Iris Classification Prediction: {iris.target_names[prediction][0]}")

# Streamlit Title
st.title("Machine Learning Model App with Fragment-like Behavior")

# Call the prediction function
load_model_and_predict()
