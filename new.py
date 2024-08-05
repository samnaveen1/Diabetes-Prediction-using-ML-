import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_lottie import st_lottie
import seaborn as sns
import matplotlib.pyplot as plt 
import time
import json

# Load the trained SVM model
loaded_model = pickle.load(open('C:/Users/Samnaveen.AB/Downloads/diabetes_model.sav', 'rb'))

# Predefined file path for the dataset
dataset_path = 'C:/Users/Samnaveen.AB/Downloads/diabetes_data.csv'

# Predefined features
x_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
y_feature = 'Outcome'

# Streamlit app
def main():  
    st.title('Diabetes Prediction')

    # Load dataset
    st.subheader('Dataset')
    df = pd.read_csv(dataset_path)
    st.write(df)
    st.success('Dataset loaded successfully!')

    # Load animation
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    lottie_codeing = load_lottiefile("C:/Users/Samnaveen.AB/Downloads/Animation - 1713103330944.json")
    st_lottie(lottie_codeing, speed=1, reverse=False, loop=True, quality="low", height=None, width=None, key=None)

    # Add a sidebar menu
    st.sidebar.title('Menu')
    menu = st.sidebar.radio('Navigation', ['Home', 'Predict'])

    if menu == 'Home':
        # Display correlation heatmap
        if st.sidebar.checkbox('Show Correlation Heatmap'):
            st.subheader('Correlation Heatmap')
            with st.spinner('Generating heatmap...'):
                time.sleep(2)
                fig, ax = plt.subplots()
                sns.heatmap(df[x_features + [y_feature]].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            st.success('Heatmap generated successfully!')

        # Visualize selected features
        if st.sidebar.button('Visualize'):
            with st.spinner('Generating visualization...'):
                time.sleep(2)
                visualize_features(x_features, y_feature, df)
            st.success('Visualization generated successfully!')

    elif menu == 'Predict':
        st.subheader('Input Features')
        input_features = {}
        for feature in x_features:
            input_features[feature] = st.number_input(feature, value=0)

        if st.button('Predict'):
            input_data = np.array([list(input_features.values())])
            prediction = loaded_model.predict(input_data)
            if prediction[0] == 0:
                st.write('The person is not diabetic')
            else:
                st.write('The person is diabetic')

# Function to visualize selected features
def visualize_features(x_features, y_feature, df):  
    fig, ax = plt.subplots(figsize=(8, 6))

    for feature in x_features:
        sns.scatterplot(x=feature, y=y_feature, data=df, ax=ax, label=feature)
    ax.set_title(f'Scatter Plot of {", ".join(x_features)} vs {y_feature}')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == '__main__':
    main()  
