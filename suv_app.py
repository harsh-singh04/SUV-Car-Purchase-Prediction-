import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#load data
suv_car_df = pd.read_csv('suv_data.csv')

#Split data into train and test sets
X = suv_car_df.iloc[:,[2,3]]
y = suv_car_df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit the model
model = LogisticRegression()
model.fit(X_train,y_train)
test_y_pred = model.predict(X_test)

# web app
def app():
    #Load image
    image_path = 'Suv car.jpg'
    #display image centered with fixed size of 300*300
    with st.container():
        st.image(image_path, width=350)
    st.title("🚗SUV Car Purchasing Prediction")
    st.write('This app predicts whether a customer will purchase an SUV car based on their age and salary.')

    #sidebar
    age = st.sidebar.slider("Select Age", min_value=18, max_value=100, step=1, value=30)

    salary = st.sidebar.slider("Select salary", min_value=10000, max_value=2000000, step=1000, value=50000)
    
    #Make Prediction
    X_new = [[age, salary]]
    X_new_scaled = sc.transform(X_new)
    y_new = model.predict(X_new_scaled)

    if y_new ==1:
        st.write("✅This Person has bought SUV Car")
    else:
        st.write("❌This Person has not bought SUV Car")
        
        # Show prediction probability
        proba = model.predict_proba(X_new_scaled)[0][1]
        st.metric("Purchase Probability", f"{proba:.2%}")



#Run the app
if __name__== "__main__":
    app()
    