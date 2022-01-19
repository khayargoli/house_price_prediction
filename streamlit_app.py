# Importing Libraries

import streamlit as st
import pandas as pd
import numpy as np
import locale


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression # Linear Regression
from sklearn.ensemble import GradientBoostingRegressor
import pickle 
import streamlit as st

model_file=open("regression.pkl","rb")
regressor=pickle.load(model_file) # our model

# Defining Prediction Function

def predict_rating(model, df):
           # predictions_data = predict_model(estimator = model, data = df)
    return None


# Writing App Title and Description
st.set_page_config(layout="wide")
st.title('House Price Regressor Web App')
st.write('This is a web app to predict the price of house in Kathmandu based on several features of the house that you can see in the sidebar. Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the regressor.')


# Making Sliders and Feature Variables
st.sidebar.title('Feature Input')

area_in_squarefeet = st.sidebar.number_input(label = 'Area of house in Sq.Feet', min_value = 100.0,
                        max_value =  25000.00,
                        value = 2000.00,
                        step = 1.0)


number_of_floors  = st.sidebar.slider(label = 'Total number of Floor', min_value = 1.0,
                        max_value = 30.0 ,
                        value = 3.0,
                        step = 0.5)

number_of_bedroom = st.sidebar.slider(label = 'Total number of Bedroom', min_value = 1,
                        max_value = 50 ,
                        value = 5,
                        step = 1)

number_of_bathroom  = st.sidebar.slider(label = 'Total number of Bathroom', min_value = 1,
                        max_value = 20 ,
                        value = 5,
                        step = 1)



road_area_in_squarefeet  = st.sidebar.slider(label = 'Road width in Sq.Feet', min_value = 1.0,
                        max_value = 100.0 ,
                        value = 10.0,
                        step = 1.0)


   
face_dir_num = 0
face_dir = st.sidebar.radio("Direction Faced",('East', 'North', 'Nort East', 'North West', 'South', 'South East', 'South West', 'West'), key="face_dir")

if face_dir == 'East':
     face_dir_num=0;   
elif face_dir == 'North':
     face_dir_num=1;
elif face_dir == 'Nort East':
     face_dir_num=2;
elif face_dir == 'North West':
     face_dir_num=3;
elif face_dir == 'South':
     face_dir_num=4;
elif face_dir == 'South East':
     face_dir_num=5;
elif face_dir == 'South West':
     face_dir_num=6;        
else:
     face_dir_num=7;
    
         
# selected_city = 0
# city = st.sidebar.radio("City", ('Kathmandu', 'Bhaktapur', 'Lalitpur'), key="city")

# if city == 'Kathmandu':
#      selected_city = 1
# elif city == 'Bhaktapur':
#      selected_city = 0
# elif city == 'Lalitpur':
#      selected_city = 2
        
        
        
# selected_road_type = 0

# road_type = st.sidebar.radio("Road Type",('Blacktopped', 'Concrete', 'Gravelled', 'Paved', 'Soil Stabilized'), key="road_type")

# if road_type == 'Blacktopped':
#      selected_road_type=0;   
# elif road_type == 'Concrete':
#      selected_road_type=1;
# elif road_type == 'Gravelled':
#      selected_road_type=2;
# elif road_type == 'Paved':
#      selected_road_type=3;
# else:
#     selected_road_type=4;
    
    
    
# Mapping Feature Labels with Slider Values

features = {
  'Bedroom':number_of_bedroom,
  'Bathroom':number_of_bathroom,
  'Floors':number_of_floors,
  #'Year':built_year,
  'DirectionFaceOrdinal': face_dir_num,
  #'RoadTypeOrdinal': selected_road_type,
  #'CityOrdinal': selected_city,
  'SqFeetArea':area_in_squarefeet,
  'RoadWidth-Feet':road_area_in_squarefeet
}
    
    
# Converting Features into DataFrame

features_df  = pd.DataFrame([features])
st.dataframe(features_df.style.format(subset=['SqFeetArea', 'Floors', 'RoadWidth-Feet'], formatter="{:.2f}"))


# Predicting house price
predict_btn = st.button("Predict Price", key="predict_button")
    
def predict_chance(features):
    prediction=regressor.predict(features_df)
    return prediction

if predict_btn:
    prediction = predict_chance(features_df)
    locale.setlocale(locale.LC_MONETARY, 'en_IN')
    st.markdown('From the selected features of the house, the predicted price is: **'+ str(locale.currency(int(prediction), grouping=True)) + '**')
    
    
