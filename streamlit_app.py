# Importing Libraries

import streamlit as st
import pandas as pd
import numpy as np
import locale


from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle 
import streamlit as st





# Writing App Title and Description
st.set_page_config(layout="wide")
st.title('House Price Regressor Web App')
st.write('This is a web app to predict the price of house in Kathmandu based on several features of the house that you can see in the sidebar. Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the regressor.')


# Making Sliders and Feature Variables
st.sidebar.title('Feature Input')
#estimator = st.sidebar.selectbox(
#     'Choose the regression model',
#     ('GradientBoostingRegressor', 'LinearRegression', 'Lasso', 'Ridge', 'Random Forest'))

model_file=open("regression.pkl","rb")
regressor=pickle.load(model_file) # our model
st.write('Regression model in use: ', regressor)
area_in_squarefeet = st.sidebar.number_input(label = 'Area of house in Sq.Feet', min_value = 1000.0,
                        max_value =  2500.00,
                        value = 1200.00,
                        step = 1.0)


number_of_floors  = st.sidebar.slider(label = 'Total number of Floor', min_value = 1.0,
                        max_value = 15.0 ,
                        value = 3.0,
                        step = 0.5)

number_of_bedroom = st.sidebar.slider(label = 'Total number of Bedroom', min_value = 1,
                        max_value = 20 ,
                        value = 5,
                        step = 1)

number_of_bathroom  = st.sidebar.slider(label = 'Total number of Bathroom', min_value = 1,
                        max_value = 15,
                        value = 5,
                        step = 1)



road_area_in_feet  = st.sidebar.slider(label = 'Road width in Sq.Feet', min_value = 3.0,
                        max_value = 40.0 ,
                        value = 10.0,
                        step = 1.0)

built_year  = st.sidebar.slider(label = 'Built year', min_value = 2001,
                        max_value = 2077 ,
                        value = 2070,
                        step = 1)
   
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
    
         
selected_city = 0
city = st.sidebar.radio("City", ('Kathmandu', 'Bhaktapur', 'Lalitpur'), key="city")

if city == 'Kathmandu':
     selected_city = 1
elif city == 'Bhaktapur':
     selected_city = 0
elif city == 'Lalitpur':
     selected_city = 2
        
        
        
selected_road_type = 0

road_type = st.sidebar.radio("Road Type",('Blacktopped', 'Concrete', 'Gravelled', 'Paved', 'Soil Stabilized'), key="road_type")



if road_type == 'Blacktopped':
     selected_road_type=0;   
elif road_type == 'Concrete':
     selected_road_type=1;
elif road_type == 'Gravelled':
     selected_road_type=2;
elif road_type == 'Paved':
     selected_road_type=3;
else:
    selected_road_type=4;
    
    
# st.sidebar.markdown('Select Amenities')
# hasParking= st.sidebar.checkbox('Parking')
# hasLawn= st.sidebar.checkbox('Lawn')
# hasDrainage= st.sidebar.checkbox('Drainage')
# hasGarage= st.sidebar.checkbox('Garage')
# hasAirCondition= st.sidebar.checkbox('Air Condition')
# hasBalcony= st.sidebar.checkbox('Balcony')
# hasDeck= st.sidebar.checkbox('Deck')
# hasFencing= st.sidebar.checkbox('Fencing')
# hasWaterSupply= st.sidebar.checkbox('Water Supply')
# hasGarden= st.sidebar.checkbox('Garden')
# hasCCTV= st.sidebar.checkbox('CCTV')
# hasModularKitchen= st.sidebar.checkbox('Modular Kitchen')
# hasSolarWater= st.sidebar.checkbox('Solar Water')
# hasWaterWell= st.sidebar.checkbox('Water Well')
# hasWaterTank= st.sidebar.checkbox('Water Tank')
# hasKidsPlayground = st.sidebar.checkbox('Kids Playground')
# hasLift = st.sidebar.checkbox('Lift')
    
# Mapping Feature Labels with Slider Values
#'Bedroom', 'Bathroom', 'Floors', 'Year', 'DirectionFaceOrdinal', 'CityOrdinal', 'RoadTypeOrdinal', 'SqFeetArea', 'RoadWidth-Feet

features = {
    'Bedroom':number_of_bedroom,
    'Bathroom':number_of_bathroom,
    'Floors':number_of_floors,
    'Year':built_year,       
    'DirectionFaceOrdinal': face_dir_num,
    'CityOrdinal': selected_city,
    'RoadTypeOrdinal': selected_road_type,
    'SqFeetArea':area_in_squarefeet,
    'RoadWidth-Feet':road_area_in_feet,
}
    
    # 'RoadWidth-Feet':road_area_in_feet,
    # 'Parking': 1 if hasParking else 0,
    # 'Lawn': 1 if hasLawn else 0,
    # 'Drainage': 1 if hasDrainage else 0,
    # 'Garage': 1 if hasGarage else 0,
    # 'Air Condition': 1 if hasAirCondition else 0,
    # 'Balcony': 1 if hasBalcony else 0,
    # 'Deck': 1 if hasDeck else 0,
    # 'Fencing' : 1 if hasFencing else 0,
    # 'Water Supply' : 1 if hasWaterSupply else 0,
    # 'Garden' : 1 if hasGarden else 0,
    # 'CCTV': 1 if hasCCTV else 0,
    # 'Modular Kitchen': 1 if hasModularKitchen else 0,
    # 'Solar Water' :1 if hasSolarWater else 0,
    # 'Water Well' :1 if hasWaterWell else 0,
    # 'Water Tank' : 1 if hasWaterTank else 0,
    # 'Kids Playground' : 1 if hasKidsPlayground else 0,
    # 'Lift' : 1 if hasLift else 0 
    
# Converting Features into DataFrame

features_df  = pd.DataFrame([features])
st.dataframe(features_df.style.format(subset=['SqFeetArea', 'Floors', 'RoadWidth-Feet'], formatter="{:.2f}"), width = 1800)


# Predicting house price
predict_btn = st.button("Predict Price", key="predict_button")
    
def formatNPR(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

def predict_chance(features): 
    prediction=regressor.predict(features)
    return prediction

if predict_btn:
    prediction = predict_chance(features_df)
    st.markdown('From the selected features of the house, the predicted price is: ** Rs. '+ str(formatNPR(int(prediction))) + '**')
    
    
    
    
