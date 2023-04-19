# Adding the dataset
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\ilesannmi\Desktop\Projectssss___\House Rent+ Deployment\House_Rent_Dataset.csv')
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


df_FT = df ## Backup

# Feature selection
## Just wanna work with 4 columns
feat = ['BHK', 'City', 'Bathroom', 'Size']
y = 'Rent'
features = df_FT[feat] 
target = df_FT[y].values

## So i have FEATURES and TARGET



### Encoding the city column alone
City_enc = features[['City']].apply(enc.fit_transform)

#adding the city column with the encoded column
features['city_enc'] = City_enc
features2 = features.drop('City', axis= 1) #dropping the city column

# FEATURES2 and TARGET now
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features2, target, random_state = 15)

#Random Forest
from sklearn.ensemble import RandomForestRegressor

def RandomFR_model(x_train, x_test, y_train, y_test):
    regr = RandomForestRegressor(max_depth=3, random_state=25)
    regr.fit(x_train, y_train)

    print('training accuracy: ', regr.score(x_train, y_train))

    print('testing accuracy: ', regr.score(x_test, y_test))
    return regr

# pred = regr.predict(x_test)
# print(pred)
FT = RandomFR_model(x_train, x_test, y_train, y_test)

#saving model to pickle
import pickle

pickle_out = open('modelmodel2.pkl', mode = 'wb')
pickle.dump(FT, pickle_out)
pickle_out.close()

#Using model
pickle_in = open('modelmodel2.pkl', 'rb')
clf = pickle.load(pickle_in)


###############################################################################################################
import pickle
import streamlit as st


pickle_in = open('modelmodel2.pkl', 'rb')

clf = pickle.load(pickle_in)


@st.cache()

def make_prediction(bhk, Bathroom, Size, City):
    if City == 'Mumbai':
        city_enc = 5
    elif City == 'Chennai':
        city_enc = 1
    elif City == 'Bangalore':
        city_enc = 0
    elif City == 'Hyderabad':
        city_enc = 3
    elif City == 'Delhi':
        city_enc = 2
    elif City == 'Kolkata':
        city_enc = 4
    
    prediction = clf.predict([[bhk, Bathroom, Size, city_enc]])[0]
    return prediction


def main():
    #front end elements
    html_temp = """
    <div style = "background-color:green;padding:13px">
    <h1 style = "color:blue;text-align:center;"> House Prices Prediction by Ilesanmi </h1>
    </div>
    """

    #front end 
    #st.markdown("![](https://miro.medium.com/max/640/1*D6s2K1y7kjE14swcgITB1w.png)")
    
    st.markdown(html_temp, unsafe_allow_html = True)

    #following lines create the visuals
    City = st.selectbox('City', ('Mumbai','Chennai','Bangalore','Hyderabad','Delhi','Kolkata'))    
    Bathroom = st.number_input('Enter number of Bathroom')
    Size = st.number_input('Enter Size of the Land')
    bhk = st.number_input('Enter number of BHK')

    #when 'predict' is clicked, make prediction
    if st.button('Make Prediction'):
        result = make_prediction(bhk, Bathroom, Size, City)
        st.success('The price is mostlikely: {}'.format(result))
        print('Just test')

if __name__ == '__main__':
    main()


