import streamlit as st
import pandas as pd
import pickle

model=pickle.load(open('bagging.pkl','rb'))

def main():
    st.title('Car Price Prediction Using ML')
    img='car.png'
    st.image(img)
    st.subheader('Car Price Predictor')
    st.info('''We need some information to predict Car price''')


    df=pd.read_csv('CAR DETAILS.csv')
    cars=(df['name'].unique())
    transmission=(df['transmission'].unique())
    seller=(df['seller_type'].unique())
    owner=(df['owner'].unique())
    fuel=(df['fuel'].unique())

    p2=st.slider('Model Year',2005,2020,2005)

    p3=st.selectbox('Seller Type',seller)
    if p3=='Individual':
        p3=1
    elif p3=='Dealer':
        p3=0
    elif p3=='Trustmark Dealer':
        p3=2

    p4=st.selectbox('Owner Type',owner)
    if p4=='First Owner':
        p4=0
    elif p4=='Second Owner':
        p4=2
    elif p4=='Third Owner':
        p4=4
    elif p4=='Fourth & Above Owner':
        p4=1
    elif p4=='Test Drive Car':
        p4=3

    p5=st.selectbox('Transmission Type',transmission)
    if p5=='Manual':
        p5=1
    elif p5=='Automatic':
        p5=0

    p6=st.selectbox('Fuel Type',fuel)
    if p6=='Petrol':
        p6=4
    elif p6=='Diesel':
        p6=1
    elif p6=='CNG':
        p6=0
    elif p6=='LPG':
        p6=3
    elif p6=='Electric':
        p6=2

    p7=(st.slider('KM Driven',500,10000000,500))/100000

    x=pd.DataFrame({'year':[p2],'fuel':[p6],'seller_type':[p3],
                    'transmission':[p5],'owner':[p4],'km_driven_in_lacks':[p7]})
    ok=st.button('Predict Car Price')
    if ok:
        prediction=model.predict(x)
        st.success('Predicted Car Price:'+str( prediction*100000) +'Rupees')
        st.caption('Thanks for using!')
        st.balloons()

if __name__=='__main__':
    main()