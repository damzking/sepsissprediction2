import streamlit as st
import requests

#backend_url = http://127.0.0.1:8000

st.set_page_config(
    page_title='Prediction',
    layout='wide',
    page_icon='expanded'

)

def show_form():
    st.title('SEPSSIS Prediction')
    st.subheader('Please enter your Sepssis features')
    with st.form(key = 'sepssis'):
        PRG = st.number_input('PRG', min_value=0.00, max_value=100.0)
        PL = st.number_input('PL', min_value=0.00, max_value=100.0)
        PR = st.number_input('PR', min_value=0.00, max_value=100.0)
        SK = st.number_input('SK', min_value=0.00, max_value=100.0)
        TS = st.number_input('TS', min_value=0.00, max_value=100.0)
        M11 = st.number_input('M11', min_value=0.0, max_value=100.0)
        BD2 = st.number_input('BD2', min_value=0.0, max_value=100.0)
        AGE = st.number_input('Age', min_value=0.0, max_value=100.0)

        if st.form_submit_button('Predict Sepssis Result'):
            input_data= {
                'PRG': PRG,
                'PL': PL,
                'PR': PR,
                'SK': SK,
                'TS': TS,
                'M11': M11,
                'BD2': BD2,
                'AGE': AGE
            }
            st.write(input_data)

        #response = requests.post(f'{backend_url}/XGBoost_prediction', json=input_data)

        #Display the prediction
        #if response.status_code == 200:
            #prediction = response.json()['prediction']
            #st.success(f'The prediction is: {prediction}')
        #else:
            #st.error('An error occurred while making the prediction')

if __name__ == '__main__':
    show_form()