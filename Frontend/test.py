import streamlit as st
import requests

api_url = 'http://127.0.0.1:8000'

st.set_page_config(
    page_title="Sepsis Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="https://example.com/favicon.ico",
)

# Define a function to make the prediction
def make_prediction():
    model_name = st.session_state['model_name']
    data = {
        'PRG': st.session_state['PRG'],
        'PL': st.session_state['PL'],
        'PR': st.session_state['PR'],
        'SK': st.session_state['SK'],
        'TS': st.session_state['TS'],
        'M11': st.session_state['M11'],
        'BD2': st.session_state['BD2'],
        'Age': st.session_state['Age'],
        'Insurance': st.session_state['Insurance']
    }
    
    response = requests.post(f'{api_url}/predict/{model_name}', json=data)
    if response.status_code == 200:
        result = response.json()
        st.session_state['prediction'] = result['prediction']
        st.session_state['probability'] = result['probability']
    else:
        st.error(f"Error: {response.json()['detail']}")

# Show the form to input features
def show_form():
    st.title("Sepsis Prediction features")
    st.write('- Sepsis: Positive: if a patient in ICU will develop a sepsis , and Negative: otherwise')
    
    st.selectbox('Model', options=['GradientBoosting', 'LogisticRegression', 'SVM', 'XGBoost'], key='model_name')
    
    with st.form('enter_features'):
        st.number_input("PRG-Plasma glucose level", min_value=0, max_value=100, key='PRG')
        st.number_input("PL-Blood Work Result-1 (mu U/ml)", min_value=0, max_value=500, key='PL')
        st.number_input("PR-Blood Pressure (mm Hg)", min_value=0, max_value=500, key='PR')
        st.number_input("SK-Blood Work Result-2 (mm)", min_value=0, max_value=500, key='SK')
        st.number_input("TS-Blood Work Result-3 (mu U/ml)", min_value=0, max_value=500, key='TS')
        st.number_input('M11-Body mass index (weight in kg/(height in m)^2)', min_value=0, max_value=200, key='M11')
        st.number_input('BD2-Blood Work Result-4 (mu U/ml)', min_value=0, max_value=100, key='BD2')
        st.number_input("Age-patients age (years)", min_value=0, max_value=150, key='Age')
        st.number_input("A patient holds a valid insurance card", min_value=0, max_value=1, key='Insurance')
        
        st.form_submit_button('Predict Sepsis Status', on_click=make_prediction)

    if 'prediction' in st.session_state and 'probabilities' in st.session_state:
        st.write(f"Prediction: {st.session_state['prediction']}")
        st.write(f"Probability: {st.session_state['probability']}")

if __name__ == "__main__":  
    show_form()
