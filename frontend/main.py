import streamlit as st
import requests

# Define the API base URL
api_url = 'http://127.0.0.1:8000'  # Adjust the URL if running on a different host or port

# Initialize session state variables
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'probability' not in st.session_state:
    st.session_state['probability'] = None


st.set_page_config(
    page_title="Sepsis Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Sepsis Prediction Features")
st.write('- Sepsis: Positive if a patient in ICU will develop sepsis, and Negative otherwise')
col1, col2 = st.columns(2)
with col1:
    st.selectbox('Select Model', options=['decision_tree', 'logistic_regression'], key='model_name')
    model_name = st.session_state['model_name']

def show_form():
    with st.form('enter_features'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("PRG-Plasma glucose level", min_value=0, max_value=100, key='PRG')
            st.number_input("PL-Blood Work Result-1 (mu U/ml)", min_value=0, max_value=500, key='PL')
            st.number_input("PR-Blood Pressure (mm Hg)", min_value=0, max_value=500, key='PR')
        with col2:        
            st.number_input("SK-Blood Work Result-2 (mm)", min_value=0, max_value=500, key='SK')
            st.number_input("TS-Blood Work Result-3 (mu U/ml)", min_value=0, max_value=500, key='TS')
            st.number_input('M11-Body mass index (weight in kg/(height in m)^2)', min_value=0, max_value=200, key='M11')
        with col3:
            st.number_input('BD2-Blood Work Result-4 (mu U/ml)', min_value=0, max_value=100, key='BD2')
            st.number_input("Age-patients age (years)", min_value=0, max_value=150, key='Age')
            st.number_input("A patient holds a valid insurance card", min_value=0, max_value=1, key='Insurance')
        st.form_submit_button('Predict Sepsis Status', on_click=make_prediction)

def make_prediction():
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
    
    # Determine the API endpoint based on the selected model
    if model_name == 'decision_tree':
        endpoint = f'{api_url}/predict_decision_tree'
    elif model_name == 'logistic_regression':
        endpoint = f'{api_url}/predict_logistic_regression'
    
    # Send the data to the appropriate endpoint
    response = requests.post(endpoint, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        st.session_state['prediction'] = response_data.get('prediction', [None])[0]
        # If your model returns a probability, handle it here
        # st.session_state['probability'] = response_data.get('probability', 'N/A')
    else:
        st.session_state['prediction'] = None
        st.session_state['probability'] = None
        st.error("Error occurred during prediction. Please try again.")

if __name__ == "__main__":  
    show_form()
    
    final_prediction = st.session_state['prediction']
    if final_prediction is None:
        st.write('### Prediction will show here after submission')
    else:
        col1, col2 = st.columns(2)

        with col1:
            if final_prediction == "Positive":
                st.write("### Sepsis test will be :red[Positive]\nPatient is likely to develop Sepsis")
            else: 
                st.write(f'### Sepsis test will be :green[Negative]\nPatient is not likely to develop Sepsis')     
        with col2:
            st.subheader('@ What Probability?')
                        
            if 'probability' in st.session_state and isinstance(st.session_state['probability'], (int, float)):
                if final_prediction == 'Negative':
                    st.write(f'#### :green[{round((st.session_state["probability"]*100),2)}%] chance of Patient not developing Sepsis.')
                else:
                    st.write(f'#### :red[{round((st.session_state["probability"]*100),2)}%] chance of Patient developing Sepsis.')
            else:
                st.write(f'#### Probability not available for this model')