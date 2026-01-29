import streamlit as st 
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sklearn


def clean_data():
    data = pd.read_csv("data\data.csv")
    
    data = data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    
    return data



def add_sidebar():
    st.sidebar.header("Cell Clusters : ")
    data = clean_data()
    input_dict={}
    
    mean_features = [
        
        ("Radius Mean", "radius_mean"),
        ("Texture Mean", "texture_mean"),
        ("Perimeter Mean", "perimeter_mean"),
        ("Area Mean", "area_mean"),
        ("Smoothness Mean", "smoothness_mean"),
        ("Compactness Mean", "compactness_mean"),
        ("Concavity Mean", "concavity_mean"),
        ("Concave Points Mean", "concave points_mean"),
        ("Symmetry Mean", "symmetry_mean"),
        ("Fractal Dimension Mean", "fractal_dimension_mean")]
    
    
    with st.sidebar.expander("MEAN FEATURES",expanded=True):
        for label,key in mean_features:
            input_dict[key]=st.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value = float(data[key].mean()),
                
            )
    
    se_features =[("Radius SE", "radius_se"),
    ("Texture SE", "texture_se"),
    ("Perimeter SE", "perimeter_se"),
    ("Area SE", "area_se"),
    ("Smoothness SE", "smoothness_se"),
    ("Compactness SE", "compactness_se"),
    ("Concavity SE", "concavity_se"),
    ("Concave Points SE", "concave points_se"),
    ("Symmetry SE", "symmetry_se"),
    ("Fractal Dimension SE", "fractal_dimension_se")]
    
    with st.sidebar.expander("STANDARD ERROR FEATURES",expanded=True):
        for label,key in se_features:
            input_dict[key]=st.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value = float(data[key].mean()),
        )
        
    worst_features=[("Radius Worst", "radius_worst"),
    ("Texture Worst", "texture_worst"),
    ("Perimeter Worst", "perimeter_worst"),
    ("Area Worst", "area_worst"),
    ("Smoothness Worst", "smoothness_worst"),
    ("Compactness Worst", "compactness_worst"),
    ("Concavity Worst", "concavity_worst"),
    ("Concave Points Worst", "concave points_worst"),
    ("Symmetry Worst", "symmetry_worst"),
    ("Fractal Dimension Worst", "fractal_dimension_worst"),
        ]

    
    with st.sidebar.expander("WORST FEATURES",expanded=True):
        for label,key in worst_features:
            input_dict[key]=st.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value = float(data[key].mean()),
                
            )        
        
    return input_dict    


def scaled_data(input_dict):
    data = clean_data()
    
    X = data.drop(['diagnosis'],axis=1)
    
    scaled_dict ={}
    
    for key,value in input_dict.items():
        max_val =X[key].max()
        min_val =X[key].min()
        if (max_val - min_val)==0:
            scaled_value=0
        else:
            scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict    

def get_radar_chart(input_data):
    
    input_data=scaled_data(input_data)
    
    categories = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
        'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
        'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
        ]

    fig = go.Figure()


    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'],
            input_data['perimeter_mean'], input_data['area_mean'], 
            input_data['smoothness_mean'], input_data['compactness_mean'], 
            input_data['concavity_mean'], 
            ],
        theta=categories,
        fill='toself',
        name='Mean value'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[ input_data['radius_se'], input_data['texture_se'], 
            input_data['perimeter_se'], input_data['area_se'], 
            input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'],
            input_data['symmetry_se'], input_data['fractal_dimension_se'],],
        theta=categories,
        fill='toself',
        name='Standard error'
    ))


    fig.add_trace(go.Scatterpolar(
    r=[ 
        input_data['radius_worst'], input_data['texture_worst'], 
        input_data['perimeter_worst'], input_data['area_worst'], 
        input_data['smoothness_worst'], input_data['compactness_worst'],
        input_data['concavity_worst'], input_data['concave points_worst'], 
        input_data['symmetry_worst'], input_data['fractal_dimension_worst']

        ],
    theta=categories,
    fill='toself',
    name='Worst value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=False
    )

    return fig

def get_prediction(input_data):
    
    model = pickle.load(open("model.pkl",'rb'))
    scaler = pickle.load(open("scaler.pkl",'rb'))
    
    input_array =np.array(list(input_data.values())).reshape(1,-1)
    
    input_array_scaled = scaler.transform(input_array)
    
    predictions = model.predict(input_array_scaled)
    
    st.subheader(" CELL CLUSTER PREDICTIONS ")
    
    st.markdown("<div class='pred-card'> <h3> RESULTS </h3> </div>",unsafe_allow_html=True)
    
    
    
    if predictions[0] == 0:
        st.write("<span class = 'diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>",unsafe_allow_html=True)    
    
    st.write(model.predict_proba(input_array_scaled)[0][0])
    st.progress(float(model.predict_proba(input_array_scaled)[0][0]))
    st.caption("probability of Bening")
    
    
    
    st.write(model.predict_proba(input_array_scaled)[0][1])
    st.progress(float(model.predict_proba(input_array_scaled)[0][1]))
    st.caption("probability of Malignant")
    
def main():
    st.set_page_config(
        page_title=" Breast Cancer Prediction",
        page_icon=":woman_health_worker:",
        layout='wide',
        initial_sidebar_state='expanded'
        
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html = True)
    
    input_data=add_sidebar()
    
    with st.container():
        st.write("<h1 class=head> &#x1F397 BREAST CANCER PREDICTOR  </h1>",unsafe_allow_html=True)
        st.write("Breast Cancer Prediction App uses machine learning to predict whether a tumor is benign or malignant based on medical input data. It provides fast and reliable results through a simple, user-friendly interface. The app supports early detection and helps users make informed healthcare decisions.")

    col1,col2 = st.columns([4,1])
    
    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        get_prediction(input_data)    


if __name__ == '__main__':
    main()