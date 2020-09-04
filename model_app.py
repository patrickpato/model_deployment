import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""This app predicts flower species""")
st.sidebar.header("Enter input parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length, 
            'sepal_width': sepal_width, 
            'petal_length': petal_length, 
            'petal_width': petal_width}
    features = pd.DataFrame(data, index = [0])
    return features
data = user_input_features()
st.subheader('User Input Parameters')
st.write(data)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)
prediction = clf.predict(data)
prediction_probability = clf.predict_proba(data)
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('predictions')
st.write(iris.target_names[prediction])
st.subheader('prediction probability')
st.write(prediction_probability)


