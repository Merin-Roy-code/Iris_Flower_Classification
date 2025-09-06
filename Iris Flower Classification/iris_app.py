import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

# Train model
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X, y)

# Streamlit app
st.title("🌸 Iris Flower Classification App")
st.subheader("📊 Iris Dataset")
st.dataframe(df)  

# Sidebar for user input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width  = st.sidebar.slider("Petal Width (cm)",  float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

if st.sidebar.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    st.success(f"🌼 Predicted Species: **{target_names[prediction[0]]}**")


# Scatter plot
st.subheader("🌿 Sepal Length vs Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", palette="Set2", ax=ax)
st.pyplot(fig)

# Histogram
st.subheader("🌸 Petal Length Distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x="petal length (cm)", hue="species", multiple="stack", palette="Set2", ax=ax)
st.pyplot(fig)