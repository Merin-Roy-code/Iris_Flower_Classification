# Iris_Flower_Classification
Iris_Flower_Classification a Machine Learning Project

# Iris Flower Classification Project

## Project Overview
This project focuses on classifying iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica** based on their **sepal** and **petal** measurements. The project includes both a **Jupyter Notebook** for experimentation and a **Streamlit app** for interactive predictions.

---

## Dataset
The project uses the classic **Iris Dataset**, which contains 150 samples with 4 features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Target variable: Species (Setosa, Versicolor, Virginica).

---

## Jupyter Notebook Description
The Jupyter Notebook contains the following steps:

1. Data Loading
   - Loading the Iris dataset using pandas.
   - Inspecting the dataset using .head(), .info(), and .describe().

2. Data Exploration & Visualization
   - Plotting histograms and scatter plots to visualize feature distributions.
   - Using pairplots and correlation heatmaps to identify relationships.

3. Data Preprocessing
   - Encoding the target variable using LabelEncoder.
   - Splitting the dataset into training and testing sets.

4. Model Training & Evaluation
   - Models used: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors.
   - Evaluated using accuracy, confusion matrix, and classification report.

5. Model Selection
   - Selecting the best-performing model for deployment in the Streamlit app.

---

## Streamlit App
The Streamlit app allows users to interactively predict the iris species by entering feature values.

### Features:
- Input fields for:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
- Displays the predicted species.
- Shows the model accuracy on the test data.

### How to Run:
1. Make sure you have streamlit installed:

    pip install streamlit

2. Run the Streamlit app:

    streamlit run iris_app.py

---

## Project Structure
    iris-flower-classification/
    │
    ├── iris_classification.ipynb       # Jupyter Notebook with data analysis and model training
    ├── iris_app.py                     # Streamlit app for interactive predictions
    ├── model.pkl                        # Trained machine learning model (if saved)
    ├── requirements.txt                # Required Python packages
    └── README.md                       # Project documentation

---

## Requirements
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- streamlit  

Install all dependencies using:

    pip install -r requirements.txt

---

## Conclusion
This project demonstrates a complete machine learning workflow from data analysis to interactive deployment. Users can explore the Iris dataset in the notebook and predict new samples using the Streamlit app.
