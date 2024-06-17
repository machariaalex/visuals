import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Data Visualizer")

st.write("""
This app helps you to choose a dataset, view descriptive statistics, the first few rows, perform EDA, and visualize the data.
""")

# Dataset selection
data_source = st.radio("Select data source", ["Upload your own dataset", "Use a default dataset"])

# Default datasets from Kaggle
default_datasets = {
    "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic Dataset": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Wine Quality Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "Boston Housing Dataset": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "Diabetes Dataset": "https://raw.githubusercontent.com/selva86/datasets/master/diabetes.csv",
    "Heart Disease Dataset": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/heart.csv",
    "MNIST Dataset": "https://raw.githubusercontent.com/myleott/mnist_png/master/mnist_png/training/0/1.png",
    "CIFAR-10 Dataset": "https://raw.githubusercontent.com/YoongiKim/CIFAR-10-images/master/train/0_frog.png",
    "COVID-19 Dataset": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/latest/owid-covid-latest.csv",
    "Pokemon Dataset": "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv"
}

if data_source == "Upload your own dataset":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    dataset_name = st.selectbox("Select a default dataset", list(default_datasets.keys()))
    dataset_url = default_datasets[dataset_name]
    df = pd.read_csv(dataset_url)

if 'df' in locals():
    # Split the screen into two columns for dataset info and EDA
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("### Dataset")
        st.dataframe(df)

        st.write("### Descriptive Statistics")
        st.write(df.describe())

        st.write("### First Few Rows")
        st.write(df.head())

    with col2:
        st.write("### Exploratory Data Analysis (EDA)")

        st.write("#### Missing Values")
        st.write(df.isnull().sum())

        st.write("#### Data Types")
        st.write(df.dtypes)

        st.write("#### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("#### Pair Plot")
        sns.pairplot(df)
        st.pyplot()

    # Split the screen into two columns for visualization and filtering
    col3, col4 = st.columns([1, 1])

    with col3:
        st.write("### Data Visualization")
        
        plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot"])
        
        columns = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis", columns)
        y_axis = st.selectbox("Select Y-axis", columns)
        
        if plot_type == "Scatter Plot":
            color = st.selectbox("Select Color", [None] + columns)
            size = st.selectbox("Select Size", [None] + columns)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, size=size, title="Scatter Plot")
            st.plotly_chart(fig)
        
        elif plot_type == "Line Plot":
            fig = px.line(df, x=x_axis, y=y_axis, title="Line Plot")
            st.plotly_chart(fig)
        
        elif plot_type == "Bar Plot":
            fig = px.bar(df, x=x_axis, y=y_axis, title="Bar Plot")
            st.plotly_chart(fig)
        
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_axis, title="Histogram", nbins=50)
            st.plotly_chart(fig)
        
        elif plot_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, title="Box Plot")
            st.plotly_chart(fig)

    with col4:
        st.write("### Filter Data")
        filter_column = st.selectbox("Select column to filter", columns)
        unique_values = df[filter_column].unique()
        selected_values = st.multiselect("Select values", unique_values)
        filtered_df = df[df[filter_column].isin(selected_values)]
        
        st.write("#### Filtered Data")
        st.dataframe(filtered_df)
