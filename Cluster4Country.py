import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load Dataset
@st.cache(allow_output_mutation=True)
def load_data():
    file_path = 'Country_Dataset.csv'  # Make sure this file is in the same directory
    df = pd.read_csv(file_path)
    
    # Data Cleaning: Handle missing values
    df = df.fillna(df.median(numeric_only=True))  # Fill missing values with median for numeric columns
    df = df.dropna()  # Drop rows with any remaining missing values
    
    return df

# Enhanced Exploratory Data Analysis (EDA) with Plotly
def eda(df):
    st.write("### Enhanced Exploratory Data Analysis (EDA)")

    # Plotly Correlation Heatmap
    st.write("#### Correlation Heatmap between Variables")
    corr_matrix = df.drop(columns='Country').corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu', origin="lower")
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables", 
                      coloraxis_showscale=True)
    st.plotly_chart(fig)

    # Interactive Distribution Plots with KDE
    st.write("#### Distribution of Variables")
    num_cols = df.columns[1:]  # Skip 'Country' column
    
    for col in num_cols:
        fig = px.histogram(df, x=col, marginal="box", nbins=50, title=f"Distribution of {col}", 
                           color_discrete_sequence=['#2a9d8f'])
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        st.plotly_chart(fig)

    # Box Plots for Outliers
    st.write("#### Box Plots for Outliers")
    for col in num_cols:
        fig = px.box(df, y=col, title=f"Box Plot of {col}", color_discrete_sequence=['#e76f51'])
        fig.update_layout(yaxis_title=col)
        st.plotly_chart(fig)

# Plot Elbow Method for Optimal Number of Clusters
def plot_elbow_method(df):
    st.write("### Elbow Method for Optimal Number of Clusters")

    # Standardize features
    features = df.select_dtypes(include=['float64', 'int64'])  # Only numeric columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Compute inertia for different numbers of clusters
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    fig, ax = plt.subplots()
    ax.plot(k_range, inertias, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

# Calculate Silhouette Score
def calculate_silhouette_score(df, n_clusters, use_pca):
    features = df.select_dtypes(include=['float64', 'int64'])  # Only numeric columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if use_pca:
        pca = PCA(n_components=2)
        features = pca.fit_transform(scaled_features)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(features)
    score = silhouette_score(features, clusters)
    return score

# Clustering Analysis with Toggle for PCA
def clustering_analysis(df, n_clusters, use_pca):
    st.write("### KMeans Clustering Analysis")

    # Standardizing the features
    features = df.select_dtypes(include=['float64', 'int64'])  # Only numeric columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if use_pca:
        # Applying PCA
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)

        # Applying KMeans on PCA-transformed data
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(pca_features)
        df['Cluster'] = clusters  # Save clusters in a column

        # Visualization PCA clusters
        st.write(f"#### Clustering with PCA (Silhouette Score: {calculate_silhouette_score(df, n_clusters, use_pca):.2f})")
        fig = px.scatter(x=pca_features[:, 0], y=pca_features[:, 1], color=clusters.astype(str),
                         title='Clustering with PCA', color_discrete_sequence=px.colors.qualitative.Set2)
        centroids = kmeans.cluster_centers_
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(color='red', size=12), name="Centroids"))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Applying KMeans without PCA
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        df['Cluster'] = clusters  # Save clusters in a column

        # Visualization without PCA
        st.write(f"#### Clustering without PCA (Silhouette Score: {calculate_silhouette_score(df, n_clusters, use_pca):.2f})")
        fig = px.scatter(x=scaled_features[:, 0], y=scaled_features[:, 1], color=clusters.astype(str),
                         title='Clustering without PCA', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

# Recommendations based on clustering
def recommendations(df):
    st.write("### Recommendations Based on Clustering")

    if 'Cluster' in df.columns:
        # Assuming numerical columns for generating recommendations
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        cluster_groups = numeric_df.groupby(df['Cluster']).mean()
        
        # Feature Importance
        feature_importance = pd.DataFrame()
        for cluster in cluster_groups.index:
            importance = cluster_groups.loc[cluster] - numeric_df.mean()
            feature_importance[cluster] = importance.abs()
        
        st.write("#### Feature Importance by Cluster")
        fig = go.Figure()
        for cluster in feature_importance.columns:
            fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance[cluster], name=f'Cluster {cluster}'))
        fig.update_layout(title="Feature Importance by Cluster", xaxis_title="Features", yaxis_title="Importance", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        # Income and GDP Maps
        st.write("#### Income and GDP Visualization")

        # Add income and GDP to the DataFrame for plotting
        df['Income_Category'] = pd.cut(df['income'], bins=[-float('inf'), 10000, 50000, float('inf')], labels=['Low', 'Medium', 'High'])
        df['GDP_Category'] = pd.cut(df['gdpp'], bins=[-float('inf'), 20000, 50000, float('inf')], labels=['Low', 'Medium', 'High'])
        df['Life_Expectancy_Category'] = pd.cut(df['life_expec'], bins=[-float('inf'), 60, 75, float('inf')], labels=['Low', 'Medium', 'High'])

        # Income Map
        fig_income = px.choropleth(df, locations="Country", locationmode="country names", color="Income_Category",
                                  color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"},
                                  title="Income by Country")
        st.plotly_chart(fig_income, use_container_width=True)

        # GDP Map
        fig_gdp = px.choropleth(df, locations="Country", locationmode="country names", color="GDP_Category",
                                color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"},
                                title="GDP by Country")
        st.plotly_chart(fig_gdp, use_container_width=True)

        # Life Expectancy Map
        fig_life_expectancy = px.choropleth(df, locations="Country", locationmode="country names", color="Life_Expectancy_Category",
                                            color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"},
                                            title="Life Expectancy by Country")
        st.plotly_chart(fig_life_expectancy, use_container_width=True)

        # Recommendations based on the cluster analysis
        for cluster in cluster_groups.index:
            st.write(f"#### Recommendations for Cluster {cluster}:")
            cluster_summary = cluster_groups.loc[cluster]
            st.write(f"Cluster {cluster} has the following average values:")
            st.write(cluster_summary)

            income = cluster_summary.get('income', None)
            gdp = cluster_summary.get('gdpp', None)
            life_expectancy = cluster_summary.get('life_expec', None)

            if income is not None:
                st.write(f"- The average income in this cluster is ${income:.2f}.")
            if gdp is not None:
                st.write(f"- The average GDP in this cluster is ${gdp:.2f}.")
            if life_expectancy is not None:
                st.write(f"- The average life expectancy in this cluster is {life_expectancy:.2f}.")

            if income < 20000:
                st.write("Consider economic development programs to increase income levels.")
            elif income > 50000:
                st.write("Implement advanced economic strategies to maintain high-income levels.")

            if gdp < 30000:
                st.write("Focus on economic growth and infrastructure development.")
            elif gdp > 70000:
                st.write("Leverage high GDP for advanced technological and economic initiatives.")

            if life_expectancy < 70:
                st.write("Enhance healthcare infrastructure and services.")
            elif life_expectancy > 80:
                st.write("Promote wellness programs and maintain current healthcare standards.")

# Display Data Summary
def display_data_summary(df):
    st.write("### Data Summary")

    # Display basic information about the dataset
    st.write("#### Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Display basic statistics
    st.write("#### Basic Statistics")
    st.write(df.describe())

    # Display the first few rows of the dataset
    st.write("#### Sample Data")
    st.write(df.head())

# Main Application
def main():
    st.title("Country Dataset: EDA and Clustering Analysis")
    
    st.write("""
    This application provides an analysis of various country-level indicators such as child mortality, income, and life expectancy.
    Through exploratory data analysis and clustering techniques, we aim to uncover patterns and provide recommendations.
    """)

    # Load Data
    df = load_data()

    # Sidebar Menu for Navigation
    menu = st.sidebar.selectbox("Menu", ["Introduction", "EDA", "Clustering Analysis", "Recommendations"])

    if menu == "Introduction":
        st.write("### Country Dataset")
        st.write(df)
        display_data_summary(df)
        
    elif menu == "EDA":
        st.header("1. Exploratory Data Analysis")
        eda(df)

    elif menu == "Clustering Analysis":
        st.header("2. Clustering Analysis")

        # Options for clustering
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        use_pca = st.sidebar.checkbox("Use PCA", value=False)

        clustering_analysis(df, n_clusters, use_pca)

        # Show the elbow method plot
        plot_elbow_method(df)

    elif menu == "Recommendations":
        st.header("3. Recommendations")
        recommendations(df)

if __name__ == "__main__":
    main()
