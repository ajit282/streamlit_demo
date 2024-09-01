import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Function to load dataset
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        # Clean the column for enrollment numbers
        #df['course_students_enrolled'] = df['course_students_enrolled'].apply(convert_enrollment_to_number)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Convert enrollment numbers from string format to numeric format
def convert_enrollment_to_number(enrollment_str):
    try:
        if isinstance(enrollment_str, str):
            # Remove any non-numeric characters except for decimal points and commas
            enrollment_str = enrollment_str.replace(',', '')
            if 'k' in enrollment_str.lower():
                return float(enrollment_str.lower().replace('k', '')) * 1000
            elif 'm' in enrollment_str.lower():
                return float(enrollment_str.lower().replace('m', '')) * 1000000
            else:
                return float(enrollment_str)
        return float(enrollment_str)
    except ValueError:
        st.warning(f"Cannot convert value to number: {enrollment_str}")
        return None

# Function to perform sentiment analysis on course reviews
def analyze_sentiment(review):
    try:
        return TextBlob(review).sentiment.polarity
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return 0

# Main function to run the app
def main():
    st.title("Coursera Courses EDA")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Visualization", "Insights and Recommendations", "Conclusion"])

    # Dataset selection
    #st.sidebar.header("Dataset Selection")
    #dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Upload Your Own", "Use Example Dataset"])
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df.empty:
            st.stop()
        else:
            st.sidebar.success("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

    # Feature selection
    st.sidebar.header("Feature Selection")
    all_features = df.columns.tolist()
    selected_features = st.sidebar.multiselect("Select Features", options=all_features, default=all_features)
    df = df[selected_features]

    # Sections of the app
    if selection == "Introduction":
        st.header("Introduction")
        st.write("""
        This application provides an exploratory data analysis (EDA) of Coursera courses.
        You can upload your own dataset or use the example dataset provided. 
        The analysis includes data exploration, visualization, insights, and recommendations.
        Use the sidebar to navigate through the different sections.
        """)

    elif selection == "Data Exploration":
        st.header("Data Exploration")
        st.write("### Raw Data")
        st.write(df.head())

        st.write("### Basic Statistics")
        st.write(df.describe())

        # Filters for data exploration
        if 'course_organization' in df.columns:
            st.sidebar.header("Filters")
            organization = st.sidebar.multiselect(
                "Select Organization",
                options=df["course_organization"].unique(),
                default=df["course_organization"].unique()
            )
            df = df[df['course_organization'].isin(organization)]

        if 'course_difficulty' in df.columns:
            difficulty = st.sidebar.multiselect(
                "Select Difficulty Level",
                options=df["course_difficulty"].unique(),
                default=df["course_difficulty"].unique()
            )
            df = df[df['course_difficulty'].isin(difficulty)]

        st.write("### Filtered Data")
        st.write(df)

        st.subheader("Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation analysis.")

    elif selection == "Visualization":
        st.header("Visualization")
        df['course_students_enrolled'] = df['course_students_enrolled'].apply(convert_enrollment_to_number)
        # Top Rated Course Providers
        if 'course_organization' in df.columns and 'course_rating' in df.columns:
            st.write("#### Top Rated Course Providers")
            top_providers = df.groupby('course_organization')['course_rating'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(top_providers, x=top_providers.index, y=top_providers.values, 
                         title='Top 10 Rated Course Providers',
                         labels={'x': 'Course Provider', 'y': 'Average Course Rating'},
                         color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig)

        # Distribution of Course Ratings
        if 'course_rating' in df.columns:
            st.write("#### Distribution of Course Ratings")
            fig = px.histogram(df, x='course_rating', nbins=20, title='Distribution of Course Ratings',
                               labels={'course_rating': 'Course Rating'}, 
                               color_discrete_sequence=['#636EFA'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

        # Number of Courses per Organization
        if 'course_organization' in df.columns:
            st.write("#### Number of Courses per Organization")
            org_counts = df['course_organization'].value_counts()
            fig = px.bar(org_counts, x=org_counts.index, y=org_counts.values, 
                         title='Number of Courses per Organization',
                         labels={'x': 'Organization', 'y': 'Number of Courses'},
                         color_discrete_sequence=['#EF553B'])
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig)

        # Course Ratings by Difficulty Level
        if 'course_difficulty' in df.columns and 'course_rating' in df.columns:
            st.write("#### Course Ratings by Difficulty Level")
            fig = px.box(df, x='course_difficulty', y='course_rating', 
                         title='Course Ratings by Difficulty Level',
                         labels={'course_difficulty': 'Difficulty Level', 'course_rating': 'Course Rating'},
                         color='course_difficulty', 
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig)

        # Students Enrolled vs. Course Rating
        if 'course_students_enrolled' in df.columns and 'course_rating' in df.columns:
            st.write("#### Students Enrolled vs. Course Rating")
            fig = px.scatter(df, x='course_students_enrolled', y='course_rating', 
                             size='course_students_enrolled', 
                             title='Students Enrolled vs. Course Rating',
                             labels={'course_students_enrolled': 'Students Enrolled', 'course_rating': 'Course Rating'},
                             color='course_rating', 
                             color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title="Students Enrolled", yaxis_title="Course Rating")
            st.plotly_chart(fig)

        # Distribution of Course Ratings Based on Difficulty Level
        if 'course_rating' in df.columns and 'course_difficulty' in df.columns:
            st.write("#### Distribution of Course Ratings Based on Difficulty Level")
            fig = px.histogram(df, x='course_rating', color='course_difficulty', 
                               title='Course Rating Distribution by Difficulty Level',
                               labels={'course_rating': 'Course Rating', 'course_difficulty': 'Difficulty Level'},
                               barmode='overlay', 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

        # Distribution of Course Difficulty Based on Certificate Type
        if 'course_difficulty' in df.columns and 'course_certificate_type' in df.columns:
            st.write("#### Distribution of Course Difficulty Based on Certificate Type")
            fig = px.histogram(df, x='course_difficulty', color='course_certificate_type', 
                               title='Course Difficulty Distribution by Certificate Type',
                               labels={'course_difficulty': 'Course Difficulty', 'course_certificate_type': 'Certificate Type'},
                               barmode='group', 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

    elif selection == "Insights and Recommendations":
        st.header("Insights and Recommendations")
        st.write("""
        Based on the visualizations and data exploration, you can derive various insights.

        - **Top Rated Courses**: Courses with higher ratings often have higher enrollments.
        - **Popular Organizations**: Organizations offering a broad range of courses may have a significant impact on the overall enrollment.
        - **Course Difficulty**: Mixed difficulty courses tend to attract a larger and more diverse student base.

        ### Recommendations:
        - **Focus on High-Rated Courses**: Institutions should focus on promoting high-rated courses to attract more students.
        - **Diversify Course Offerings**: Offering courses with varied difficulty levels can attract a wider audience.
        """)

    elif selection == "Conclusion":
        st.header("Conclusion")
        st.write("""
        This exploratory data analysis provided insights into the Coursera courses dataset.
        You explored various features of the dataset, visualized key metrics, and derived insights and recommendations.
        The flexibility of the app allows you to upload your dataset and explore it based on your needs.
        """)

if __name__ == "__main__":
    main()
