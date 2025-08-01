import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# data loading
@st.cache_data
def load_data():
    # Replace with your cleaned dataset path
    df = pd.read_csv('cleaned_dataset.csv')
    return df

df = load_data()

# For Recommendation
@st.cache_resource
def compute_similarity(df):
    user_item_matrix = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    item_vectors = user_item_matrix.T
    item_similarity = pd.DataFrame(
        cosine_similarity(item_vectors),
        index=item_vectors.index,
        columns=item_vectors.index
    )
    return item_similarity, item_vectors

item_similarity, item_vectors = compute_similarity(df)
code_to_desc = df.drop_duplicates(['StockCode', 'Description']).set_index('StockCode')['Description']

# For Clustering
kmeans = joblib.load('kmeans_model.pkl') 
recency_mean, recency_std = 91.2, 59.4
frequency_mean, frequency_std = 7.36, 4.25
monetary_mean, monetary_std = 4747.4, 2986.5
recency_30 = (30 - recency_mean)/recency_std
recency_90 = (90 - recency_mean)/recency_std
recency_180 = (180 - recency_mean)/recency_std
frequency_2 = (2 - frequency_mean)/frequency_std
frequency_4 = (4 - frequency_mean)/frequency_std
frequency_10 = (10 - frequency_mean)/frequency_std
monetary_1000 = (1000 - monetary_mean)/monetary_std
monetary_5000 = (5000 - monetary_mean)/monetary_std

def label_segment_scaled(r, f, m):
    if r < recency_30 and f > frequency_10 and m > monetary_5000:
        return "High-Value"
    elif f >= frequency_4 and m >= monetary_1000:
        return "Regular"
    elif f <= frequency_2 and r > recency_90:
        return "Occasional"
    elif r > recency_180 and f <= frequency_2:
        return "At-Risk"
    else:
        return "Other"

# SIDEBAR & SECTION SWITCHING
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Home", "Clustering", "Recommendation"])


# HOME SECTION
if section == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the Customer Analytics and Recommendation App!")
    st.markdown("""
        - **Clustering:** Explore customer segmentation using RFM features.<br>
        - **Recommendation:** Find similar products using collaborative filtering.<br>
        Use the sidebar to switch between sections.
    """, unsafe_allow_html=True)

# CLUSTERING SECTION
elif section == "Clustering":
    st.title("🧑‍🤝‍🧑 Customer Clustering")

    st.header("📊 Customer Segmentation Module")
    st.markdown("Input RFM values to predict this customer’s segment:")

    recency_input = st.number_input("Recency (days since last purchase):", min_value=0, step=1, value=30)
    frequency_input = st.number_input("Frequency (number of purchases):", min_value=1, step=1, value=3)
    monetary_input = st.number_input("Monetary (total spend):", min_value=0, step=1, value=1000)

    if st.button("Predict Cluster"):
        recency_scaled = (recency_input - recency_mean) / recency_std
        frequency_scaled = (frequency_input - frequency_mean) / frequency_std
        monetary_scaled = (monetary_input - monetary_mean) / monetary_std

        # Prepare input for KMeans (pad more features with 0 if your model used more than RFM)
        features = np.array([[recency_scaled, frequency_scaled, monetary_scaled, 0,0,0]])

        cluster_id = kmeans.predict(features)[0]
        segment_label = label_segment_scaled(recency_scaled, frequency_scaled, monetary_scaled)

        st.success(f"This customer is predicted as: **{segment_label}** (Cluster {cluster_id})")

# RECOMMENDATION SECTION
elif section == "Recommendation":
    st.title("🔗 Product Recommendation System")
    st.markdown("Enter a product **StockCode** to see 5 similar products based on customer buying patterns.")

    input_code = st.text_input("🔎 Enter Product StockCode", placeholder="e.g., 84029E")

    def get_similar_products(product_code, top_n=5):
        if product_code not in item_similarity.index:
            return []
        similar = item_similarity[product_code].drop(product_code)
        top_similar = similar.sort_values(ascending=False).head(top_n)
        recommendations = []
        for idx, score in top_similar.items():
            title = code_to_desc.get(idx, 'Unknown Product')
            recommendations.append((idx, title, round(score, 2)))
        return recommendations

    if st.button("🔁 Get Recommendations"):
        if input_code:
            recs = get_similar_products(input_code)
            if recs:
                st.success(f"Top 5 recommendations for `{input_code}`:")
                for code, name, score in recs:
                    st.markdown(f"""
                    <div style='border:1px solid #ddd;padding:10px;margin-bottom:10px;border-radius:6px;'>
                        <strong>{code}</strong>: {name}<br>
                        <i>Similarity score: {score}</i>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("❌ Product not found or no similar items available.")
        else:
            st.info("Please enter a valid StockCode.")
