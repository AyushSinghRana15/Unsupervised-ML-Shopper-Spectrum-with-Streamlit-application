# Unsupervised-ML-Shopper-Spectrum-with-Working-Streamlit-Application


Customer Segmentation & Product Recommendation System


This project implements a full customer analytics and recommendation pipeline on retail transaction data. The solution identifies valuable customer segments using RFM-based clustering and provides product recommendations using item-based collaborative filtering. An interactive Streamlit web app brings these predictive insights to end users.

<img width="1470" height="956" alt="Screenshot 2025-08-01 at 12 48 01 PM" src="https://github.com/user-attachments/assets/718f3ce4-94be-4dc9-bc47-9aef287f1926" />
<img width="1470" height="956" alt="Screenshot 2025-08-01 at 12 48 47 PM" src="https://github.com/user-attachments/assets/ecc471ff-6aac-4cac-aee0-a5b5f61ff42b" />
<img width="1470" height="956" alt="Screenshot 2025-08-01 at 12 49 12 PM" src="https://github.com/user-attachments/assets/e13d9faf-1c85-4f60-b88c-2ceea7109cba" />



🚀 Features


	•	📊 Customer Segmentation (Clustering) Segment customers using RFM (Recency, Frequency, Monetary) metrics and additional behavioral features with a trained K-Means model.
	•	🔁 Product Recommendation System Suggest similar products using item-based collaborative filtering based on cosine similarity of historical purchase behavior.
	•	📺 Streamlit Web App Simple and intuitive user interface with sidebar navigation to toggle between:
	•	🏠 Home
	•	👥 Clustering Module
	•	🔗 Recommendation Module

 
📁 Project Structure


├── app.py                 # Streamlit app (main entry point)


├── cleaned_dataset.csv    # Cleaned retail transaction data


├── kmeans_model.pkl       # Trained KMeans model (6-feature input)


├── requirements.txt       # Required packages


└── README.md              # Project overview and usage guide


✅ Input Features


Customer Segmentation (Clustering):


	•	Recency (days since last purchase)
	•	Frequency (total number of purchases)
	•	Monetary (total spend)
	•	Avg Order Value
	•	Purchase Span (days between first and last order)
	•	Order Rate (orders per day)

 
 ▶️ How to Run the App

 
	1.	Clone the repository and navigate to it:
 git clone https://github.com/AyushSinghRana15/Unsupervised-ML-Shopper-Spectrum-with-Streamlit-application

 
cd your-project-name


	2.	Install dependencies:
 pip install -r requirements.txt

 
	3.      Launch the Streamlit app:
 streamlit run app.py

 
	4.	Explore in the browser: The app will open at `http://localhost:8501`
💡 App Instructions


1️⃣ Product Recommendation Module


	•	Enter a product’s StockCode (e.g., `84029E`) in the input field.
	•	Click “Get Recommendations.”
	•	View the Top 5 most similar products purchased by similar customers.

 
2️⃣ Customer Segmentation Module


	•	Input:
	•	Recency (in days)
	•	Frequency (purchase count)
	•	Monetary (total spend)
	•	Click “Predict Cluster”
	•	View the customer’s cluster number and segment label (e.g., High-Value, Regular, Occasional, At-Risk).

 
📌 Notes


	•	Make sure the file `cleaned_dataset.csv` is in the app root folder.
	•	The model `kmeans_model.pkl` must match the feature set used during training (6 features: RFM + 3 engineered features).
	•	Product similarity is calculated using a cosine similarity matrix built from the customer-product purchase history.

 
🔒 License


This project is licensed under the MIT License – feel free to use, share, or adapt the code.
Let me know if you’d also like this in downloadable Markdown (.md) format or if you want preview badges (Python version, last updated, etc.) added to the top of the README.
