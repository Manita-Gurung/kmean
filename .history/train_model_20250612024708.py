import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("final_features_Airbnb_Open_Data.csv")

# Use only the 7 features from the HTML form
features = [
    'price',
    'service_fee',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]

X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save model and scaler
pickle.dump(kmeans, open("simple_model.pkl", "wb"))
pickle.dump(scaler, open("simple_scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully.")
