from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Set your OpenAI API key
load_dotenv(dotenv_path=r'.env')
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the client
client = OpenAI(api_key=api_key)

# Load user feedback from CSV (Ensure it has a "Comment" column)
df = pd.read_excel("kmeans_test_file.xlsx")

# Drop any empty comments
df = df.dropna(subset=["Comment"])

# Function to generate OpenAI text embeddings
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Convert comments into embeddings
print("Generating text embeddings...")
df["embedding"] = df["Comment"].apply(lambda x: get_embedding(x))

# Convert list of embeddings into a NumPy array
embeddings = np.vstack(df["embedding"].values)

# Set number of clusters (adjust as needed)
num_clusters = 8

# Run K-Means clustering
print("Clustering comments into groups...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(embeddings)

# Function to get cluster labels using GPT-4
def label_cluster(comments):
    prompt = f"""
    Here are some user feedback comments:
    {comments}

    What common UX issue do they describe?
    Possible categories: ["Bug/Error", "Ease of use", "Early pop up", "Findability/Navigation", "Sign in/access", "Other", "Integration"]

    Output format: <Best Category>
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Auto-label each cluster using GPT-4
print("Labeling clusters with GPT-4...")
cluster_labels = {}
for cluster_id in range(num_clusters):
    cluster_comments = df[df["Cluster"] == cluster_id]["Comment"].tolist()
    cluster_labels[cluster_id] = label_cluster(cluster_comments[:10])  # Limit to 10 samples for GPT efficiency

# Assign labels to dataframe
df["Cluster Label"] = df["Cluster"].map(cluster_labels)

# Save results to CSV
df.to_excel("clustered_feedback.xlsx", index=False)
print("Clustering complete! Results saved to clustered_feedback.xlsx.")

# **Optional: Visualize Clusters using PCA**
print("Generating visualization...")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df["Cluster Label"], palette="tab10", alpha=0.7)
plt.title("UX Feedback Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster Label")
plt.show()
