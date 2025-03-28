from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
from sklearn.cluster import DBSCAN
import hashlib
import json
from pathlib import Path

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

def calculate_hash(obj):
    """Calculate a hash for any object that can be converted to string"""
    return hashlib.md5(str(obj).encode()).hexdigest()

def load_cache():
    """Load the cache from disk"""
    cache_file = Path("cache/embeddings_cache.json")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache_data):
    """Save the cache to disk"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / "embeddings_cache.json", "w") as f:
        json.dump(cache_data, f)

def get_cache_key(df, model, prompt):
    """Generate a cache key based on input data and parameters"""
    components = {
        'data': df["Comment"].to_list(),
        'model': model,
        'prompt': prompt,
        'num_clusters': num_clusters,
        'possible_labels': ["Bug/Error", "Ease of use", "Early survey pop-up", 
                          "Findability/Navigation", "Sign in/access", "Other", "Integration"],
        'version': 2,  # Increment this when logic changes significantly
        'explanation_mode': 'individual'  # Added to indicate we're using per-comment explanations
    }
    return calculate_hash(components)

def label_cluster(comments):
    prompt = f"""
    Here are some user feedback comments:
    {comments}

    1. What common UX issue does the comment describe? Use the list of labels as a guide, but don't feel constrained by them.
    2. Provide a brief 1 sentence explanation for why this cluster label fits the comment. The explanation should be factual and not include the category name.

    Possible labels: ["Bug/Error", "Ease of use", "Early survey pop-up", "Findability/Navigation", "Sign in/access", "Other", "Integration"]

    Output format:
    Category: <Best Label>
    Explanation: <1 sentence explanation of why the cluster label fits the comment>
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content.strip()
    category = result.split('Category:')[1].split('Explanation:')[0].strip()
    explanation = result.split('Explanation:')[1].strip()
    
    if "Category:" in explanation:
        explanation = explanation.split("Category:")[0].strip()
    
    return category, explanation

# Set number of clusters (adjust as needed)
num_clusters = 8

# Get the prompt template for cache key
prompt_template = """
Here are some user feedback comments:
{comments}

1. What common UX issue does the comment describe? Use the list of labels as a guide, but don't feel constrained by them.
2. Provide a brief 1 sentence explanation for why this cluster label fits the comment. The explanation should be factual and not include the category name.

Possible labels: ["Bug/Error", "Ease of use", "Early survey pop-up", "Findability/Navigation", "Sign in/access", "Other", "Integration"]

Output format:
Category: <Best Label>
Explanation: <1 sentence explanation of why the cluster label fits the comment>
"""

# Load the cache and calculate cache key
cache = load_cache()
cache_key = get_cache_key(df, "text-embedding-3-small", prompt_template)

# Check if we have cached results
if cache_key in cache:
    print("Loading results from cache...")
    cached_data = cache[cache_key]
    df["Cluster"] = cached_data["clusters"]
    df["Cluster Label"] = cached_data["labels"]
    df["Explanation"] = cached_data["explanations"]
else:
    print("No cache found, processing data...")
    # Convert comments into embeddings
    print("Generating text embeddings...")
    df["embedding"] = df["Comment"].apply(lambda x: get_embedding(x))
    embeddings = np.vstack(df["embedding"].values)

    # Run K-Means clustering
    print("Clustering comments into groups...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(embeddings)

    # Auto-label each cluster using GPT-4
    print("Labeling clusters with GPT-4...")
    cluster_labels = {}
    for cluster_id in range(num_clusters):
        cluster_comments = df[df["Cluster"] == cluster_id]["Comment"].tolist()
        label, _ = label_cluster(cluster_comments[:10])  # Get label from cluster
        cluster_labels[cluster_id] = label

    # Assign labels to dataframe
    df["Cluster Label"] = df["Cluster"].map(cluster_labels)

    # Generate individual explanations for each comment
    print("Generating explanations for each comment...")
    def explain_comment(row):
        _, explanation = label_cluster([row["Comment"]])  # Pass single comment as list
        return explanation
    
    df["Explanation"] = df.apply(explain_comment, axis=1)

    # Cache the results
    cache[cache_key] = {
        "clusters": df["Cluster"].tolist(),
        "labels": df["Cluster Label"].tolist(),
        "explanations": df["Explanation"].tolist()
    }
    save_cache(cache)

# Save results to Excel
df.to_excel("clustered_feedback.xlsx", index=False)
print("Results saved to clustered_feedback.xlsx.")
