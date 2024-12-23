import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'C:/Users/OITCOPowerR/OneDrive - Department of Veterans Affairs/Documents/Medallia Text Data Analysis/.env')

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set in the environment.")

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=api_key
)

# File path for your Excel file
file_path = r"C:/Users/OITCOPowerR/OneDrive - Department of Veterans Affairs/Documents/Medallia Text Data Analysis/SecureMessages_Comments.xlsx"

# Load the data from Excel (ensure the Excel file has a column named "Comment")
df = pd.read_excel(file_path)

# Predefined tags
tags = [
    "Triage Group", "Unrelated to VA.gov", "Error", "Integration", "Ease of use",
    "Benefits", "Feature Request", "Supplies", "Other", "Mixed Status", "Can't Reply",
    "Early pop up", "Missing Rx", "Findability/ Navigation", "Sign in or access",
    "Content", "Sort", "Answered Question", "Can't Refill", "Page Length"
]

# Function to analyze a single comment
def analyze_comment(comment):
    prompt = f"""
    Classify the following comment into one of the predefined tags and determine its sentiment.
    Comment: "{comment}"
    Tags: {tags}
    Sentiments: [Positive, Negative, Neutral]  # Removed quotes for cleaner formatting

    Output the result in this format:
    Tag: Selected Tag
    Sentiment: Selected Sentiment
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an expert in text classification."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing comment: {e}")
        return None

# Apply the analysis to the comments
df["Analysis"] = df["Comment"].apply(analyze_comment)

# Split the analysis into Tag and Sentiment columns, and clean up any extra quotes
df["Tag"] = df["Analysis"].apply(lambda x: x.split("\n")[0].split(":")[1].strip().strip("'\"") if x else None)
df["Sentiment"] = df["Analysis"].apply(lambda x: x.split("\n")[1].split(":")[1].strip().strip("'\"") if x else None)

# Drop the intermediate "Analysis" column
df = df.drop(columns=["Analysis"])

# Save the results back to Excel
output_path = r"C:/Users/OITCOPowerR/OneDrive - Department of Veterans Affairs/Documents/Medallia Text Data Analysis/Tagged_Sentiments.xlsx"
df.to_excel(output_path, index=False)

print(f"Analysis complete! Results saved to {output_path}")
