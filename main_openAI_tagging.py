from utils import TAGS
import openai
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from hashlib import sha256
import json

load_dotenv(dotenv_path=r'.env')

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set in the environment.")

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=api_key
)

# File paths
feedback_file = r"Comments_QuickTest.xlsx"
prompts_file = r"prompts.xlsx"
output_file = r"output.xlsx"
cache_directory = r"cache"

# Caching function to skip API calls if no prompt changes have occurred. 
def get_cache_file_path(comment, prompt):
    key = f"{comment}_{prompt}"
    hashed_key = sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_directory, f"{hashed_key}.json")

# Function to analyze a single comment with a specific prompt
def analyze_comment_with_prompt(comment, prompt):
    cache_file_path = get_cache_file_path(comment, prompt)
    # If the response file already exists, load the cached response and skip the API call. 
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached_response = json.load(f)
            return cached_response.get("tag"), cached_response.get("sentiment")
    full_prompt = f"""
    {prompt}
    Comment: "{comment}"
    Tags: {TAGS}
    Sentiments: [Positive, Negative, Neutral]

    Here are some examples:
    Comment: "This site is VERY CONFUSING!  We are not IT tech savvy!  Make it simple for us to send and receive messages from our Medical Providers!"
    Tag: Ease of use
    Sentiment: Negative

    Comment: "website easy to use and I can usually find what I need, if not someone is always available to help"
    Tag: Findability/Nav
    Sentiment: Positive

    Comment: "I was able to get a message to my primary care doctor and that was my goal"
    Tag: Answered Question
    Sentiment: Neutral

    Comment: "I came to reorder medication but didn't get an email or popup box telling me that it had been received or the date it would be shipped."
    Tag: Findability/Nav
    Sentiment: Neutral

    Comment: "It is usually easier to go through the website versus making a phone call."
    Tag: Ease of use
    Sentiment: Neutral

    Please consider the following:
    1. Use the tag that most accurately reflects the topic of the comment.
    2. If multiple tags seem applicable, choose the one that best captures the main idea.
    3. If a comment is neutral or unclear, classify it as "Neutral" sentiment.

    Output the result in this format:
    Tag: Selected Tag
    Sentiment: Selected Sentiment
    """
    try:
        print("Calling OpenAI...")
        # Model selection. 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" - but 4 is more expensive**

            # Establishes role and context for the system and the user.
            messages=[
                {"role": "system", "content": "You are an expert in text classification and user experience design for the VA.gov website and benefits ecosystem. You are tasked with analyzing open-text feedback to pinpoint potential UX issues."},
                {"role": "user", "content": full_prompt}
            ],
            
            # Controls randomness, ensuring determinstic outputs accross results.
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        # Parse the output
        tag = None
        sentiment = None
        for line in output.split("\n"):
            if line.startswith("Tag:"):
                tag = line.split(":", 1)[1].strip()
            elif line.startswith("Sentiment:"):
                sentiment = line.split(":", 1)[1].strip()

        # Save the result to a file
        with open(cache_file_path, "w") as f:
            json.dump({"prompt": prompt, "comment": comment, "tag": tag, "sentiment": sentiment}, f, indent=4)
        
        return tag, sentiment
    except Exception as e:
        print(f"Error processing comment: {e}")
        return None, None

def generate_output():

    # Ensures cache directory exists. If not, makes one. 
    os.makedirs(cache_directory, exist_ok=True)

    # Load human-labeled feedback and prompts
    feedback_df = pd.read_excel(feedback_file)  # Ensure this file has "Comment", "Human Tag", "Human Sentiment" columns
    prompts_df = pd.read_excel(prompts_file)  # Ensure this file has a "Prompt" column

    # Ensure necessary columns are present in the feedback file
    if not {"Comment", "Human Tag", "Human Sentiment"}.issubset(feedback_df.columns):
        raise ValueError("Feedback file must contain 'Comment', 'Human Tag', and 'Human Sentiment' columns.")

    # Extract the prompts into a list
    prompts = prompts_df["Prompt"].tolist()

    # Initialize results dictionary
    results = []

    # Loop through each comment and apply each prompt
    for _, row in feedback_df.iterrows():
        comment = row["Comment"]
        human_tag = row["Human Tag"]
        human_sentiment = row["Human Sentiment"]

        result = {
            "Input": comment,
            "Human Tag": human_tag,
            "Human Sentiment": human_sentiment
        }

        for i, prompt in enumerate(prompts):
            tag, sentiment = analyze_comment_with_prompt(comment, prompt)
            result[f"Prompt {i + 1} Tag"] = tag
            result[f"Prompt {i + 1} Sentiment"] = sentiment

        results.append(result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to an output Excel file
    results_df.to_excel(output_file, index=False)

    print(f"Analysis complete! Results saved to {output_file}")

if __name__ == '__main__':
    generate_output()
