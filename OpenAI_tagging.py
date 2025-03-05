import openai
import pandas as pd
from openpyxl import load_workbook
import os
from dotenv import load_dotenv
import sys
from hashlib import sha256
import json
import logging
import shutil
import stat
from utils import TAGS
from utils import get_cache_file_path
# from utils import clear_cache

# clear_cache()

logging.basicConfig(
    filename="debug_log.txt",      # Log file name
    level=logging.DEBUG,           # Log everything (DEBUG level)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv(dotenv_path=r'.env')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set in the environment.")

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=api_key
)

# File paths
feedback_file = r"main_test.xlsx"
output_file = r"output.xlsx"
prompts_file = r"prompts.xlsx"
cache_directory = r"cache"


# Analyze tag for a comment using the openAI API. 
def analyze_tag(comment, human_tag, prompt):
    task = "tag"
    model = "gpt-3.5-turbo"
    cache_file_path = get_cache_file_path(comment, task, model, prompt)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached_response = json.load(f)
            return (
                cached_response.get("tag", "[Error]"),
                cached_response.get("confidence", 0.0),
                cached_response.get("model", "unknown")
            )

    examples = """
    Here are examples of how to tag user feedback. Study these carefully:

    Comment: "This site is VERY CONFUSING! We are not IT tech savvy and can't figure out how to use basic features"
    Tag: Ease of use

    Comment: "I spent 20 minutes trying to find where to reorder my prescription. The menu structure makes no sense"
    Tag: Findability/Nav

    Comment: "Why did you change everything? The previous version was much easier to understand and use"
    Tag: Integration

    Comment: "I just logged in and immediately got this survey. Let me use the site first!"
    Tag: Early pop up

    Comment: "The site works fine, no issues to report"
    Tag: Other
    """

    prompt = f"""
    You are an expert at analyzing user feedback for website usability.
    Your task is to categorize the following comment into exactly ONE of these tags:
    {TAGS}

    Critical Guidelines:
    1. Identify the PRIMARY issue - what is the user's main concern?
    2. Look for explicit mentions of:
       - Difficulty using features (Ease of use)
       - Navigation/finding items (Findability/Nav)
       - System changes/comparisons (Integration)
       - Survey timing issues (Early pop up)
    3. If multiple issues appear, determine which one receives the most emphasis
    4. Use "Other" only when no main issue clearly fits the other categories

    {examples}

    Comment: "{comment}"

    Output format:
    Tag: [Selected Tag]
    """

    try:
        print("Calling OpenAI for comment tagging...")
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        tag = None
        confidence = 0.75  # Placeholder confidence; replace with actual logic later
        for line in output.split("\n"):
            if line.startswith("Tag:"):
                tag = line.split(":", 1)[1].strip().strip('"').strip("'")
                # Heuristic confidence calculation
                if tag == human_tag:  # Match with human tag
                    confidence = 1.0  # High confidence
                else:
                    confidence = 0.75  # Medium confidence for non-matching tags

        # Cache the response
        with open(cache_file_path, "w") as f:
            json.dump({"model": model, "tag": tag, "confidence": confidence}, f, indent=4)

        return tag, confidence, model
    except Exception as e:
        print(f"Error processing tag: {e}")
        print(f"Debug: Failed to process comment: {comment}")
        return "[Error]", 0.0, model

# Analyze sentiment for a comment using the openAI API. 
def analyze_sentiment(comment, human_sentiment, prompt):
    model = "gpt-3.5-turbo"
    task = "sentiment"
    cache_file_path = get_cache_file_path(comment, task, model, prompt)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached_response = json.load(f)
            return (
                cached_response.get("sentiment", "[Error]"),
                cached_response.get("confidence", 0.0),
                cached_response.get("model", "unknown")
            )

    examples = """
    """
    prompt = f"""
    {examples}
    {prompt}
    Comment: "{comment}"

    Output format:
    Sentiment: Selected Sentiment
    """
    try:
        print("Calling OpenAI for sentiment analysis...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        sentiment = None
        confidence = 0.75  # Placeholder confidence; replace with actual logic later
        for line in output.split("\n"):
            if line.startswith("Sentiment:"):
                sentiment = line.split(":", 1)[1].strip()
                # Confidence scoring logic
                if sentiment == human_sentiment:  # Match with human sentiment
                    confidence = 1.0  # High confidence
                else:
                    confidence = 0.75  # Medium confidence for non-matching sentiments

        # Cache the response
        with open(cache_file_path, "w") as f:
            json.dump({"model": model, "sentiment": sentiment, "confidence": confidence}, f, indent=4)

        return sentiment, confidence, model
    except Exception as e:
        print(f"Error processing sentiment: {e}")
        print(f"Debug: Failed to process comment: {comment}")
        return "[Error]", 0.0, model

def generate_output():
    # Ensure cache directory exists. If not, create one.
    os.makedirs(cache_directory, exist_ok=True)

    # Load feedback data
    feedback_df = pd.read_excel(feedback_file)
    prompts_df = pd.read_excel(prompts_file)

    # Ensure necessary columns are present in the feedback file
    if not {"Comment", "Human Tag", "Human Sentiment"}.issubset(feedback_df.columns):
        raise ValueError("Feedback file must contain 'Comment', 'Human Tag', and 'Human Sentiment' columns.")
    
    if not {"Prompt Type", "Prompt"}.issubset(prompts_df.columns):
        raise ValueError("Prompts file must contain 'Prompt Type' and 'Prompt' columns.")

    # Extract only sentiment prompts
    sentiment_prompts = prompts_df[prompts_df["Prompt Type"] == "Sentiment"]["Prompt"].tolist()

    # Initialize results list
    results = []

    # Loop through each comment
    for _, row in feedback_df.iterrows():
        comment = row["Comment"]
        human_tag = row["Human Tag"]
        human_sentiment = row["Human Sentiment"]

        # Create a dictionary to store results for the current comment
        result = {
            "Input": comment,
            "Human Tag": human_tag,
            "Human Sentiment": human_sentiment
        }

        # Single tagging analysis with optimized prompt
        try:
            tag, tag_confidence, tag_model = analyze_tag(comment, human_tag, "")  # Empty prompt as it's now built into analyze_tag
            result["AI Tag"] = tag.replace('"', '').replace("'", "")
            result["Tag Confidence"] = tag_confidence
            result["Tag Model"] = tag_model
        except Exception as e:
            print(f"Error processing tagging: {e}")
            result["AI Tag"] = "[Error]"
            result["Tag Confidence"] = 0.0

        # Apply sentiment prompts (keeping multiple prompts for sentiment)
        for i, prompt in enumerate(sentiment_prompts):
            try:
                sentiment, sentiment_confidence, sentiment_model = analyze_sentiment(comment, human_sentiment, prompt)
                result[f"Sentiment Prompt {i + 1} Sentiment"] = sentiment.replace('"', '').replace("'", "")
                result[f"Sentiment Prompt {i + 1} Confidence"] = sentiment_confidence
                result[f"Sentiment Prompt {i + 1} Model"] = sentiment_model
            except Exception as e:
                print(f"Error processing sentiment with Prompt {i + 1}: {e}")
                result[f"Sentiment Prompt {i + 1} Sentiment"] = "[Error]"
                result[f"Sentiment Prompt {i + 1} Confidence"] = 0.0

        # Append the results to the results list
        results.append(result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to an output Excel file
    results_df.to_excel(output_file, index=False)

    wb = load_workbook(output_file)
    sheet = wb.active

    # Update formula to match new column structure
    sheet["C36"] = '=COUNTIF(E2:E35, "1") / COUNTA(E2:E35)'  # Adjust column letter based on your new structure
    sheet["F36"] = '=COUNTIF(H2:H35, "1") / COUNTA(H2:H35)'  # Adjust column letter based on your new structure


    # Save the workbook with formulas
    wb.save(output_file)

    print(f"Analysis complete! Results saved to {output_file}")

if __name__ == '__main__':
    generate_output()