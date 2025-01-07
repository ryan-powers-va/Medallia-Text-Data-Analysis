from utils import TAGS
import openai
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from hashlib import sha256
import json
import logging
import shutil
import stat

# cache_directory = r"cache"
# def force_delete_readonly(func, path, exc_info):
#     # Change the permission and reattempt removal
#     os.chmod(path, stat.S_IWRITE)
#     func(path)

# def clear_cache():
#     try:
#         # Delete directory, handling read-only files
#         shutil.rmtree(cache_directory, onerror=force_delete_readonly)
#         print("Cache cleared successfully.")
#     except FileNotFoundError:
#         print("Cache directory does not exist. Nothing to clear.")
#     except PermissionError:
#         print("Permission denied. Ensure the directory is not open in another program.")
#     except Exception as e:
#         print(f"An error occurred while clearing the cache: {e}")

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
feedback_file = r"Comments_QuickTest.xlsx"
output_file = r"output.xlsx"
prompts_file = r"prompts.xlsx"
cache_directory = r"cache"

# Caching function to skip API calls if no prompt changes have occurred. 
def get_cache_file_path(comment, task):
    key = f"{comment}_{task}"
    hashed_key = sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_directory, f"{hashed_key}.json")

# Analyze tag for a comment using the openAI API. 
def analyze_tag(comment, human_tag, prompt):
    task = "tag"
    cache_file_path = get_cache_file_path(comment, task)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached_response = json.load(f)
            return cached_response.get("tag"), cached_response.get("confidence")
    examples = """
    Here are some examples:
    Comment: This site is VERY CONFUSING! We are not IT tech savvy
    Tag: Ease of use

    Comment: I was able to get a message to my primary care doctor and that was my goal
    Tag: Answered Question

    Comment: I came to reorder medication but didn't get an email or popup box
    Tag: Findability/Nav

    Comment: It is usually easier to go through the website versus making a phone call
    Tag: Ease of use

    Comment: This site is VERY CONFUSING! We are not IT tech savvy
    Tag: Ease of use

    Comment: Have to go through several layers to find what I need
    Tag: Ease of use

    Comment: This pop-up box pops up at the beginning just after signing in
    Tag: Early pop up

    Comment: Because I just opened the new VA format, and you are asking for a review before I have even had a chance to see it fully
    Tag: Early pop up
    
    Comment: Because I asked to send a message regarding canceling an appt due to continued reschedule of my Dental and you sent me to a Survey and now I am here performing a survey instead I should be cancelling my Dental Appt
    Tag: Early pop up

    Comment: Making sure my profile information is up to date
    Tag: Answered Question

    Comment: I was able to get a message to my primary care doctor and that was my goal
    Tag: Answered Question

    Comment: Page did not show the recent past refill request for my prescription
    Tag: Findability/Nav

    Comment: Hard to navigate
    Tag: Findability/Nav

    Comment: Everything keeps changing
    Tag: Integration

    Comment: Too difficult to manage site as compared to previous sites
    Tag: Integration

    Comment: I cannot message my provider due to changes in the platform
    Tag: Error

    Comment: I keep receiving an error message when trying to log in
    Tag: Error

    Comment: Great service
    Tag: Other

    Comment: I am trying to contact two separate clinics and cannot
    Tag: Other

    Comment: I very much appreciate the messaging system among other features
    Tag: Other

    Comment: "Website is easy to navigate. That's all you can ask for when online
    Tag: Ease of use

    Comment: Was not able to message the person I wanted
    Tag: Triage Group

    Comment: I cannot find the mental health department under the 'teams' list
    Tag: Triage Group

    Comment: I always feel lost when trying to locate the resources I need
    Tag: Findability/Nav

    Comment: The drop-down menus are overly complex, and the links aren't intuitive
    Tag: Findability/Nav

    Comment: I couldnt figure out how to access the prescription refill area without assistance
    Tag: Findability/Nav
    """
    prompt = f"""
    {prompt}
    Comment: "{comment}"
    {examples}
    Tags: {TAGS}

    Output format:
    Tag: Selected Tag
    """
    try:
        print("Calling OpenAI for comment tagging...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
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
            json.dump({"tag": tag, "confidence": confidence}, f, indent=4)

        return tag, confidence
    except Exception as e:
        print(f"Error processing tag: {e}")
        print(f"Debug: Failed to process comment: {comment}")
        return "[Error]", 0.0  # Explicitly return "[Error]" as a placeholder tag and a confidence of 0.0

# Analyze sentiment for a comment using the openAI API. 
def analyze_sentiment(comment, human_sentiment, prompt):
    task = "sentiment"
    cache_file_path = get_cache_file_path(comment, task)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached_response = json.load(f)
            return cached_response.get("sentiment"), cached_response.get("confidence")

    examples = """
    Comment: Making sure my profile information is up to date. 
    Sentiment: Neutral

    Comment: Access is sometimes problematic
    Sentiment: Neutral

    Comment: At times it can confusing to log in. You seem to have four different type of logins.
    Sentiment: Neutral
    """
    prompt = f"""
    {prompt}
    Comment: "{comment}"

    Output format:
    Sentiment: Selected Sentiment
    """
    try:
        print("Calling OpenAI for sentiment analysis...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if available
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
            json.dump({"sentiment": sentiment, "confidence": confidence}, f, indent=4)

        return sentiment, confidence
    except Exception as e:
        print(f"Error processing sentiment: {e}")
        print(f"Debug: Failed to process comment: {comment}")
        return "[Error]", 0.0  # Explicitly return "[Error]" as a placeholder sentiment and a confidence of 0.0

def generate_output():
    # Ensure cache directory exists. If not, create one.
    os.makedirs(cache_directory, exist_ok=True)

    # Load feedback data
    feedback_df = pd.read_excel(feedback_file)  # Ensure this file has "Comment", "Human Tag", "Human Sentiment" columns
    prompts_df = pd.read_excel(prompts_file)  # Ensure this file has a "Prompt" column

    # Ensure necessary columns are present in the feedback file
    if not {"Comment", "Human Tag", "Human Sentiment"}.issubset(feedback_df.columns):
        raise ValueError("Feedback file must contain 'Comment', 'Human Tag', and 'Human Sentiment' columns.")
    
    if not {"Prompt Type", "Prompt"}.issubset(prompts_df.columns):
        raise ValueError("Prompts file must contain 'Prompt Type' and 'Prompt' columns.")

    # Extract the prompts into a list
    tagging_prompts = prompts_df[prompts_df["Prompt Type"] == "Tagging"]["Prompt"].tolist()
    sentiment_prompts = prompts_df[prompts_df["Prompt Type"] == "Sentiment"]["Prompt"].tolist()

    # Initialize results list
    results = []

    # Loop through each comment and apply each prompt
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

        # Apply tagging prompts
        for i, prompt in enumerate(tagging_prompts):
            try:
                tag, tag_confidence = analyze_tag(comment, human_tag, prompt)
                result[f"Tagging Prompt {i + 1} Tag"] = tag.replace('"', '').replace("'", "")
                result[f"Tagging Prompt {i + 1} Confidence"] = tag_confidence
            except Exception as e:
                print(f"Error processing tagging with Prompt {i + 1}: {e}")
                result[f"Tagging Prompt {i + 1} Tag"] = "[Error]"
                result[f"Tagging Prompt {i + 1} Confidence"] = 0.0

        # Apply sentiment prompts
        for i, prompt in enumerate(sentiment_prompts):
            try:
                sentiment, sentiment_confidence = analyze_sentiment(comment, human_sentiment, prompt)
                result[f"Sentiment Prompt {i + 1} Sentiment"] = sentiment.replace('"', '').replace("'", "")
                result[f"Sentiment Prompt {i + 1} Confidence"] = sentiment_confidence
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

    print(f"Analysis complete! Results saved to {output_file}")

if __name__ == '__main__':
    generate_output()