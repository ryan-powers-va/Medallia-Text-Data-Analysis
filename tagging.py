import openai
import pandas as pd
import os
import json
from dotenv import load_dotenv
from hashlib import sha256

# === Setup ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set.")

client = openai.OpenAI(api_key=api_key)

# === File paths ===
input_file = "training_data/Unlabeled_Cleaned.xlsx"
output_file = "training_data/output/Tagged_Comments_Output.xlsx"
cache_dir = "cache"
checkpoint_dir = "checkpoints"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# === Sentiment Analysis ===
def analyze_sentiment(comment):
    """Analyze sentiment using a dedicated OpenAI API call, forcing either Positive or Negative."""
    prompt = f"""
    You are an expert in sentiment analysis. Analyze the sentiment of the following comment and classify it as Positive or Negative. Use Neutral only if the tone is completely neutral or unclear. 

    Comment: "{comment}"

    Output format:
    Sentiment: [Positive/Negative]
    """

    cached = from_cache(prompt)
    if cached and cached["sentiment"]:
        return cached["sentiment"]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant performing sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        sentiment = "[Error]"

        for line in output.split("\n"):
            if line.startswith("Sentiment:"):
                sentiment = line.split(":", 1)[1].strip()

        # Cache the sentiment (primary tag will be cached separately)
        to_cache(prompt, None, sentiment)
        return sentiment

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "[Error]"

# === Prompt construction ===
def build_prompt(comment):
    return f"""
You are analyzing user feedback from the VA.gov website.

For each user comment, assign:
- One **Primary Tag** from the list below. You must choose exactly one.
- Any number of **Secondary Tags** that apply (optional).

### Primary Tags (choose one):

1. Login & Access  
   Problems signing in, verifying ID, or accessing account. Includes ID.me/Login.gov.

2. Navigation & Usability  
   Feedback about the ease or difficulty of locating information, understanding the layout, 
   navigating menus, or completing tasks on the site. This includes both positive and negative experiences 
   with the site's structure or usability.

3. Claims & Benefits  
   Feedback about claim status, ratings, VA benefits applications, compensation, eligibility, or claims process delays.

4. Health Care
   Comments related to health care services, appointments, or providers. 
   This includes scheduling, managing, or understanding care appointments, providers, or pre-check-in. 
   Feedback about health data, labs, secure messaging, prescriptions, or portal tools.

5. Technical Performance  
   Errors, outages, crashes, load failures, or broken features. Language that may describe a technical bug, NOT usability or design problems. 

6. Education & GI Bill  
   Comments specifically about VA education benefits or school application issues.

7. Travel & Reimbursement  
   Trouble with travel pay, voucher systems, or mileage claims.

8. Customer Support / Contact 
   Difficulty reaching customer support of any type, such as long hold times, unresponsive phone/email/chat, 
   or lack of helpful assistance when contacting VA support channels.

10. Positive Experience  
    Praise or affirmation of VA.gov, staff, ease of use, or completed task. Should be used only for clearly positive comments that 
    are general praise, not specific to a feature or service.

11. Other / Unclear  
    Vague, unrelated, or non-actionable content (e.g. "just checking in").

### Secondary Tags (zero or more):

- Frustrated / Escalated Tone  
  Strong emotional tone, cursing, sarcasm, or repeated failed attempts.

- Mobile-Specific Friction  
  User is experiencing issue specifically on phone/tablet.

- Survey Timing (Early / Irrelevant) 
  Comment says feedback survey pop-up is too early, disruptive, or not relevant to the user's current experience. 

- System Down / Error Message  
  Explicit mention of an outage or major system error.

---

Return your answer in this format:

Primary Tag: [one primary tag]  
Secondary Tags: [comma-separated list of any secondary tags, or leave blank]  

Comment: "{comment}"

"""

# === Caching ===
def get_cache_path(text, model="gpt-3.5-turbo"):
    hash_id = sha256(text.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{model}_{hash_id}.json")

def from_cache(text, model="gpt-3.5-turbo"):
    path = get_cache_path(text, model)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def to_cache(text, primary, sentiment, model="gpt-3.5-turbo"):
    path = get_cache_path(text, model)
    with open(path, "w") as f:
        json.dump({"primary": primary, "sentiment": sentiment}, f, indent=2)

# === Main tagging function ===
def tag_comment(comment):
    prompt = build_prompt(comment)
    cached = from_cache(prompt)
    if cached:
        return cached["primary"], cached.get("secondary", "")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant tagging user feedback for UX analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        primary = "[Error]"
        secondary = ""

        for line in output.split("\n"):
            if line.startswith("Primary Tag:"):
                primary = line.split(":", 1)[1].strip()
            elif line.startswith("Secondary Tags:"):
                secondary = line.split(":", 1)[1].strip()

        # Cache the primary and secondary tags
        to_cache(prompt, primary, secondary)
        return primary, secondary

    except Exception as e:
        print(f"Error tagging comment: {e}")
        return "[Error]", ""

# === Run tagging ===
def run():
    print("\n=== Starting Tagging Process ===")
    print(f"Loading data from: {input_file}")
    df = pd.read_excel(input_file)
    df["Comment_Cleaned"] = df["Comment_Cleaned"].fillna("").astype(str)
    print(f"Loaded {len(df)} rows to process")
    print("(Press Ctrl+C to stop processing and save progress)\n")

    primary_tags = []
    secondary_tags = []
    sentiments = []
    skipped = 0
    checkpoint_size = 250 # Save checkpoint every 250 comments

    try:
        print("Processing comments...")
        for i, comment in enumerate(df["Comment_Cleaned"]):
            print(f"Progress: [{i + 1}/{len(df)}]", end="\r")
            # Skip empty or whitespace-only comments
            if not comment.strip():
                primary_tags.append("")
                secondary_tags.append("")
                sentiments.append("")
                skipped += 1
                continue

            primary, secondary = tag_comment(comment)  # API call for tagging
            sentiment = analyze_sentiment(comment)  # API call for sentiment

            primary_tags.append(primary)
            secondary_tags.append(secondary)
            sentiments.append(sentiment)

            # Save checkpoint every 300 comments
            if (i + 1) % checkpoint_size == 0:
                checkpoint_num = (i + 1) // checkpoint_size
                save_checkpoint(df, primary_tags, secondary_tags, sentiments, checkpoint_num)

    except KeyboardInterrupt:
        print("\n\n=== Processing Interrupted ===")
        print("Saving final checkpoint...")
        checkpoint_num = (len(primary_tags) // checkpoint_size) + 1
        save_checkpoint(df, primary_tags, secondary_tags, sentiments, checkpoint_num)
        print("Partial results saved. Exiting...")
        return

    print("\n\n=== Tagging Complete ===")
    print(f"Total rows processed: {len(df)}")
    print(f"Empty comments skipped: {skipped}")
    print(f"Comments tagged: {len(df) - skipped}")

    # Get the position of Comment_Cleaned column
    comment_col_idx = df.columns.get_loc("Comment_Cleaned")

    # Insert new columns right after Comment_Cleaned
    df.insert(comment_col_idx + 1, "Primary_Tag", primary_tags)
    df.insert(comment_col_idx + 2, "Secondary_Tags", secondary_tags)
    df.insert(comment_col_idx + 3, "Sentiment", sentiments)

    print(f"\nSaving results to: {output_file}")
    df.to_excel(output_file, index=False)
    print("Done! 🎉")

# === Checkpoint handling ===
def save_checkpoint(df, primary_tags, secondary_tags, sentiments, checkpoint_num):
    # Create a dataframe with only the processed rows
    df_checkpoint = df.iloc[:len(primary_tags)].copy()

    # Get the position of Comment_Cleaned column
    comment_col_idx = df_checkpoint.columns.get_loc("Comment_Cleaned")

    # Insert new columns right after Comment_Cleaned
    df_checkpoint.insert(comment_col_idx + 1, "Primary_Tag", primary_tags)
    df_checkpoint.insert(comment_col_idx + 2, "Secondary_Tags", secondary_tags)
    df_checkpoint.insert(comment_col_idx + 3, "Sentiment", sentiments)

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.xlsx")
    df_checkpoint.to_excel(checkpoint_file, index=False)
    print(f"\nCheckpoint saved: {checkpoint_file}")

if __name__ == "__main__":
    run()
