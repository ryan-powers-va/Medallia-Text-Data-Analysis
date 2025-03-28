import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
from utils import get_cache_file_path

# ========== SETUP ==========
load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in .env")

client = OpenAI(api_key=api_key)

INPUT_FILE = "data/small_test.xlsx"    # Must have "Comment" and "Tag" columns
OUTPUT_FILE = "data/ux_test_output.xlsx"
CACHE_DIR = "ux_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== CONFIG ==========
UX_KEYWORDS = [
    "hard to find", "confusing", "can’t find", "can't find", "unclear",
    "too many steps", "not intuitive", "difficult to", "didn’t work", "didn't work",
    "site is slow", "popup", "didn’t expect", "unexpected", "everything keeps changing"
]
UX_PROMPT = """Does the following user comment raise a UX (user experience) concern? If yes, classify the type of concern.

UX Types:
- Navigation
- Clarity/Language
- Technical/Performance
- Interruption/Flow
- Other

Comment: "{comment}"

Output:
UX Concern: Yes/No
UX Type: [Type if Yes, else None]
"""

SENTIMENT_PROMPT = """You are a sentiment classifier. For each comment, classify it as either Positive or Negative.

Only use the two labels: Positive or Negative.
Neutral, unclear, or mixed sentiments should be treated as Negative.

Examples:
Comment: "I love this new layout. It's easy to use and fast."
Sentiment: Positive

Comment: "I couldn’t find what I was looking for. Very frustrating."
Sentiment: Negative

Comment: "The site worked fine but then crashed while I was checking out."
Sentiment: Negative

Comment: "{comment}"

Output:
Sentiment:"""

# ========== HELPERS ==========

def extract_sentiment(output):
    for line in output.splitlines():
        lower_line = line.lower()
        if "negative" in lower_line:
            return "Negative"
        elif "positive" in lower_line:
            return "Positive"
    return "Unknown"

def contains_ux_keywords_fuzzy(comment, keywords=UX_KEYWORDS, threshold=75):
    comment_lower = comment.lower()
    return any(fuzz.partial_ratio(comment_lower, k.lower()) >= threshold for k in keywords)

def analyze_sentiment(comment):
    prompt = SENTIMENT_PROMPT.format(comment=comment)
    model = "gpt-3.5-turbo"
    task = "sentiment"
    cache_path = get_cache_file_path(comment, task, model, prompt)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)["sentiment"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        sentiment = extract_sentiment(output)
        with open(cache_path, "w") as f:
            json.dump({"sentiment": sentiment}, f)
        return sentiment
    except Exception as e:
        print(f"[Sentiment Error] {e}")
        return "Error"

def analyze_ux_concern(comment):
    prompt = UX_PROMPT.format(comment=comment)
    model = "gpt-3.5-turbo"
    task = "ux"
    cache_path = get_cache_file_path(comment, task, model, prompt)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
            return cached["ux_concern"], cached["ux_type"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        concern, concern_type = "No", "None"
        for line in output.split("\n"):
            if line.lower().startswith("ux concern:"):
                concern = line.split(":", 1)[1].strip()
            elif line.lower().startswith("ux type:"):
                concern_type = line.split(":", 1)[1].strip()
        with open(cache_path, "w") as f:
            json.dump({"ux_concern": concern, "ux_type": concern_type}, f)
        return concern, concern_type
    except Exception as e:
        print(f"[UX Concern Error] {e}")
        return "Error", "Error"

def compute_ux_score(sentiment, keyword_hit, ux_flag):
    score = 0
    if sentiment.lower() == "negative":
        score += 1
    if keyword_hit:
        score += 1
    if ux_flag.lower() == "yes":
        score += 1
    return score  # Max = 3

# ========== MAIN ==========
def main():
    df = pd.read_excel(INPUT_FILE)
    if not {"Comment", "Tag"}.issubset(df.columns):
        raise ValueError("Input file must contain 'Comment' and 'Tag' columns")

    prediction_cols = {
        "Sentiment": [],
        "UX Keyword Flag": [],
        "UX Concern GPT": [],
        "UX Type GPT": [],
        "UX Composite Score": []
    }

    for _, row in df.iterrows():
        comment = row["Comment"]
        print(f"Processing: {comment[:60]}...")

        keyword_flag = contains_ux_keywords_fuzzy(comment)
        sentiment = analyze_sentiment(comment)
        ux_flag, ux_type = analyze_ux_concern(comment)
        score = compute_ux_score(sentiment, keyword_flag, ux_flag)

        prediction_cols["Sentiment"].append(sentiment)
        prediction_cols["UX Keyword Flag"].append(keyword_flag)
        prediction_cols["UX Concern GPT"].append(ux_flag)
        prediction_cols["UX Type GPT"].append(ux_type)
        prediction_cols["UX Composite Score"].append(score)

    for col_name, values in prediction_cols.items():
        df[col_name] = values

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n🎉 Done! Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
