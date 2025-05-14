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
    raise ValueError("OPENAI_API key is not set.")
client = openai.OpenAI(api_key=api_key)

# === Paths & dirs ===
input_file = "training_data/A11_Comments_Apr2025_cleaned.xlsx"
output_file = "training_data/output/Apr2025_Tagged.xlsx"
cache_dir = "cache"
checkpoint_dir = "April_checkpoints"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# === Checkpoint Directory ===
# Remove secondary_checkpoint_dir and use checkpoint_dir directly

# -----------------------------------------------------------------------------
# 1. Utility: generic cache helpers (now store arbitrary JSON payload)
# -----------------------------------------------------------------------------

def _cache_path(text: str, model: str = "gpt-3.5-turbo") -> str:
    h = sha256(text.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{model}_{h}.json")

def _cache_get(text: str, model: str = "gpt-3.5-turbo"):
    p = _cache_path(text, model)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def _cache_put(text: str, payload: dict, model: str = "gpt-3.5-turbo"):
    p = _cache_path(text, model)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)

# -----------------------------------------------------------------------------
# 2. Sentiment (unchanged)
# -----------------------------------------------------------------------------

def analyze_sentiment(comment: str) -> str:
    prompt = f"""
You are an expert in sentiment analysis. Classify the sentiment of the
following comment as **Positive**, **Negative**, or **Neutral**.
Comment: "{comment}"
Answer format:
Sentiment: <word>
"""
    if cached := _cache_get("sent:" + prompt):
        return cached["sentiment"]

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        out = resp.choices[0].message.content.strip()
        sent = next((ln.split(":", 1)[1].strip() for ln in out.split("\n") if ln.lower().startswith("sentiment")), "Neutral")
        _cache_put("sent:" + prompt, {"sentiment": sent})
        return sent
    except Exception as e:
        print("Sentiment error:", e)
        return "Error"

# -----------------------------------------------------------------------------
# 3. Primary‑tag prompt (unchanged except we return only primary)
# -----------------------------------------------------------------------------
def build_prompt(comment):
    return f"""
You are analyzing user feedback from the VA.gov website.

Tagging ground-rules
────────────────────
1. Assign **exactly one Primary Tag** from the list below; secondary tags come later.
2. Treat the keyword lists as clues, not a checklist. Use overall context and plain language.
3. Error first: if the core problem is a page, form, or button that fails to load, crashes, spins, or shows an HTTP/JS error → **Technical Performance** (even if the user also complains about navigation).
4. Healthcare first: if the comment deals with Rx refills, appointments, secure messaging, My HealtheVet, labs, telehealth, etc. → **Health Care** (regardless of navigation wording).
5. Login first: if the pain-point is signing in, verifying identity, or ID.me / Login.gov → **Login & Access** (even if the user praises ease of use).
6. Benefits first: if the topic is disability compensation, GI Bill, home loan, pension, burial, or other VA benefit → tag that benefit area according to the primary tag options, not Navigation.
7. Support first: if the main issue is phone/chat/email wait-time or agent quality → **Customer Support / Contact**.
8. **Navigation & Usability** is only for site-wide way-finding or layout feedback when **no single feature** dominates.
9. **Positive Feedback – General** is only for broad praise with **no feature mentioned** (“Great site—thank you!”).  
   – If praise names a feature (“Easy to schedule my appointment”) tag the feature (e.g. appointments would be > Healthcare), not Positive Feedback.
10. If the comment is too short, off-topic, or nonsensical → **Other / Unclear**.


### Primary Tags (choose one):

1. Login & Access  
Problems signing in, verifying identity, or accessing an account. Includes ID.me and Login.gov issues.  
Keywords: login, log in, sign in, sign-on, logged out, ID.me, idme, Login.gov, logingov, verify identity, verification code, two-factor, authentication, reauth, password reset, can't access account, credentials  

2. Navigation & Usability  
Difficulty finding information, navigating menus, or completing tasks due to layout or design.  
Includes both positive and negative feedback about the site's usability.  
Keywords: couldn't find, can't find, hard to locate, navigation, menu, breadcrumb, search bar, site map, too many steps, loop back, confusing, layout, where do I, scroll, tab order, drop-down, button placement, UX, UI, mobile menu, tablet view, font size  

3. Disability Claims  
Issues related to VA disability claims, including status, ratings, compensation, eligibility, applications, appeals, supplemental claims, or delays.  
Keywords: claim status, disability rating, C&P exam, compensation, service-connected, supplemental claim, appeal, higher-level review, claims backlog, benefit letter, eligibility, claim filed, VA Form 21-526, eBenefits transfer  

4. Health Care  
Feedback about VA health benefits, appointments, providers, My HealtheVet (MHV), secure messaging, labs, refill prescriptions, travel pay reimbursement, or health records.  
Includes scheduling, managing, or understanding care.  
Keywords: appointment, schedule visit, provider, My HealtheVet, MHV, secure message, labs, test results, refill, prescription, pharmacy, Blue Button, medical record, travel pay, community care, telehealth, urgent care, VA Form 10-10, copay, billing statement  

5. Technical Performance  
Errors, crashes, system failures, or broken features. Use only for bugs or technical malfunctions, not design issues.  
Keywords: error, 404, 500, 502, 504, gateway, white screen, blank page, spinning, loading forever, crashed, freeze, timeout, bug, technical issue, site down, server, maintenance, javascript error, unexpected token, cannot connect  

6. Benefits (Non-Health Care)  
Comments about education (GI Bill), housing, pension, burials, vocational rehab (VR&E), or home loans. Excludes health care and disability claims.  
Keywords: GI Bill, Post-9/11, COE, home loan, certificate of eligibility, chap 35, voc rehab, VR&E, VetSuccess, pension, survivor benefits, burial, headstone, plot allowance, dependency, DEA, housing grant, SAH, VET TEC  

7. Customer Support / Contact  
Trouble reaching support by phone, email, or chat. Includes long wait times, no response, or unhelpful service.  
Keywords: call center, 800-827-1000, hold for, wait time, no one answered, transferred, hung up, phone lines busy, chat support, agent, emailed, no response, help desk, support ticket, contact us, escalate, representative  

8. Positive Feedback - General
Clear praise for VA.gov, its tools, or staff. Only use for general positive feedback not tied to a specific feature.  
Keywords: great job, thank you, easy, quick, love the site, awesome, worked perfectly, smooth, no issues, appreciate, kudos, excellent, user-friendly, very helpful  

9. Other / Unclear  
Vague, unrelated, or non-actionable feedback. Includes off-topic or placeholder comments like "just checking in."  
Keywords: testing, n/a, none, just checking, blank, "." (single period), emojis only, spam URL, unrelated politics, incoherent text


---

Return your answer exactly like this:

Primary Tag: <one primary tag> 

Comment: "{comment}"

"""

def tag_comment(comment: str) -> str:
    """Return exactly one primary tag (cached)."""
    prompt = build_prompt(comment)

    if cached := _cache_get(prompt):
        return cached["primary"]            # ← pulled straight from cache

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant tagging user feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        out = resp.choices[0].message.content.strip()
        primary = next(
            (ln.split(":", 1)[1].strip() for ln in out.split("\n")
             if ln.lower().startswith("primary tag")),
            "[Error]",
        ).lstrip("0123456789. ").strip()

        _cache_put(prompt, {"primary": primary})   # write to cache
        return primary

    except Exception as e:
        print("Primary-tag error:", e)
        return "[Error]"
# -----------------------------------------------------------------------------
# 4. Secondary‑tag logic
# -----------------------------------------------------------------------------
SECONDARY_ELIGIBLE = {
    "Health Care",
    "Disability Claims",
    "Navigation & Usability",
    "Login & Access",
    "Technical Performance",
    "Benefits (Non-Health Care)",
    "Customer Support / Contact",
}

# -- mini‑taxonomy dict {primary: [(subtag, [keywords]), ... ]}
TAXONOMY = {
    "Health Care": [
        ("Appointment Scheduling", ["schedule appointment", "book appointment", "manage appointment", "vaos", "cancel visit", "reschedule", "clinic time"]),
        ("Rx Refill & Pharmacy", ["rx refill", "refill prescription", "prescription", "pharmacy", "meds", "medication", "drug refill", "champva rx"]),
        ("Secure Messaging", ["secure message", "mhv message", "inbox", "send message", "myhealthevet message", "provider message"]),
        ("Labs & Test Results", ["lab result", "test result", "blood work", "imaging", "x‑ray", "blue button", "download record"]),
        ("Travel Pay", ["travel pay", "beneficiary travel", "mileage reimbursement", "btsss", "travel reimbursement"]),
        ("Telehealth / Video Connect", ["video connect", "telehealth", "video visit", "virtual visit", "vc session"]),
        ("Community Care", ["community care", "outside provider", "ccn", "referral", "optum", "triwest"]),
    ],
    "Disability Claims": [
        ("File New Claim", ["21‑526", "new claim", "start claim", "file claim", "apply for disability"]),
        ("Check Claim Status", ["claim status", "track claim", "status of claim", "decision pending", "where is my claim"]),
        ("Upload Evidence", ["upload evidence", "submit documents", "supporting docs", "add evidence"]),
        ("Rating / Decision", ["rating", "decision letter", "percent", "service‑connected rating", "rating increase"]),
        ("Appeal / HLR / Supplemental", ["appeal", "hlr", "higher‑level review", "supplemental claim", "board appeal"]),
    ],
    "Navigation & Usability": [
        ("Site Search", ["search bar", "search box", "search field", "search results", "site search", "query"]),
        ("Menu & IA", ["menu", "dropdown", "navigation bar", "breadcrumb", "mega menu", "top nav", "information architecture"]),
        ("Page Layout", ["layout", "page layout", "font", "white space", "spacing", "clutter", "too busy"]),
        ("Mobile UX", ["mobile", "tablet", "phone", "responsive", "hamburger", "small screen", "scroll on mobile"]),
        ("Accessibility", ["screen reader", "aria", "wcag", "contrast", "keyboard nav", "tab order", "accessibility"]),
    ],
    "Login & Access": [
        ("ID.me Verification", ["id.me", "id dot me", "idme", "selfie", "face match", "verify identity", "driver's license"]),
        ("Login.gov Verification", ["login.gov", "login gov", "login dot gov", "gov login", "sign‑in partner"]),
        ("2FA / Codes", ["2fa", "two‑factor", "verification code", "security code", "sms code", "text code", "one‑time code", "otp", "authenticator", "auth"]),
        ("Password Reset", ["reset password", "forgot password", "change password", "password reset link"]),
        ("Account Lockout", ["locked out", "account locked", "too many attempts", "account disabled"]),
    ],
    "Technical Performance": [
        ("HTTP Errors", ["404", "500", "502", "503", "504", "bad gateway", "gateway timeout", "http error"]),
        ("Timeout / Spinning", ["timeout", "timed out", "spinning", "loading forever", "never loads", "progress wheel"]),
        ("Crash / Blank", ["blank page", "white screen", "page crashed", "javascript error", "unexpected token"]),
        ("Login Service Outage", ["id.me down", "login.gov down", "login service unavailable", "idme outage", "login outage"]),
        ("PDF / Upload Failure", ["upload failed", "cannot upload", "upload error", "download pdf", "pdf won't open", "pdf error"]),
    ],
    "Benefits (Non-Health Care)": [
        ("Education / GI Bill", ["gi bill", "education benefit", "22‑1990", "school cert", "post‑9/11", "chapter 33"]),
        ("Home Loan COE", ["coe", "certificate of eligibility", "home loan", "mortgage", "va loan"]),
        ("Voc Rehab / VR&E", ["vr&e", "voc rehab", "chapter 31", "vetsuccess"]),
        ("Burial & Memorial", ["burial", "headstone", "plot allowance", "preneed", "interment"]),
        ("Pension / Survivor", ["survivor pension", "dic", "dependency", "pension"]),
    ],
    "Customer Support / Contact": [
        ("Wait Time", ["800‑827", "call center", "on hold", "phone wait", "hold time", "phone line"]),
        ("Chat / Live Agent", ["chat support", "chat agent", "live chat", "virtual agent"]),
        ("Email / Online Form", ["emailed", "no response email", "contact form", "web inquiry", "support email"]),
        ("Warm Transfer", ["transferred", "transfer call", "hung up", "dropped call", "escalate", "disconnect", "lost call"]),
        ("In‑Person Assistance", ["regional office", "vso", "in‑person", "visitor", "walk‑in"]),
    ],
}

def secondary_tag(primary: str, comment: str) -> str:
    """Return one sub‑tag or empty string if not eligible/none matches."""
    if primary not in SECONDARY_ELIGIBLE:
        return ""

    # --- simple keyword pass first (fast) ---
    c_lower = comment.lower()
    for sub, kws in TAXONOMY[primary]:
        if any(k in c_lower for k in kws):
            return sub

    # --- fallback to LLM if keyword miss (rare) ---
    prompt = f"""You are classifying VA.gov feedback. The primary tag is **{primary}**.
Choose exactly one sub‑tag from this list (else say "Other {primary}"):
{[x[0] for x in TAXONOMY[primary]]}
Comment: "{comment}"
Answer format:
Sub-Tag: <name>
"""
    if cached := _cache_get("sub:" + prompt):
        return cached["sub_tag"]

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful tagging assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        out = resp.choices[0].message.content.strip()
        sub = next((ln.split(":",1)[1].strip() for ln in out.split("\n") if ln.lower().startswith("sub-tag")), f"Other {primary}")
        _cache_put("sub:" + prompt, {"sub_tag": sub})
        return sub
    except Exception as e:
        print("Secondary tag error:", e)
        return f"Other {primary}"

# -----------------------------------------------------------------------------
# 5. Main loop (adds Secondary_Tag + Secondary_Tag_Family columns)
# -----------------------------------------------------------------------------

def save_checkpoint_csv(df, primary_tags, sentiments, sub_tags, sub_families, checkpoint_num):
    """Save a checkpoint CSV file with the processed rows."""
    df_checkpoint = df.iloc[:len(primary_tags)].copy()

    # Insert new columns
    ci = df_checkpoint.columns.get_loc("Comment_Cleaned")
    df_checkpoint.insert(ci + 1, "Primary_Tag", primary_tags)
    df_checkpoint.insert(ci + 2, "Secondary_Tag_Family", sub_families)
    df_checkpoint.insert(ci + 3, "Secondary_Tag", sub_tags)
    df_checkpoint.insert(ci + 4, "Sentiment", sentiments)

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.csv")
    df_checkpoint.to_csv(checkpoint_file, index=False)
    print(f"\nCheckpoint saved: {checkpoint_file}")

def run():
    print("\n=== Tagging Process (Primary + Secondary) ===")
    df = pd.read_excel(input_file)
    df["Comment_Cleaned"] = df["Comment_Cleaned"].fillna("").astype(str)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows")

    primary_tags, sentiments, sub_tags, sub_families = [], [], [], []
    checkpoint_size = 100  # Save checkpoint every 100 rows
    checkpoint_num = 0

    try:
        for i, comment in enumerate(df["Comment_Cleaned"]):
            if not comment.strip():
                primary_tags.append("")
                sentiments.append("")
                sub_tags.append("")
                sub_families.append("")
                continue

            primary = tag_comment(comment)
            sentiment = analyze_sentiment(comment)
            sub = secondary_tag(primary, comment)
            fam = primary if sub else ""

            primary_tags.append(primary)
            sentiments.append(sentiment)
            sub_tags.append(sub)
            sub_families.append(fam)

            # Print progress bar
            progress = (i + 1) / total_rows
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f'\rProgress: [{bar}] {i+1}/{total_rows} ({progress:.1%})', end='')

            # Save checkpoint every `checkpoint_size` rows
            if (i + 1) % checkpoint_size == 0:
                checkpoint_num += 1
                save_checkpoint_csv(df, primary_tags, sentiments, sub_tags, sub_families, checkpoint_num)
                print(f"\nCheckpoint {checkpoint_num} saved")

        print("\n")  # New line after progress bar

    except KeyboardInterrupt:
        print("\n\n=== Processing Interrupted ===")
        print(f"Last processed row: {i+1}/{total_rows} ({(i+1)/total_rows:.1%})")
        print("Saving final checkpoint...")
        checkpoint_num += 1
        save_checkpoint_csv(df, primary_tags, sentiments, sub_tags, sub_families, checkpoint_num)
        print("Partial results saved. Exiting...")
        return

    # Insert new columns
    ci = df.columns.get_loc("Comment_Cleaned")
    df.insert(ci + 1, "Primary_Tag", primary_tags)
    df.insert(ci + 2, "Secondary_Tag_Family", sub_families)
    df.insert(ci + 3, "Secondary_Tag", sub_tags)
    df.insert(ci + 4, "Sentiment", sentiments)

    print(f"\nSaving results to: {output_file}")
    df.to_excel(output_file, index=False)
    print("Done! 🎉")

if __name__ == "__main__":
    run()