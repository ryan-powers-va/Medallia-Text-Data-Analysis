# VA.gov Feedback Analysis with GPT-3.5

## Overview
This project uses OpenAI's GPT-3.5-turbo model to analyze and classify user feedback from VA.gov. The `genAI_tagging.py` script processes comments to assign primary tags, secondary tags, and sentiment analysis, with built-in caching and checkpointing for reliability.

## Features
- **Primary Tagging**: Assigns one of nine predefined primary tags to each comment (e.g., Login & Access, Health Care, Technical Performance)
- **Secondary Tagging**: For eligible primary tags, assigns specific sub-tags (e.g., "Appointment Scheduling" under Health Care)
- **Sentiment Analysis**: Classifies comments as Positive, Negative, or Neutral
- **Caching System**: Stores API responses to avoid redundant calls and reduce costs
- **Checkpoint System**: Saves progress every 100 rows to prevent data loss
- **Error Handling**: Gracefully handles API errors and interruptions

## File Structure
```
.
├── genAI_tagging.py          # Main processing script
├── training_data/           
│   ├── A11_Comments_*.xlsx   # Input data file
│   └── output/              # Processed results
├── cache/                   # API response cache
├── April_checkpoints/       # Processing checkpoints
└── .env                     # Environment variables (API key)
```

## Setup

### Prerequisites
- Python 3.8+ (tested with 3.11)
- OpenAI API key
- Required packages:
  - `openai`
  - `pandas`
  - `openpyxl`
  - `python-dotenv`

### Installation
1. Install dependencies:
   ```bash
   pip install openai pandas openpyxl python-dotenv
   ```

2. Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Create required directories:
   ```bash
   mkdir cache April_checkpoints
   ```

## Usage
1. Place your input Excel file in the `training_data` directory
2. Update the input/output file paths in `genAI_tagging.py` if needed
3. Run the script:
   ```bash
   python genAI_tagging.py
   ```

## Tag Categories

### Primary Tags
1. Login & Access
2. Navigation & Usability
3. Disability Claims
4. Health Care
5. Technical Performance
6. Benefits (Non-Health Care)
7. Customer Support / Contact
8. Positive Feedback - General
9. Other / Unclear

### Secondary Tags
Available for most primary tags, providing more specific categorization. For example:
- Health Care: Appointment Scheduling, Rx Refill, Secure Messaging, etc.
- Technical Performance: HTTP Errors, Timeout/Spinning, Crash/Blank, etc.
- Login & Access: ID.me Verification, Login.gov Verification, 2FA/Codes, etc.

## Performance
- Uses GPT-3.5-turbo for all classifications, cheapest. Other models are available as required
- Implements caching to reduce API calls
- Saves checkpoints every 100 rows, can change checkpoint interval as required
- Handles interruptions gracefully, allowing resume from last checkpoint

## Notes
- The script includes detailed tagging rules and keyword lists for accurate classification
- Comments are processed sequentially with progress tracking
- Empty or whitespace-only comments are skipped
- All API responses are cached to improve performance and reduce costs
