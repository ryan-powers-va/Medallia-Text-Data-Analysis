# OpenAI-Powered Feedback Analysis

## Overview
This project leverages OpenAI's GPT models to analyze and classify user feedback from a dataset. It automates tagging and sentiment analysis using dynamically generated prompts, caching for efficiency, and heuristic confidence scoring for results. The processed data is output to an Excel file, including accuracy calculations for evaluation.

## Features
- **Tagging**: Automatically assigns a predefined tag to each comment based on its main topic.
- **Sentiment Analysis**: Determines the sentiment (Positive, Neutral, or Negative) of each comment.
- **Dynamic Prompts**: Reads tagging and sentiment instructions from a separate prompts file for flexibility.
- **Caching**: Avoids redundant API calls by storing results for previously processed comments.
- **Error Handling**: Logs errors and skips problematic entries to ensure smooth execution.
- **Excel Output**: Saves processed results with accuracy formulas to an Excel file.

## File Structure
- main_openAI_tagging.py / main script for processing and analysis
- prompts.xlsx / Prompts file with tagging and sentiment instructions
- main_test.xlsx / Input dataset with user comments and human labels
- output.xlsx / Output file with processed results
- utils.py / Helper functions for caching and other logic
- debug_log.txt / Debug file for tracking issues
- cache/ - Directory for caching API results
- .env / Environment variables, including API key. 

## Setup

### Prerequisites
1. Python 3.8 or higher installed (don't use the latest version, I ran it with 3.11).
2. OpenAI API key.
3. Required Python packages:
   - `openai`
   - `pandas`
   - `openpyxl`
   - `python-dotenv`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Create a .env file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
6. Set up cache directory:
   ```bash
   mkdir cache
## Current Benchmarks - as of 1/7

**GPT-4**

**Tagging:** 57.14%

**Sentiment:** 85.71%

----
**GPT-3.5-turbo**

**Tagging:** 64.71%

**Sentiment:** 79.41%

----
**Test sample size:** 35
