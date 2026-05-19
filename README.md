# AI & Data Science Technical Exercise

**Candidate:** Sreyneang Nhor  
**Track:** AI & Data Science  

## Project Overview
This project consists of three main tasks:
1. **Task 1: Data Analysis** - Analyzing sales data from a CSV file using pandas.
2. **Task 2: AI Model API** - Classifying text sentiment using Hugging Face's Inference API with exponential backoff retry logic.
3. **Task 3: AI Pipeline** - An end-to-end pipeline that combines data cleaning, API classification, and statistical reporting.

## Setup Instructions

### 1. Environment Variables
To run Task 2 and Task 3, you must set your Hugging Face API token as an environment variable. 
In PowerShell:
```powershell
$env:HF_API_TOKEN = "your_huggingface_token_here"