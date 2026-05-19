import pandas as pd
import requests
import time
import os
import json

def run_pipeline(filepath):
    """
    Load product reviews, classify sentiment using an AI API, 
    and return a structured summary report.
    """
    # 1. Check if file exists and is not empty
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        return {}

    try:
        df = pd.read_csv(filepath)
        
        # 2. Data Cleaning: Drop rows with missing or empty review text
        df = df.dropna(subset=['review_text'])
        df = df[df['review_text'].str.strip() != ""]

        if df.empty:
            return {}

        # 3. AI Sentiment Classification via API
        texts = df['review_text'].tolist()
        api_token = os.getenv("HF_API_TOKEN")
        model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        # Payload for batch processing
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            sentiments = [max(res, key=lambda x: x['score'])['label'] for res in data]
            df['sentiment'] = sentiments
        else:
            # Return empty if API fails to maintain robustness
            return {}

        # 4. Aggregating Statistics
        total = len(df)
        pos_count = len(df[df['sentiment'] == 'POSITIVE'])
        neg_count = len(df[df['sentiment'] == 'NEGATIVE'])

        # 5. Summarizing by Product
        by_product = {}
        for prod, group in df.groupby('product'):
            by_product[prod] = {
                'POSITIVE': len(group[group['sentiment'] == 'POSITIVE']),
                'NEGATIVE': len(group[group['sentiment'] == 'NEGATIVE'])
            }

        # 6. Tie-breaking Logic (Alphabetical order)
        sorted_prods = sorted(by_product.keys())
        most_pos = max(sorted_prods, key=lambda x: by_product[x]['POSITIVE'])
        most_neg = max(sorted_prods, key=lambda x: by_product[x]['NEGATIVE'])

        # 7. Final Report Format
        return {
            'total_reviews': total,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'sentiment_rate': {
                'POSITIVE': round((pos_count / total) * 100, 1),
                'NEGATIVE': round((neg_count / total) * 100, 1)
            },
            'by_product': by_product,
            'most_positive_product': most_pos,
            'most_negative_product': most_neg
        }
    except Exception:
        return {}

# --- ADDED: Testing Section to display results in Terminal ---
if __name__ == "__main__":
    # Ensure "reviews.csv" exists in the same folder
    csv_file = "task3/reviews.csv" 
    
    print(f"--- Processing: {csv_file} ---")
    result = run_pipeline(csv_file)
    
    if result:
        # Print the result in a readable JSON format
        print(json.dumps(result, indent=4))
    else:
        print("Error: Could not generate report. Check your CSV or API Token.")