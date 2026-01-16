import os
import csv
import pandas as pd
from datetime import datetime
from openai import OpenAI

# CONFIG - Replace with your REAL keys
OPENAI_API_KEY = "your_api_key_as_variable_goes_here"
GOOGLE_API_KEY = "your_api_key_as_variable_goes_here"

INPUT_CSV = "queries.csv"
CHATGPT_MODEL = "gpt-5.1"
GEMINI_MODEL = "gemini-3-flash-preview"

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def load_queries_enhanced(csv_path: str) -> dict:
    """Load queries.csv with new columns and group by funnel."""
    df = pd.read_csv(csv_path)
    print(f"CSV columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Combine query + addition for full prompt
    df['full_query'] = df['query'].astype(str) + ' ' + df['addition'].astype(str)
    
    # Group by funnel
    funnels = {}
    for funnel in ['High', 'Mid', 'Low']:
        funnel_df = df[df['funnel'].str.contains(funnel, case=False, na=False)]
        funnels[funnel.lower()] = funnel_df.to_dict('records')
        print(f"{funnel} funnel: {len(funnels[funnel.lower()])} queries")
    
    return funnels

def create_location_context(geography: str) -> str:
    """Create location context to simulate query from specific geography."""
    if pd.isna(geography) or str(geography).lower() == 'all':
        return "You are responding to a global traveler with no specific location bias."
    
    location_contexts = {
        'Los Angeles': "You are helping someone currently located in Los Angeles, California, USA. Consider local context, events, and preferences when relevant.",
        'San Francisco': "You are helping someone currently located in San Francisco, California, USA. Consider local context, events, and preferences when relevant.",
        'Seatlet': "You are helping someone currently located in Seattle, Washington, USA. Consider local context, events, and preferences when relevant.",
        'Australia': "You are helping someone currently located in Australia. Consider local context, events, and preferences when relevant."
    }
    
    return location_contexts.get(geography, f"You are helping someone currently located in {geography}. Consider local context when relevant.")

def call_chatgpt(client, full_query: str, geography: str) -> str:
    """Call OpenAI GPT with location context."""
    location_context = create_location_context(geography)
    
    try:
        completion = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {"role": "system", "content": f"{location_context}\n\nYou are a helpful travel assistant. Provide sources where applicable."},
                {"role": "user", "content": full_query},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_gemini(client, full_query: str, geography: str) -> str:
    """Call Gemini with location context."""
    location_context = create_location_context(geography)
    
    try:
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": f"{location_context}\n\nYou are a helpful travel assistant. Provide sources where applicable."},
                {"role": "user", "content": full_query},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_funnel(funnel_name: str, queries: list, output_file: str):
    """Process queries for one funnel and save to CSV."""
    file_exists = os.path.exists(output_file)
    
    with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writerow([
                "timestamp", "category", "funnel", "geography", "original_query", "addition", 
                "full_query", "location_context", "chatgpt_model", "chatgpt_response",
                "gemini_model", "gemini_response"
            ])
        
        print(f"\n=== Processing {funnel_name} funnel ({len(queries)} queries) ===")
        for i, row in enumerate(queries, 1):
            geography = row.get('geography', 'All')
            location_context = create_location_context(geography)
            
            print(f"[{i}/{len(queries)}] {row['query'][:50]}... (from: {geography})")
            
            # Get responses with location context
            chatgpt_response = call_chatgpt(openai_client, row['full_query'], geography)
            gemini_response = call_gemini(gemini_client, row['full_query'], geography)
            
            # Write row
            writer.writerow([
                datetime.utcnow().isoformat(),
                row['category'], row['funnel'], row['geography'],
                row['query'], row['addition'], row['full_query'],
                location_context,
                CHATGPT_MODEL, chatgpt_response,
                GEMINI_MODEL, gemini_response,
            ])
            
            print(f"  ✅ {geography}: ChatGPT & Gemini completed")

def run_funnel_tests():
    """Process all three funnels into separate files."""
    funnels = load_queries_enhanced(INPUT_CSV)
    
    # Process each funnel
    process_funnel("High", funnels['high'], "high_funnel_responses.csv")
    process_funnel("Mid", funnels['mid'], "mid_funnel_responses.csv") 
    process_funnel("Low", funnels['low'], "low_funnel_responses.csv")
    
    print("\n🎉 All done! Check these files:")
    print("- high_funnel_responses.csv")
    print("- mid_funnel_responses.csv") 
    print("- low_funnel_responses.csv")

if __name__ == "__main__":
    print("=== LOCATION-AWARE MULTI-FUNNEL TESTS ===")
    run_funnel_tests()
