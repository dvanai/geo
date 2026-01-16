import os
import math
import json
import re
import pandas as pd
from openai import OpenAI
import glob

# ========= CONFIG ========= #

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_api_key_as_variable_goes_here")
JUDGE_MODEL = "gpt-4o"

# Input: ALL 4 files from cleaning script
INPUT_FILES = [
    "llm_table_view_combined_per_source.csv",
    "llm_table_view_high_per_source.csv", 
    "llm_table_view_mid_per_source.csv",
    "llm_table_view_low_per_source.csv"
]

# Output: matching Excel files
OUTPUT_XLSX = {
    "combined": "geo_eval_results_combined.xlsx",
    "high": "geo_eval_results_high.xlsx",
    "mid": "geo_eval_results_mid.xlsx",
    "low": "geo_eval_results_low.xlsx"
}

# decay factor for position weight
POSITION_LAMBDA = 0.3

client = OpenAI(api_key=OPENAI_API_KEY)

# ========= OBJECTIVE METRICS ========= #

def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())

def parse_citation_position(citation: str) -> int:
    if not isinstance(citation, str):
        return 0
    m = re.search(r"\[(\d+)\]", citation)
    return int(m.group(1)) if m else 0

def position_weight(position: int) -> float:
    if position <= 0:
        return 0.0
    return math.exp(-POSITION_LAMBDA * (position - 1))

def add_objective_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["WordCount"] = df["Answer"].apply(word_count)
    df["CitationPosition"] = df["Citation"].apply(parse_citation_position)
    df["PositionWeight"] = df["CitationPosition"].apply(position_weight)
    df["PAWordCount"] = df["WordCount"] * df["PositionWeight"]

    # normalize to [0,1]
    for col in ["WordCount", "PositionWeight", "PAWordCount"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_Norm"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0

    return df

# ========= SUBJECTIVE METRICS ========= #

SUBJECTIVE_KEYS = [
    "diversity", "follow", "influence", "relevance", 
    "subjcount", "subjpos", "uniqueness"
]

RUBRIC_TEXT = """
You are evaluating how a single citation (Source [x]) contributes to a large language model answer.

Rate Source [x] on seven metrics from 1 to 5 (integers only):
1) Diversity of Ideas (diversity)
- How broad is the range of ideas or topics that Source [x] contributes to the answer?
- 1 = not cited or adds almost no new ideas.
- 5 = cited and adds a wide range of ideas, giving a comprehensive, balanced view.

2) Follow-Up Likelihood (follow)
- How likely is a user to click the citation / visit this URL for more information?
- 1 = not cited or not engaging.
- 5 = highly engaging and very likely to prompt a click.

3) Influence of Citation (influence)
- How much does the quality, completeness, and coherence of the answer depend on Source [x]?
- 1 = not cited or answer would be almost the same without it.
- 5 = central to making the answer correct, coherent, and complete.

4) Relevance of Citation to Query (relevance)
- How directly does Source [x] address the user’s query with precise, clear, useful info?
- 1 = not cited or basically irrelevant.
- 5 = highly relevant, precise, clear, and useful.

5) Subjective Count / Remembrance (subjcount)
- How much content in the answer feels like it comes from Source [x], and how memorable is it?
- 1 = not cited or barely contributes to understanding or memory.
- 5 = contributes a lot of content and is particularly memorable.

6) Subjective Position (subjpos)
- How likely is a typical user to encounter this citation while reading the answer (perceived prominence)?
- 1 = not cited or tucked away where the user is unlikely to see it.
- 5 = highly prominent; the user is almost certain to see it.

7) Uniqueness in Response (uniqueness)
- How different is the information from Source [x] compared with other sources used in the answer?
- 1 = not cited or redundant with other sources.
- 5 = clearly unique information that stands out from other sources.



8) Sentiment (Emotional Framing)
- How warm, compelling, and emotionally positive is the tone when Vancouver is mentioned?
Score Guidance
5 –Vancouver is mentioned and tone is very warm, vivid, and inviting. Vancouver is framed as inspiring, refreshing, or energizing 
4 – Vancouver is mentioned and tone is positive; Vancouver is recommended but framed more functionally or grouped with peers
3 – Vancouver is mentioned and tone is neutral or factual; Vancouver is mentioned without emotional pull
2 – Vancouver is mentioned, but described inaccurately, dismissively, or in a way that conflicts with brand values
1 – Vancouver is not mentioned 
Examples of strong sentiment phrases
•	“Vancouver is one of the most refreshing cities in North America…”
•	“Few cities offer such a calming balance of nature and culture as Vancouver…”

9) Specificity (Place & Experience Detail)
Definition
Does the response reference real, specific Vancouver places, neighbourhoods, events, or experiences?
Score Guidance
5 - Vancouver is mentioned and there are multiple specific and accurate Vancouver references made (e.g. Stanley Park, cherry blossoms in Queen Elizabeth Park, neighbourhoods, Michelin restaurants)
4 – Vancouver is mentioned and there is at least one specific Vancouver place, experience, or neighbourhood is named
3 – Vancouver is mentioned generally, without concrete detail
2 – Vancouver is mentioned but information is inaccurate 
1 – Vancouver is not mentioned  

10) Brand Alignment (Strategic Fit)
Definition
How well does the response reflect Destination Vancouver’s brand?
Vancouver Brand / Content Pillars 
•	Effortless 
•	Embracing 
•	Energizing 
•	Fresh 
•	Immersive Outdoors 
•	Converging Cultures 
•	Fresh perspectives 
•	Wellbeing 
•	Invigoration 
•	Fresh 
•	Nature / Proximity of City to Nature
•	Culinary 
•	Wellness 
•	Major Events 
•	Unique Neighbourhoods 
•	Arts and Culture  

Score Guidance
5 - Vancouver is mentioned and one or more of the above brand pillars are clearly reflected 
4 - Vancouver is mentioned and brand themes are touched indirectly
3 - Vancouver is mentioned, but brand pillars are not evident
2 - Vancouver is mentioned but themes are misaligned with our brand
1 – Vancouver is not mentioned  

*For 8-10, it is not based on per Citation - it is more to the whole response


Other Measurement Suggestions:  
•	Across all High-Funnel questions, track how many times Vancouver is mentioned. 
•	Culinary questions – track how many times Michelin comes up when Vancouver is mentioned. 
•	Attractions – track how many times our top attractions are recommended across all questions to see whether some are appearing more than others  






Evaluation process:
1. Read the query and generated answer.
2. Focus on the parts of the answer that appear to rely on Source [x] (the citation token and its URL).
3. Consider the seven criteria above and assign each a score from 1 to 5.


Return ONLY a valid JSON object with this exact schema and no extra commentary:
{
  "diversity": 1,
  "follow": 1,
  "influence": 1,
  "relevance": 1,
  "subjcount": 1,
  "subjpos": 1,
  "uniqueness": 1
}
"""

def build_judge_prompt(row: pd.Series) -> str:
    return f"""
User query: "{row['Prompt']}"

Generated answer: "{row['Answer']}"

Source [1] is {row['Citation']} with URL: {row['URL']}

{RUBRIC_TEXT}
"""

def judge_row(row: pd.Series) -> dict:
    prompt_text = build_judge_prompt(row)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
            {"role": "user", "content": prompt_text},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {k: 1 for k in SUBJECTIVE_KEYS}
    return data

def add_subjective_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subj_cols = {k: [] for k in SUBJECTIVE_KEYS}
    subj_avg = []

    print(f"Judging {len(df)} rows with {JUDGE_MODEL}...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(df)} rows")
        
        scores = judge_row(row)
        vals = []
        for k in SUBJECTIVE_KEYS:
            v = scores.get(k, 1)
            if not isinstance(v, (int, float)):
                v = 1
            v = max(1, min(5, int(v)))
            subj_cols[k].append(v)
            vals.append(v)
        subj_avg.append(sum(vals) / len(SUBJECTIVE_KEYS))

    for k in SUBJECTIVE_KEYS:
        df[k] = subj_cols[k]
    df["SubjectiveImpressions"] = subj_avg
    return df

# ========= EXPORT TO EXCEL ========= #

def export_to_excel(df: pd.DataFrame, path: str, dataset_name: str):
    df = df.copy()
    
    # normalize and compute total metric
    mn, mx = df["SubjectiveImpressions"].min(), df["SubjectiveImpressions"].max()
    df["SubjectiveImpressions_Norm"] = (
        (df["SubjectiveImpressions"] - mn) / (mx - mn) if mx > mn else 0.0
    )
    df["TotalMetric"] = (df["PAWordCount_Norm"] + df["SubjectiveImpressions_Norm"]) / 2

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Summary sheet first
        summary = (
            df.groupby(["Funnel", "Model", "Source"])
            .agg(
                Count=("Source", "size"),
                AvgTotalMetric=("TotalMetric", "mean"),
                AvgSubjective=("SubjectiveImpressions", "mean"),
                AvgPAWordCount=("PAWordCount_Norm", "mean")
            )
            .round(3)
            .reset_index()
        )
        summary["RelevanceScore"] = summary["Count"] * summary["AvgTotalMetric"]
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # All data
        df.to_excel(writer, sheet_name="ALL", index=False)

        # Per prompt/model sheets (first 30 chars)
        for (prompt, model), g in df.groupby(["Prompt", "Model"]):
            sheet_name = f"{model}_{hash(prompt) % 10000}"[:31]
            g.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Saved {len(df)} rows to {path} ({dataset_name})")

# ========= MAIN ========= #

def process_file(input_file, output_file, name):
    if not os.path.exists(input_file):
        print(f"⚠️  Skipping {input_file} (not found)")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    df = add_objective_metrics(df)
    df = add_subjective_metrics(df)
    export_to_excel(df, output_file, name)

def main():
    print("🔍 JUDGE MODEL EVALUATION - PROCESSING 4 FILES")
    print("="*60)
    
    # Process all 4 files
    file_pairs = [
        ("llm_table_view_combined_per_source.csv", "geo_eval_results_combined.xlsx", "Combined"),
        ("llm_table_view_high_per_source.csv", "geo_eval_results_high.xlsx", "High Funnel"),
        ("llm_table_view_mid_per_source.csv", "geo_eval_results_mid.xlsx", "Mid Funnel"),
        ("llm_table_view_low_per_source.csv", "geo_eval_results_low.xlsx", "Low Funnel")
    ]
    
    for input_file, output_file, name in file_pairs:
        process_file(input_file, output_file, name)
    
    print("\n🎉 COMPLETE! Created 4 evaluation Excel files:")
    for input_file, output_file, name in file_pairs:
        if os.path.exists(output_file):
            size = len(pd.read_csv(input_file))
            print(f"  ✅ {output_file}: {size} citations judged")
        else:
            print(f"  ❌ {output_file}: (input file missing)")

if __name__ == "__main__":
    main()
