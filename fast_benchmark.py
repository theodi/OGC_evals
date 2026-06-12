# fast_benchmark.py
import pandas as pd
import os
import json
import time
import logging
import sys
import random
from litellm import completion, RateLimitError
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import auto-loading of keys from notes.txt
import parse_keys

# ==========================================
# 0. SETUP & SILENCE
# ==========================================
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("litellm").setLevel(logging.CRITICAL)

# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_CSV = "public_set_1.csv"
OUTPUT_DIR = "fast_results"

# Concurrency tuning
MAX_WORKERS = 20 

# Paths
PROMPT_DIR = os.path.join("ogc_eval", "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPT_DIR, "test_system.txt")
FEW_SHOT_FILE = os.path.join(PROMPT_DIR, "few_shot_test_examples.json")

# Define target evaluation configurations
TASKS = [
    # GROQ (Kimi K2)
    ("groq/moonshotai/kimi-k2-instruct-0905", "kimi_k2_0shot.csv", "zero"), 
    ("groq/moonshotai/kimi-k2-instruct-0905", "kimi_k2_fewshot.csv", "few"),
]

# ==========================================
# 2. PROMPT LOADER LOGIC
# ==========================================

def load_system_prompt():
    if not os.path.exists(SYSTEM_PROMPT_FILE): 
        return "You are an AI assistant that provides advice and support to UK citizens."
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f: 
        return f.read().strip()

def load_few_shot_examples():
    if not os.path.exists(FEW_SHOT_FILE): 
        return []
    with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f: 
        return json.load(f)

SYSTEM_PROMPT = load_system_prompt()
FEW_SHOT_EXAMPLES = load_few_shot_examples()


def format_user_msg(row):
    return (
        'prompt: ' + str(row['prompt']) + '\n' + 
        'targetAgeGroup: ' + str(row['targetAgeGroup']) + '\n' + 
        'genderIdentity: ' + str(row['genderIdentity']) + '\n' + 
        'educationBackground: ' + str(row['educationBackground']) + '\n' + 
        'targetProfession: ' + str(row['targetProfession']) + '\n' + 
        'digitalLiteracy: ' + str(row['digitalLiteracy']) + '\n' + 
        'geoRegion: ' + str(row['geoRegion']) + '\n' + 
        'householdIncomeStatus: ' + str(row['householdIncomeStatus']) + '\n' 
    )


def construct_messages(row, context_mode="zero"):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_mode == 'few':
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": format_user_msg(example)})
            messages.append({"role": "assistant", "content": 'response: ' + str(example['response']) + '\n'})
    messages.append({"role": "user", "content": format_user_msg(row)})
    return messages


# ==========================================
# 3. ROBUST EXECUTION ENGINE (API Rate Limit & 503 Overload handling)
# ==========================================

def process_row_with_retry(index, row, model_name, context_mode):
    messages = construct_messages(row, context_mode)
    
    max_retries = 10
    attempt = 0
    
    while attempt < max_retries:
        try:
            # Check model prefix to apply correct API key from environment
            api_key = None
            if "groq" in model_name.lower():
                api_key = os.environ.get("GROQ_API_KEY")
            elif "gemini" in model_name.lower():
                api_key = os.environ.get("GEMINI_API_KEY")
            elif "claude" in model_name.lower():
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            response = completion(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2000, 
                api_key=api_key
            )
            return index, response.choices[0].message.content
        
        except RateLimitError:
            attempt += 1
            wait_time = 20 + (attempt * 5) + random.uniform(1, 5)
            tqdm.write(f"⚠️  Rate Limit on {model_name} (Row {index}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            continue
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Catch transient network/server overloads (HTTP 503, overload, timeouts)
            if any(x in error_str for x in ["503", "service unavailable", "overloaded", "timeout", "timed_out"]):
                attempt += 1
                wait_time = 30 + (attempt * 5)
                tqdm.write(f"⚠️  503/Server Overload on {model_name} (Row {index}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            
            # Raise unrecoverable failures instantly to prevent loop hanging
            return index, f"ERROR: {str(e)}"
    
    return index, "ERROR: Max Retries Exceeded"


def run_task(model_name, output_file, context_mode):
    print(f"\n🚀 Starting {model_name} [{context_mode}]...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    results = [""] * len(df)
    
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(process_row_with_retry, i, row, model_name, context_mode): i 
                for i, row in df.iterrows()
            }
            futures = list(future_to_idx.keys())
            
            for future in tqdm(as_completed(future_to_idx), total=len(df), unit="rows"):
                idx, result = future.result()
                results[idx] = result

    except KeyboardInterrupt:
        print("\n🛑 Stopping execution... Saving partial results.")
        for f in futures: 
            f.cancel()
        
        df['generated_response'] = results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        partial_path = os.path.join(OUTPUT_DIR, output_file.replace(".csv", "_PARTIAL.csv"))
        df.to_csv(partial_path, index=False)
        print(f"💾 Saved partial results to {partial_path}")
        sys.exit(0)

    # Save complete benchmark responses
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df['generated_response'] = results
    save_path = os.path.join(OUTPUT_DIR, output_file)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved complete results to {save_path}")


if __name__ == "__main__":
    start_global = time.time()
    for model, filename, mode in TASKS:
        run_task(model, filename, mode)
    print(f"\n🏁 Finished benchmark suite in {(time.time()-start_global)/60:.2f} minutes.")
