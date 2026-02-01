import argparse
import os
import json
import pandas as pd
import time
import logging
import random
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from litellm import completion as original_completion

# --- 0. SILENCE & ROBUSTNESS ---
litellm.suppress_debug_info = True
litellm.drop_params = True
# Nuke all logging
for logger_name in ["litellm", "httpx", "httpcore"]:
    l = logging.getLogger(logger_name)
    l.setLevel(logging.CRITICAL)
    l.propagate = False

def robust_completion(*args, **kwargs):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 120.0 

    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["rate_limit", "429", "overloaded", "503", "timeout", "timed_out"]):
                attempt += 1
                wait_time = (5 * attempt) + random.uniform(1, 5)
                time.sleep(wait_time)
                continue
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

# --- 1. IMPORT LOGIC ---
from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier
from ogc_eval.result_writer import ResultWriter
from ogc_eval.logger import setup_logger
import ogc_eval.model
ogc_eval.model.completion = robust_completion

# --- 2. NEW: BATCH ATOMIC FACT GENERATOR (Speed Fix) ---
class BatchAtomicFactGenerator(AtomicFactGenerator):
    """
    Extracts facts from the full response in 1 API call (Batch),
    BUT respects the AFG infrastructure by including Few-Shot examples
    from the loaded 'demons.json'.
    """
    def run(self, text):
        if not text or not isinstance(text, str):
            return [], 0
            
        # 1. Build Few-Shot Context from your Infrastructure (self.demons)
        # We pick 3 examples to teach the 'Atomic' style.
        # (We use the first 3 or random 3 because BM25 is less effective 
        # when matching a whole paragraph to single sentences).
        few_shot_examples = ""
        if hasattr(self, 'demons') and self.demons:
            # Grab first 3 examples (or generic ones if you prefer)
            keys = list(self.demons.keys())[:3]
            for sentence in keys:
                facts = self.demons[sentence]
                facts_str = "\n".join([f"- {f}" for f in facts])
                few_shot_examples += f"Text: \"{sentence}\"\nAtomic Facts:\n{facts_str}\n\n"
        
        # 2. Construct the Prompt with Examples
        prompt = f"""You are an expert Atomic Fact Generator. 
Decompose the input text into a list of self-contained, atomic facts.
Follow the style of the examples below.

--- EXAMPLES ---
{few_shot_examples}
--- END EXAMPLES ---

Text:
"{text}"

Atomic Facts:"""
        
        try:
            # Single API Call
            response = self.model.generate(prompt, max_new_tokens=2000)
            
            # Parse output
            facts = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    facts.append(line[2:].strip())
            
            return facts, len(facts)
        except Exception as e:
            return [], 0

# --- 3. BATCH FACT VERIFIER (Already existed, keeping it) ---
class BatchFactVerifier(FactVerifier):
    def verify(self, hypothesis_claims, reference_claims):
        if not hypothesis_claims or not reference_claims:
            return 0.0, 0

        ref_text = "\n".join([f"- {c}" for c in reference_claims])
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(hypothesis_claims)])
        
        prompt = f"""Reference Facts:
{ref_text}

Claims to Verify:
{claims_text}

For EACH claim, determine if it is supported by the Reference Facts.
Return a JSON object: {{"decisions": ["YES", "NO", ...]}}"""

        try:
            response = self.model.generate(
                prompt, max_new_tokens=1000, response_format={ "type": "json_object" }
            ).strip()
            data = json.loads(response)
            decisions = data.get("decisions", [])
            supported_count = sum(1 for d in decisions if str(d).upper() == "YES")
        except:
            supported_count = 0
        
        k, k_hat = len(reference_claims), len(hypothesis_claims)
        precision = min(1.0, supported_count / k_hat) if k_hat > 0 else 0.0
        recall = min(1.0, supported_count / k) if k > 0 else 0.0
        score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return score, int(supported_count)

# --- 4. PHASE 1: ABSTENTION (GPU) ---
def run_abstention_phase(args):
    print(f"🧠 Running Abstention Detection...")
    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    detector = AbstentionDetector(device=args.device)
    
    for f in files:
        df = pd.read_csv(f)
        responses = df['generated_response'].fillna("").astype(str).tolist()
        # Batch Inference
        raw_results = detector.classifier(responses, batch_size=16, truncation=True)
        
        is_abstained = []
        for res in raw_results:
            label_id = int(res['label'].split('_')[-1])
            is_abstained.append(not (label_id in [3, 5] or res['score'] < 0.925))
            
        df['is_abstained'] = is_abstained
        out_path = f.replace(".csv", "_abstentions.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")

# --- 5. PHASE 2: VERIFICATION (API) ---
def worker_verify(index, row, afg, verifier):
    try:
        # Load GT Facts (Fixed Logic)
        gt_raw = row.get('response_facts', "[]")
        if pd.isna(gt_raw): gt_raw = "[]"
        gt_facts = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        
        # Skip abstained
        if row.get('is_abstained', False):
            return index, {
                "prompt": row['prompt'], "domain": row.get("serviceDomain", ""),
                "is_abstained": True, "score": 0.0, "supported_claims": 0,
                "afg_k_gen": 0, "afg_k_gt": len(gt_facts)
            }

        # 1. Batch AFG (1 Call)
        gen_facts, k_gen = afg.run(row['generated_response'])
        
        # 2. Batch Verify (1 Call)
        score, supported = verifier.verify(gen_facts, gt_facts)
        
        return index, {
            "prompt": row['prompt'], "domain": row.get("serviceDomain", ""),
            "is_abstained": False, "score": score, "supported_claims": supported,
            "afg_k_gen": k_gen, "afg_k_gt": len(gt_facts),
            "generated_response": row['generated_response']
        }
    except Exception as e:
        return index, {"prompt": row.get('prompt', ''), "error": str(e), "score": 0.0}

def run_verification_phase(args):
    # Load Reference DF ONCE
    print(f"📂 Loading Reference Data from {args.reference}...")
    ref_df = pd.read_csv(args.reference)
    # Create lookup dictionary for O(1) merging
    ref_map = dict(zip(ref_df['prompt'].astype(str), ref_df['response_facts']))

    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    
    llm = LLMWrapper(model_name=args.model, api_key=args.api_key)
    # INJECT BATCH AFG HERE
    afg = BatchAtomicFactGenerator(llm)
    verifier = BatchFactVerifier(llm)
    writer = ResultWriter()

    for f in files:
        base_name = os.path.basename(f).replace(".csv", "")
        # Look for the _abstentions file if it exists, otherwise use raw input
        if "_abstentions" not in f and os.path.exists(f.replace(".csv", "_abstentions.csv")):
            f = f.replace(".csv", "_abstentions.csv")
            print(f"   -> Found abstention file: {f}")

        print(f"🚀 Verifying {base_name}...")
        df = pd.read_csv(f)
        
        # MERGE FIX: Map response_facts onto the dataframe
        df['prompt'] = df['prompt'].astype(str)
        df['response_facts'] = df['prompt'].map(ref_map)
        
        # Check if merge worked
        missing_facts = df['response_facts'].isna().sum()
        if missing_facts > 0:
            print(f"⚠️  WARNING: {missing_facts} rows missing Ground Truth facts. Scores will be 0.0 for them.")

        results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(worker_verify, i, row, afg, verifier): i for i, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df), desc="Progress"):
                idx, res = future.result()
                results[idx] = res
        
        valid_results = [r for r in results if r is not None]
        # Calculate summary manually to avoid ResultWriter crash on empty/weird data
        scores = [r['score'] for r in valid_results if isinstance(r.get('score'), (int, float))]
        if scores:
            print(f"   🏆 Average Score: {sum(scores)/len(scores):.2%}")
            
        writer.write(valid_results, base_filename=f"eval_results_{base_name}")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    for cmd in ["abstain", "verify"]:
        p = subparsers.add_parser(cmd)
        p.add_argument("--input", default=None)
        p.add_argument("--input_dir", default=None)
        if cmd == "abstain": 
            p.add_argument("--device", default="cuda")
        else:
            p.add_argument("--model", required=True)
            p.add_argument("--api_key", required=True)
            # ADDED REFERENCE ARGUMENT
            p.add_argument("--reference", required=True, help="Path to reference CSV")

    args = parser.parse_args()
    if args.command == "abstain": run_abstention_phase(args)
    elif args.command == "verify": run_verification_phase(args)
    print('start')