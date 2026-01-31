import argparse
import os
import json
import pandas as pd
import time
import random
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. IMPORT LOGIC ---
from ogc_eval.data_loader import DataLoader
from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier
from ogc_eval.result_writer import ResultWriter
from ogc_eval.logger import setup_logger

# --- 2. BATCH FACT VERIFIER (API Local Phase) ---
class BatchFactVerifier(FactVerifier):
    """
    Verifies ALL claims in a single API call.
    Uses the logic we refined to handle JSON response format.
    """
    def verify(self, hypothesis_claims, reference_claims):
        # Initialize at the top to prevent UnboundLocalError
        response = "" 
        supported_count = 0
        
        if not hypothesis_claims or not reference_claims:
            return 0.0, 0

        ref_text = "\n".join([f"- {c}" for c in reference_claims])
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(hypothesis_claims)])
        
        prompt = f"""Reference Facts:
{ref_text}

Claims to Verify:
{claims_text}

Return a JSON object with the key "decisions" containing a list of "YES" or "NO" for each claim.
Example: {{"decisions": ["YES", "NO"]}}"""

        try:
            # Requires LLMWrapper to handle **kwargs for response_format
            response = self.model.generate(
                prompt, 
                max_new_tokens=1000, 
                response_format={ "type": "json_object" } 
            ).strip()
            
            data = json.loads(response)
            decisions = data.get("decisions", [])
            supported_count = sum(1 for d in decisions if str(d).upper() == "YES")
            
        except Exception:
            # Fallback: Count "YES" strings if JSON fails
            if response:
                supported_count = response.upper().count('"YES"')
        
        k, k_hat = len(reference_claims), len(hypothesis_claims)
        precision = min(1.0, supported_count / k_hat) if k_hat > 0 else 0.0
        recall = min(1.0, supported_count / k) if k > 0 else 0.0
        
        score = 0.0
        if (precision + recall) > 0:
            score = 2 * (precision * recall) / (precision + recall)
        
        return score, int(supported_count)

# --- 3. PHASE 1: ABSTENTION (SAGEMAKER/GPU) ---
def run_abstention_phase(args):
    print(f"🧠 Running Abstention Detection on GPU...")
    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    detector = AbstentionDetector(device=args.device)
    
    for f in files:
        df = pd.read_csv(f)
        responses = df['generated_response'].fillna("").astype(str).tolist()
        raw_results = detector.classifier(responses, batch_size=16, truncation=True)
        
        # Apply the 0.925 confidence threshold logic
        is_abstained = []
        for res in raw_results:
            label_id = int(res['label'].split('_')[-1])
            is_abstained.append(not (label_id in [3, 5] or res['score'] < 0.925))
            
        df['is_abstained'] = is_abstained
        out_path = f.replace(".csv", "_abstentions.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved abstention results to {out_path}")

# --- 4. PHASE 2: VERIFICATION (LOCAL/API) ---
def worker_verify(index, row, afg, verifier):
    """Processes a single row using the standard (non-batch) AFG logic."""
    try:
        gt_raw = row.get('response_facts', "[]")
        gt_facts = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        
        if row['is_abstained']:
            return index, {
                "prompt": row['prompt'], "domain": row.get("serviceDomain", ""),
                "is_abstained": True, "score": 0.0, "supported_claims": 0,
                "afg_k_gen": 0, "afg_k_gt": len(gt_facts)
            }

        # Original sentence-by-sentence AFG call [Preserved]
        gen_facts, k_gen = afg.run(row['generated_response'])
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
    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    llm = LLMWrapper(model_name=args.model, api_key=args.api_key)
    afg = AtomicFactGenerator(llm) # Uses your original afg.py logic
    verifier = BatchFactVerifier(llm)
    writer = ResultWriter()

    for f in files:
        base_name = os.path.basename(f).replace(".csv", "")
        # Skip if already in output folder (The Resume Feature)
        if os.path.exists(os.path.join("eval_outputs", f"eval_results_{base_name}.csv")):
            continue

        print(f"🚀 Verifying {base_name} locally...")
        df = pd.read_csv(f)
        results = [None] * len(df)
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(worker_verify, i, row, afg, verifier): i for i, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df), desc="Progress"):
                idx, res = future.result()
                results[idx] = res
        
        writer.write([r for r in results if r], base_filename=f"eval_results_{base_name}")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Unified CLI setup
    for cmd in ["abstain", "verify"]:
        p = subparsers.add_parser(cmd)
        p.add_argument("--input", default=None)
        p.add_argument("--input_dir", default=None)
        if cmd == "abstain": p.add_argument("--device", default="cuda")
        else:
            p.add_argument("--model", required=True)
            p.add_argument("--api_key", required=True)

    args = parser.parse_args()
    if args.command == "abstain": run_abstention_phase(args)
    elif args.command == "verify": run_verification_phase(args)