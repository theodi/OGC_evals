import argparse
import os
import json
import pandas as pd
import time
import random
import threading
import glob
import logging
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

# --- 2. SILENT & ROBUST API CLIENT ---
import litellm
from litellm import completion as original_completion

litellm.suppress_debug_info = True
litellm.drop_params = True

def robust_completion(*args, **kwargs):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 120.0 
    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            return original_completion(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["rate limit", "429", "503", "service", "overloaded", "timeout"]):
                attempt += 1
                time.sleep(10 + (attempt * 5) + random.uniform(1, 5))
                continue
            raise e
    raise Exception("Max Retries Exceeded")

litellm.completion = robust_completion

# --- 3. BATCH FACT VERIFIER ---
class BatchFactVerifier(FactVerifier):
    """Verifies ALL claims in a single API call."""
    def verify(self, hypothesis_claims, reference_claims):
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
            # Note: response_format requires model support in LLMWrapper
            response = self.model.generate(
                prompt, max_new_tokens=1000, 
                response_format={ "type": "json_object" } 
            ).strip()
            data = json.loads(response)
            decisions = data.get("decisions", [])
            supported_count = sum(1 for d in decisions if str(d).upper() == "YES")
        except Exception as e:
            if response:
                supported_count = response.upper().count('"YES"')
        
        k, k_hat = len(reference_claims), len(hypothesis_claims)
        precision = min(1.0, supported_count / k_hat) if k_hat > 0 else 0.0
        recall = min(1.0, supported_count / k) if k > 0 else 0.0
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return score, int(supported_count)

# --- 4. PHASE 1: ABSTENTION (SAGEMAKER/GPU) ---
def run_abstention_phase(args):
    print(f"🧠 Phase 1: Abstention Detection (GPU) on {args.input}")
    detector = AbstentionDetector(device=args.device)
    df = pd.read_csv(args.input)
    
    responses = df['generated_response'].fillna("").astype(str).tolist()
    inputs = [r[:4096] for r in responses]
    
    raw_results = detector.classifier(inputs, batch_size=16, truncation=True)
    
    is_abstained = []
    for res in raw_results:
        label_id = int(res['label'].split('_')[-1])
        # Score threshold from Section 5.6 of the paper
        is_abstained.append(False if (label_id in [3, 5] or res['score'] < 0.925) else True)
        
    df['is_abstained'] = is_abstained
    out_path = args.input.replace(".csv", "_abstentions.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved results with abstentions to: {out_path}")

# --- 5. PHASE 2: VERIFICATION (LOCAL/API) ---
def worker_verify(index, row, afg, verifier):
    """Individual row processor for ThreadPool."""
    try:
        gt_facts_raw = row.get('response_facts', "[]")
        gt_facts = json.loads(gt_facts_raw) if isinstance(gt_facts_raw, str) else gt_facts_raw

        if row['is_abstained']:
            return index, {
                "prompt": row['prompt'], "domain": row.get("serviceDomain", ""), 
                "is_abstained": True, "score": 0.0, "supported_claims": 0,
                "afg_k_gen": 0, "afg_k_gt": len(gt_facts)
            }

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
    print(f"🚀 Phase 2: API Verification (Local) using {args.model}")
    df = pd.read_csv(args.input)
    
    # Initialize API-based components
    llm = LLMWrapper(model_name=args.model, api_key=args.api_key)
    afg = AtomicFactGenerator(llm)
    verifier = BatchFactVerifier(llm)
    
    results = [None] * len(df)
    # Using thread pool to handle IO-bound API calls
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker_verify, i, row, afg, verifier): i for i, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(df), desc="Verifying"):
            idx, res = future.result()
            results[idx] = res

    # Write final output
    writer = ResultWriter()
    writer.write([r for r in results if r], base_filename=f"final_eval_{args.model}")
    print("✅ Verification complete. Check 'eval_outputs' folder.")

# --- 6. CLI CONTROLLER ---
if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="OGC-Eval Split Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Abstain Command
    p_abs = subparsers.add_parser("abstain")
    p_abs.add_argument("--input", required=True, help="Raw CSV with generated_responses")
    p_abs.add_argument("--device", default="cuda")

    # Verify Command
    p_ver = subparsers.add_parser("verify")
    p_ver.add_argument("--input", required=True, help="CSV from abstain phase")
    p_ver.add_argument("--model", required=True)
    p_ver.add_argument("--api_key", required=True)

    args = parser.parse_args()
    if args.command == "abstain":
        run_abstention_phase(args)
    elif args.command == "verify":
        run_verification_phase(args)