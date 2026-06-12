# fast_main.py
import argparse
import os
import json
import pandas as pd
import time
import logging
import random
import glob
import re
import ast
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from litellm import completion as original_completion

# Import auto-loading of keys from notes.txt
import parse_keys

# --- 0. SILENCE & ROBUSTNESS ---
litellm.suppress_debug_info = True
litellm.drop_params = True

# Nuke verbose third-party loggers
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

# --- 1. IMPORT CORE LOGIC ---
from ogc_eval.model import LLMWrapper
from ogc_eval.afg import AtomicFactGenerator
from ogc_eval.abstention import AbstentionDetector
from ogc_eval.afv import FactVerifier
from ogc_eval.result_writer import ResultWriter
from ogc_eval.logger import setup_logger, get_module_logger
import ogc_eval.model

ogc_eval.model.completion = robust_completion
logger = get_module_logger("fast_main")


# Helper utility to robustly deserialize pandas representation of fact lists
def safe_load_facts(gt_raw):
    if not gt_raw or pd.isna(gt_raw):
        return []
    if isinstance(gt_raw, list):
        return gt_raw
    if isinstance(gt_raw, str):
        gt_raw = gt_raw.strip()
        if not gt_raw or gt_raw.lower() == "nan":
            return []
        try:
            return json.loads(gt_raw)
        except json.JSONDecodeError:
            try:
                # Fallback to ast.literal_eval for single-quoted Python representations
                res = ast.literal_eval(gt_raw)
                if isinstance(res, list):
                    return res
            except Exception:
                pass
    return []


# --- 2. PARALLEL SENTENCE-LEVEL ATOMIC FACT GENERATOR (Unrestricted Verbosity Fix) ---
class ParallelSentenceAtomicFactGenerator(AtomicFactGenerator):
    """
    Parses generated responses sentence-by-sentence in parallel to ensure
    there is absolutely NO summarization bias or claim-count restriction,
    perfectly preserving the unrestricted verbosity metrics of the paper
    while delivering complete thread speedups.
    """
    def __init__(self, model, max_workers=5):
        super().__init__(model)
        self.max_workers = max_workers

    def run(self, text):
        if not text or not isinstance(text, str):
            return [], 0
            
        # 1. Split text into sentences using standard sent_tokenize
        sentences = self._split_sentences(text)
        all_atoms = []
        
        # 2. Run sentence-level generation in parallel using thread executor
        # We iterate in order to preserve original sentence/fact continuity!
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._get_atoms_for_sentence, s) for s in sentences]
            for f in futures:
                try:
                    atoms = f.result()
                    all_atoms.extend(atoms)
                except Exception as e:
                    logger.error(f"Failed to get atomic claims for sentence: {e}")
                    
        return all_atoms, len(all_atoms)


# --- 3. PARALLEL CLAIM-LEVEL FACT VERIFIER (High-Fidelity Entailment Alignment) ---
class ParallelFactVerifier(FactVerifier):
    """
    Verifies a batch of generated claims against reference ground truth facts.
    To ensure absolute logical consistency and alignment with our paper's aims:
    1. Loads standard system/user templates directly from prompts directory.
    2. Runs logical entailment checks on each claim concurrently in parallel.
    3. Eliminates LLM count/index formatting discrepancies completely.
    """
    def __init__(self, model, max_workers=5):
        super().__init__(model)
        self.max_workers = max_workers

    def verify(self, hypothesis_claims, reference_claims):
        if not hypothesis_claims or not reference_claims:
            return 0.0, 0

        ref_text = "\n".join([f"- {c}" for c in reference_claims])
        supported_count = 0
        
        # Concurrently evaluate logical entailment for each claim
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for claim in hypothesis_claims:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_template.format(reference_text=ref_text, claim=claim)}
                ]
                futures.append(executor.submit(self.model.classify, messages, options=["YES", "NO"]))
            
            for f in futures:
                try:
                    output = f.result()
                    if "YES" in output.upper():
                        supported_count += 1
                except Exception as e:
                    logger.error(f"Parallel verification of claim failed: {e}")

        # Compute metrics matching paper equations (Section 5.4)
        k, k_hat = len(reference_claims), len(hypothesis_claims)
        
        precision = min(1.0, supported_count / k_hat) if k_hat > 0 else 0.0
        recall = min(1.0, supported_count / k) if k > 0 else 0.0
        score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return score, int(supported_count)


# --- 4. PHASE 1: ABSTENTION (GPU Sequence Classification) ---
def run_abstention_phase(args):
    print(f"🧠 Running Abstention Detection...")
    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    detector = AbstentionDetector(device=args.device)
    
    for f in files:
        df = pd.read_csv(f)
        
        # Handlebuilt-in dataset slicing CLI argument
        if args.limit is not None:
            df = df.head(args.limit)
            print(f"✂️  Slicing input dataset to the first {args.limit} rows (due to --limit).")

        responses = df['generated_response'].fillna("").astype(str).tolist()
        
        # Chunked Inference with tqdm progress tracking
        raw_results = []
        batch_size = 16
        for i in tqdm(range(0, len(responses), batch_size), desc="Abstention Progress"):
            chunk = responses[i : i + batch_size]
            results = detector.classifier(chunk, truncation=True)
            raw_results.extend(results)
        
        is_abstained = []
        for res in raw_results:
            label_id = int(res['label'].split('_')[-1])
            # Rubric: Abstention triggers if label is NOT 3 or 5, AND classifier score >= 0.925
            is_abstained.append(not (label_id in [3, 5] or res['score'] < 0.925))
            
        df['is_abstained'] = is_abstained
        out_path = f.replace(".csv", "_abstentions.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved tagged abstentions to: {out_path}")


# --- 5. PHASE 2: VERIFICATION (Parallel API Orchestration) ---
def worker_verify(index, row, afg, verifier):
    try:
        # Robust ground truth fact deserialization
        gt_raw = row.get('response_facts', "[]")
        gt_facts = safe_load_facts(gt_raw)
        
        prompt_val = row.get('prompt', '')
        domain_val = row.get("serviceDomain", "")
        gen_resp = row.get('generated_response', '')
        if pd.isna(gen_resp): gen_resp = ''

        # Bypass pipeline evaluation for abstained responses
        if row.get('is_abstained', False):
            return index, {
                "prompt": prompt_val,
                "generated_response": gen_resp,
                "is_abstained": True,
                "score": 0.0,
                "supported_claims": 0,
                "afg_k_gen": 0,
                "afg_k_gt": len(gt_facts),
                "domain": domain_val,
                "error": ""
            }

        # 1. Parallel Claim Atomisation (Sentence-level)
        gen_facts, k_gen = afg.run(gen_resp)
        
        # 2. Parallel Claim Entailment Verification
        score, supported = verifier.verify(gen_facts, gt_facts)
        
        return index, {
            "prompt": prompt_val,
            "generated_response": gen_resp,
            "is_abstained": False,
            "score": score,
            "supported_claims": supported,
            "afg_k_gen": k_gen,
            "afg_k_gt": len(gt_facts),
            "domain": domain_val,
            "error": ""
        }
    except Exception as e:
        # Maintain complete dictionary schema to prevent NaN CSV pollution
        logger.error(f"Worker verification failed at row index {index}: {e}")
        return index, {
            "prompt": row.get('prompt', ''),
            "generated_response": row.get('generated_response', ''),
            "is_abstained": False,
            "score": 0.0,
            "supported_claims": 0,
            "afg_k_gen": 0,
            "afg_k_gt": 0,
            "domain": row.get("serviceDomain", ""),
            "error": str(e)
        }


def run_verification_phase(args):
    # Load Reference dataset once for fast mapping
    print(f"📂 Loading Ground Truth Reference Data from {args.reference}...")
    ref_df = pd.read_csv(args.reference)
    
    # Clean Reference keys to prevent trailing whitespace carriage return mismatches
    ref_df['prompt_clean'] = ref_df['prompt'].astype(str).str.strip()
    ref_map = dict(zip(ref_df['prompt_clean'], ref_df['response_facts']))

    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    
    # Auto-resolve API key from environment if 'env' is specified or missing
    api_key = args.api_key
    if api_key == "env" or not api_key:
        if "groq" in args.model.lower():
            api_key = os.environ.get("GROQ_API_KEY")
        elif "gemini" in args.model.lower():
            api_key = os.environ.get("GEMINI_API_KEY")
        elif "claude" in args.model.lower():
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    llm = LLMWrapper(model_name=args.model, api_key=api_key)
    # Instantiate Parallel generators and verifiers to enforce scientific identity
    afg = ParallelSentenceAtomicFactGenerator(llm, max_workers=5)
    verifier = ParallelFactVerifier(llm, max_workers=5)
    writer = ResultWriter()

    for f in files:
        base_name = os.path.basename(f).replace(".csv", "")
        # Resolve abstention file path
        if "_abstentions" not in f and os.path.exists(f.replace(".csv", "_abstentions.csv")):
            f = f.replace(".csv", "_abstentions.csv")
            print(f"   -> Located tagged abstention file: {f}")

        print(f"🚀 Verifying model responses in: {f}...")
        df = pd.read_csv(f)
        
        # Handle built-in dataset slicing CLI argument
        if args.limit is not None:
            df = df.head(args.limit)
            print(f"✂️  Slicing input dataset to the first {args.limit} rows (due to --limit).")
        
        # Dynamic On-the-Fly Fact Stitching (In-Memory)
        df['prompt_clean'] = df['prompt'].astype(str).str.strip()
        df['response_facts'] = df['prompt_clean'].map(ref_map)
        df = df.drop(columns=['prompt_clean'])
        
        stitched_count = df['response_facts'].notna().sum()
        failed_count = df['response_facts'].isna().sum()
        print(f"📊 Stitch Status: {stitched_count} rows matched reference facts. {failed_count} rows failed to match.")
        if failed_count > 0:
            print("⚠️  WARNING: Mismatched rows will evaluate to 0.0 factuality scores!")

        results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(worker_verify, i, row, afg, verifier): i for i, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df), desc="Progress"):
                idx, res = future.result()
                results[idx] = res
        
        valid_results = [r for r in results if r is not None]
        
        # Output verification statistics
        scores = [r['score'] for r in valid_results if not r.get('is_abstained', False) and 'error' not in r.get('error', '')]
        if scores:
            print(f"   🏆 Average Factuality (F1@K Score): {sum(scores)/len(scores):.2%}")
            
        writer.write(valid_results, base_filename=f"eval_results_{base_name}")


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="High-Performance Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    for cmd in ["abstain", "verify"]:
        p = subparsers.add_parser(cmd)
        p.add_argument("--input", default=None)
        p.add_argument("--input_dir", default=None)
        # Built-in slicing limit option for all commands
        p.add_argument("--limit", type=int, default=None, help="Limit processing to the first N rows")
        if cmd == "abstain": 
            p.add_argument("--device", default="cuda")
        else:
            p.add_argument("--model", required=True)
            p.add_argument("--api_key", required=True)
            p.add_argument("--reference", required=True, help="Path to reference CSV")

    args = parser.parse_args()
    if args.command == "abstain": 
        run_abstention_phase(args)
    elif args.command == "verify": 
        run_verification_phase(args)
