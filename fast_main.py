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


# Helper utility to strip markdown and isolate JSON blocks
def clean_json_string(s):
    s = s.strip()
    if s.startswith("```"):
        # Strip leading ```json or ```
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


# Helper utility to robustly deserialize pandas representation of fact lists
def safe_load_facts(gt_raw):
    if not gt_raw:
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


# --- 2. BATCH ATOMIC FACT GENERATOR (Speed Fix & Scientific Conformity) ---
class BatchAtomicFactGenerator(AtomicFactGenerator):
    """
    Decomposes the generated response into atomic facts in a single API call,
    while fully complying with the scientific methodology:
    1. Loads instructions from ogc_eval prompts directory.
    2. Utilizes semantic BM25 retrieval to select the top 3 similar few-shot "demons".
    """
    def run(self, text):
        if not text or not isinstance(text, str):
            return [], 0
            
        # Retrieve similar few-shot demonstrations based on BM25 embedding query mapping
        few_shot_examples = ""
        if self.bm25 and self.demons:
            tokenized_query = text.split(' ')
            top_matches = self.bm25.get_top_n(tokenized_query, list(self.demons.keys()), n=3)
            for match in top_matches:
                facts = self.demons[match]
                facts_str = "\n".join([f"- {f}" for f in facts])
                few_shot_examples += f"Text: \"{match}\"\nAtomic Facts:\n{facts_str}\n\n"
        else:
            # Fallback to static selection of first 3 if BM25 is not populated
            logger.warning("BM25 index not available. Falling back to static demon selection.")
            if self.demons:
                keys = list(self.demons.keys())[:3]
                for sentence in keys:
                    facts = self.demons[sentence]
                    facts_str = "\n".join([f"- {f}" for f in facts])
                    few_shot_examples += f"Text: \"{sentence}\"\nAtomic Facts:\n{facts_str}\n\n"

        # Construct prompt matching the style of afg_system instructions
        prompt = f"""{self.system_prompt}

Here are some retrieval-based examples of how to decompose texts into lists of atomic facts:

--- EXAMPLES ---
{few_shot_examples}--- END EXAMPLES ---

Now, perform your task on the following input text.
Text:
"{text}"

Atomic Facts:"""
        
        try:
            response = self.model.generate(prompt, max_new_tokens=2000)
            
            # Parse output facts
            facts = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    facts.append(line[2:].strip())
            
            return facts, len(facts)
        except Exception as e:
            logger.error(f"Batch AFG generation failed: {e}")
            return [], 0


# --- 3. BATCH FACT VERIFIER (Structured, Robust & Sequential Fallback) ---
class BatchFactVerifier(FactVerifier):
    """
    Verifies a batch of generated claims against reference ground truth facts.
    Implements:
    1. Instructions from afv_system prompts directory.
    2. Strict JSON parsing with markdown stripping.
    3. Manual regex fallback parsing for Claude or models without JSON format support.
    4. Robust sequential verification fallback on formatting errors to prevent silent 0.0 scores.
    """
    def verify(self, hypothesis_claims, reference_claims):
        if not hypothesis_claims or not reference_claims:
            return 0.0, 0

        ref_text = "\n".join([f"- {c}" for c in reference_claims])
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(hypothesis_claims)])
        
        prompt = f"""{self.system_prompt}

We have a premise consisting of reference facts:
{ref_text}

We have a list of candidate claims to verify:
{claims_text}

For EACH claim, determine if it is entailed/supported by the premise.
Return a JSON object with a single "decisions" key containing a list of "YES" or "NO" answers matching the exact order and length of the claims:
{{"decisions": ["YES", "NO", ...]}}"""

        response = ""
        decisions = []
        try:
            # Query the model for the decisions list
            response = self.model.generate(
                prompt, max_new_tokens=1000, response_format={ "type": "json_object" }
            ).strip()
            
            # Clean and parse JSON
            cleaned_response = clean_json_string(response)
            data = json.loads(cleaned_response)
            decisions = data.get("decisions", [])
            decisions = [str(d).upper().strip() for d in decisions]
        except Exception as e:
            logger.warning(f"Structured JSON verification failed: {e}. Attempting manual parsing/fallback.")
            
        # Fallback 1: Manual string parsing if JSON parsing failed
        if not decisions and response:
            try:
                # Seek lines containing yes/no and claim indices
                lines = response.split('\n')
                for line in lines:
                    match = re.search(r'\b(YES|NO)\b', line, re.IGNORECASE)
                    if match:
                        decisions.append(match.group(1).upper())
            except Exception as parse_err:
                logger.error(f"Fallback manual regex parsing failed: {parse_err}")

        # Fallback 2: Sequential verification as the absolute reliability mechanism
        # This completely guarantees we never silently fail to 0.0 on model format issues.
        if len(decisions) != len(hypothesis_claims):
            logger.warning(
                f"Decisions list count ({len(decisions)}) does not match claim count ({len(hypothesis_claims)}). "
                f"Executing sequential fallback verification for ultimate reliability..."
            )
            decisions = []
            for claim in hypothesis_claims:
                try:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt_template.format(reference_text=ref_text, claim=claim)}
                    ]
                    output = self.model.classify(messages, options=["YES", "NO"])
                    decisions.append("YES" if "YES" in output.upper() else "NO")
                except Exception as seq_err:
                    logger.error(f"Sequential fallback verification failed for claim '{claim}': {seq_err}")
                    decisions.append("NO")

        # Compile metrics based on verified decisions
        supported_count = sum(1 for d in decisions if d == "YES")
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
        responses = df['generated_response'].fillna("").astype(str).tolist()
        
        # Batch Inference using HuggingFace classifier
        raw_results = detector.classifier(responses, batch_size=16, truncation=True)
        
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

        # 1. Batch Claim Atomisation
        gen_facts, k_gen = afg.run(gen_resp)
        
        # 2. Batch Claim Entailment Verification
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
    ref_map = dict(zip(ref_df['prompt'].astype(str), ref_df['response_facts']))

    files = glob.glob(os.path.join(args.input_dir, "*.csv")) if args.input_dir else [args.input]
    
    llm = LLMWrapper(model_name=args.model, api_key=args.api_key)
    afg = BatchAtomicFactGenerator(llm)
    verifier = BatchFactVerifier(llm)
    writer = ResultWriter()

    for f in files:
        base_name = os.path.basename(f).replace(".csv", "")
        # Resolve abstention file path
        if "_abstentions" not in f and os.path.exists(f.replace(".csv", "_abstentions.csv")):
            f = f.replace(".csv", "_abstentions.csv")
            print(f"   -> Located tagged abstention file: {f}")

        print(f"🚀 Verifying model responses in: {f}...")
        df = pd.read_csv(f)
        
        # Map response facts onto the working dataframe
        df['prompt'] = df['prompt'].astype(str)
        df['response_facts'] = df['prompt'].map(ref_map)
        
        missing_facts = df['response_facts'].isna().sum()
        if missing_facts > 0:
            print(f"⚠️  WARNING: {missing_facts} rows are missing Ground Truth reference facts. Factuality scores will evaluate to 0.0.")

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
