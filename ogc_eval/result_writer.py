import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Handle logger import safely so script can run standalone or as module
try:
    from .logger import get_module_logger
    logger = get_module_logger("result_writer")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("result_writer")

class ResultWriter:
    """
    Handles writing evaluation results to CSV and generating summary metadata statistics.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write(self, results: List[Dict[str, Any]], base_filename: str = "eval_results"):
        """
        Writes the results to a CSV file and a summary text file.
        """
        if not results:
            logger.warning("No results to write.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_base = f"{base_filename}_{timestamp}"
        
        csv_path = os.path.join(self.output_dir, f"{final_base}.csv")
        summary_path = os.path.join(self.output_dir, f"{final_base}_summary.txt")

        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Write CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        # Generate and Write Summary
        self.write_summary_only(df, summary_path)

    def write_summary_only(self, df: pd.DataFrame, output_path: str):
        """
        Helper to write just the summary text file (UTF-8 enforced).
        """
        summary_text = self._generate_summary(df)
        
        # FIX: Force UTF-8 encoding to prevent Windows 'charmap' crash
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
            
        logger.info(f"Summary statistics saved to: {output_path}")

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """
        Calculates summary statistics from the results DataFrame.
        """
        total = len(df)
        if total == 0:
            return "No results found."

        lines = []
        lines.append(f"OGC Evals - Evaluation Summary")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Samples: {total}")
        lines.append("-" * 40)

        # Abstention Stats
        if 'is_abstained' in df.columns:
            abstentions = df[df['is_abstained'] == True]
            abstention_count = len(abstentions)
            abstention_rate = (abstention_count / total) * 100
            
            lines.append(f"Abstentions: {abstention_count} ({abstention_rate:.2f}%)")
            
            if 'abstention_type' in df.columns and not abstentions.empty:
                lines.append("\nAbstention Breakdown:")
                breakdown = abstentions['abstention_type'].value_counts()
                for label, count in breakdown.items():
                    lines.append(f"  - {label}: {count}")
        else:
             lines.append("Abstention data not available.")

        lines.append("-" * 40)

        # --- Verbosity Stats ---
        if 'afg_k_gen' in df.columns and 'afg_k_gt' in df.columns:
            # Filter for non-abstained rows to see verbosity of actual answers
            if 'is_abstained' in df.columns:
                valid_df = df[df['is_abstained'] == False]
            else:
                valid_df = df

            if not valid_df.empty:
                # Calculate Verbosity: avg(k_gen - k_gt)
                # where k_gen is \hat{K} and k_gt is K
                verbosity_diff = valid_df['afg_k_gen'] - valid_df['afg_k_gt']
                avg_verbosity = verbosity_diff.mean()
                
                lines.append(f"\nVerbosity Statistics (on {len(valid_df)} answered queries):")
                lines.append(f"  Average Verbosity (Δ K): {avg_verbosity:+.4f}")
                lines.append(f"  Avg Gen Facts (K-hat):  {valid_df['afg_k_gen'].mean():.2f}")
                lines.append(f"  Avg GT Facts (K):       {valid_df['afg_k_gt'].mean():.2f}")
            else:
                lines.append("\nVerbosity Statistics: No valid answered queries found.")
        else:
            lines.append("\nVerbosity Statistics: Fact count data (afg_k_gen/gt) not available.")
        
        lines.append("-" * 40)

        # Accuracy Stats (Score)
        if 'score' in df.columns:
            if 'is_abstained' in df.columns:
                valid_scores = df[df['is_abstained'] == False]['score']
                lines.append(f"\nAccuracy Statistics (on {len(valid_scores)} answered queries):")
            else:
                valid_scores = df['score']
                lines.append(f"\nAccuracy Statistics (on all queries):")

            if not valid_scores.empty:
                lines.append(f"  Mean:   {valid_scores.mean():.4f}")
                lines.append(f"  Median: {valid_scores.median():.4f}")
                lines.append(f"  Min:    {valid_scores.min():.4f}")
                lines.append(f"  Max:    {valid_scores.max():.4f}")
                lines.append(f"  Std Dev:{valid_scores.std():.4f}")
                
                # IQR
                q1 = valid_scores.quantile(0.25)
                q3 = valid_scores.quantile(0.75)
                iqr = q3 - q1
                lines.append(f"  IQR:    {iqr:.4f} (Q1={q1:.4f}, Q3={q3:.4f})")
            else:
                lines.append("  No valid scores to analyze.")
        
        return "\n".join(lines)

if __name__ == "__main__":
    # CLI Mode: Restore summary from an existing CSV
    parser = argparse.ArgumentParser(description="Generate Summary Report from Result CSV")
    parser.add_argument("input_csv", help="Path to the existing results CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: File {args.input_csv} not found.")
        exit(1)

    print(f"Reading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
        
        # Determine output path (same name as csv but .txt)
        output_path = os.path.splitext(args.input_csv)[0] + "_summary.txt"
        
        writer = ResultWriter()
        writer.write_summary_only(df, output_path)
        print("Done.")
        
    except Exception as e:
        print(f"Failed to process CSV: {e}")