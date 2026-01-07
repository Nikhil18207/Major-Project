"""
Human Evaluation Framework for Radiology Report Generation

This module provides tools for collecting and analyzing human (radiologist) 
evaluations of generated reports, essential for medical AI publication.

Evaluation Dimensions:
1. Clinical Accuracy - Are findings correctly identified?
2. Completeness - Are all important findings mentioned?
3. Clinical Relevance - Are findings clinically meaningful?
4. Readability - Is the report well-structured and readable?
5. Actionability - Would the report lead to correct clinical decisions?

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import random
from scipy import stats
from loguru import logger


@dataclass
class HumanEvaluationSample:
    """A single sample for human evaluation."""
    sample_id: str
    image_path: str
    generated_report: str
    reference_report: str
    model_name: str  # Anonymized during evaluation
    
    # Evaluation scores (1-5 Likert scale)
    clinical_accuracy: Optional[int] = None
    completeness: Optional[int] = None
    clinical_relevance: Optional[int] = None
    readability: Optional[int] = None
    actionability: Optional[int] = None
    
    # Additional annotations
    critical_errors: Optional[List[str]] = None
    missing_findings: Optional[List[str]] = None
    hallucinated_findings: Optional[List[str]] = None
    evaluator_notes: Optional[str] = None
    evaluator_id: Optional[str] = None
    evaluation_time_seconds: Optional[float] = None


@dataclass
class EvaluationCriteria:
    """Criteria definitions for human evaluation."""
    
    CLINICAL_ACCURACY = {
        1: "Major errors - critical findings missed or incorrectly stated",
        2: "Significant errors - important findings incorrect",
        3: "Minor errors - small inaccuracies but main findings correct",
        4: "Accurate - findings correct with minor omissions",
        5: "Highly accurate - all findings correctly identified",
    }
    
    COMPLETENESS = {
        1: "Very incomplete - most findings missing",
        2: "Incomplete - several important findings missing",
        3: "Partially complete - some findings missing",
        4: "Mostly complete - minor findings missing",
        5: "Complete - all relevant findings mentioned",
    }
    
    CLINICAL_RELEVANCE = {
        1: "Not clinically useful",
        2: "Limited clinical utility",
        3: "Moderately useful",
        4: "Clinically useful",
        5: "Highly clinically useful",
    }
    
    READABILITY = {
        1: "Very difficult to understand",
        2: "Difficult to follow",
        3: "Readable with some effort",
        4: "Clear and well-structured",
        5: "Excellent clarity and structure",
    }
    
    ACTIONABILITY = {
        1: "Would lead to incorrect clinical decisions",
        2: "May lead to suboptimal decisions",
        3: "Adequate for clinical decision-making",
        4: "Good guidance for clinical decisions",
        5: "Excellent guidance for clinical decisions",
    }


class HumanEvaluationManager:
    """
    Manager for human evaluation studies.
    
    Handles:
    - Sample selection and randomization
    - Blind evaluation setup
    - Inter-rater reliability calculation
    - Result aggregation and statistics
    """
    
    def __init__(self, output_dir: str = "../data/human_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[HumanEvaluationSample] = []
        
    def prepare_evaluation_set(
        self,
        predictions: Dict[str, List[str]],  # model_name -> predictions
        references: List[str],
        image_paths: List[str],
        num_samples: int = 100,
        seed: int = 42,
    ) -> List[HumanEvaluationSample]:
        """
        Prepare a randomized, blinded evaluation set.
        
        Args:
            predictions: Dict mapping model names to their predictions
            references: Reference reports
            image_paths: Paths to X-ray images
            num_samples: Number of samples per model
            seed: Random seed for reproducibility
            
        Returns:
            List of evaluation samples (shuffled, model names anonymized)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Select random indices
        total_samples = len(references)
        selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        samples = []
        model_codes = {name: f"Model_{chr(65 + i)}" for i, name in enumerate(predictions.keys())}
        
        for idx in selected_indices:
            for model_name, preds in predictions.items():
                sample = HumanEvaluationSample(
                    sample_id=f"{idx}_{model_codes[model_name]}",
                    image_path=image_paths[idx],
                    generated_report=preds[idx],
                    reference_report=references[idx],
                    model_name=model_codes[model_name],  # Anonymized
                )
                samples.append(sample)
        
        # Shuffle for blind evaluation
        random.shuffle(samples)
        
        self.samples = samples
        self._save_model_codes(model_codes)
        
        logger.info(f"Prepared {len(samples)} samples for human evaluation")
        return samples
    
    def _save_model_codes(self, model_codes: Dict[str, str]):
        """Save model code mapping (to be revealed after evaluation)."""
        mapping_path = self.output_dir / "model_codes_SECRET.json"
        with open(mapping_path, 'w') as f:
            json.dump(model_codes, f, indent=2)
        logger.info(f"Model codes saved to {mapping_path} (keep secret until evaluation complete)")
    
    def export_evaluation_forms(self, format: str = 'csv') -> str:
        """
        Export evaluation forms for human evaluators.
        
        Args:
            format: 'csv', 'json', or 'html'
            
        Returns:
            Path to exported file
        """
        if not self.samples:
            raise ValueError("No samples prepared. Call prepare_evaluation_set first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            df = pd.DataFrame([asdict(s) for s in self.samples])
            # Remove scores columns for empty form
            score_cols = ['clinical_accuracy', 'completeness', 'clinical_relevance', 
                         'readability', 'actionability']
            for col in score_cols:
                df[col] = ''
            
            filepath = self.output_dir / f"evaluation_form_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            
        elif format == 'html':
            filepath = self.output_dir / f"evaluation_form_{timestamp}.html"
            self._export_html_form(filepath)
            
        else:  # json
            filepath = self.output_dir / f"evaluation_form_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump([asdict(s) for s in self.samples], f, indent=2)
        
        logger.info(f"Evaluation form exported to {filepath}")
        return str(filepath)
    
    def _export_html_form(self, filepath: Path):
        """Export an interactive HTML evaluation form."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Radiology Report Evaluation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .sample { border: 1px solid #ccc; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .reports { display: flex; gap: 20px; }
        .report { flex: 1; background: #f5f5f5; padding: 15px; border-radius: 4px; }
        .scores { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 15px; }
        .score-item { text-align: center; }
        .score-item label { display: block; margin-bottom: 5px; font-weight: bold; }
        select { width: 100%; padding: 5px; }
        textarea { width: 100%; height: 80px; margin-top: 10px; }
        h3 { color: #333; }
        .reference { background: #e8f5e9; }
        .generated { background: #fff3e0; }
    </style>
</head>
<body>
    <h1>Radiology Report Human Evaluation</h1>
    <p><strong>Instructions:</strong> For each sample, compare the generated report with the reference. 
    Rate on a 1-5 scale (1=Poor, 5=Excellent). Note any critical errors or missing findings.</p>
    
    <form id="evaluationForm">
        <input type="hidden" name="evaluator_id" id="evaluator_id">
        <p>Evaluator ID: <input type="text" id="evaluator_input" required></p>
        
        {samples}
        
        <button type="submit" style="padding: 15px 30px; font-size: 18px; margin-top: 20px;">
            Submit Evaluation
        </button>
    </form>
    
    <script>
        document.getElementById('evaluationForm').onsubmit = function(e) {{
            e.preventDefault();
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            console.log(JSON.stringify(data, null, 2));
            alert('Evaluation saved! Check console for data.');
        }};
    </script>
</body>
</html>
"""
        
        sample_html = ""
        for i, sample in enumerate(self.samples):
            sample_html += f"""
        <div class="sample">
            <h3>Sample {i+1} (ID: {sample.sample_id})</h3>
            <div class="reports">
                <div class="report reference">
                    <h4>Reference Report</h4>
                    <p>{sample.reference_report}</p>
                </div>
                <div class="report generated">
                    <h4>Generated Report ({sample.model_name})</h4>
                    <p>{sample.generated_report}</p>
                </div>
            </div>
            <div class="scores">
                <div class="score-item">
                    <label>Clinical Accuracy</label>
                    <select name="{sample.sample_id}_accuracy" required>
                        <option value="">--</option>
                        <option value="1">1 - Major errors</option>
                        <option value="2">2 - Significant errors</option>
                        <option value="3">3 - Minor errors</option>
                        <option value="4">4 - Accurate</option>
                        <option value="5">5 - Highly accurate</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Completeness</label>
                    <select name="{sample.sample_id}_completeness" required>
                        <option value="">--</option>
                        <option value="1">1</option><option value="2">2</option>
                        <option value="3">3</option><option value="4">4</option>
                        <option value="5">5</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Clinical Relevance</label>
                    <select name="{sample.sample_id}_relevance" required>
                        <option value="">--</option>
                        <option value="1">1</option><option value="2">2</option>
                        <option value="3">3</option><option value="4">4</option>
                        <option value="5">5</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Readability</label>
                    <select name="{sample.sample_id}_readability" required>
                        <option value="">--</option>
                        <option value="1">1</option><option value="2">2</option>
                        <option value="3">3</option><option value="4">4</option>
                        <option value="5">5</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Actionability</label>
                    <select name="{sample.sample_id}_actionability" required>
                        <option value="">--</option>
                        <option value="1">1</option><option value="2">2</option>
                        <option value="3">3</option><option value="4">4</option>
                        <option value="5">5</option>
                    </select>
                </div>
            </div>
            <textarea name="{sample.sample_id}_notes" placeholder="Notes: Critical errors, missing findings, hallucinations..."></textarea>
        </div>
"""
        
        html_content = html_template.format(samples=sample_html)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def load_completed_evaluations(self, filepath: str) -> List[HumanEvaluationSample]:
        """Load completed evaluation forms."""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            samples = []
            for _, row in df.iterrows():
                sample = HumanEvaluationSample(**row.to_dict())
                samples.append(sample)
        else:  # json
            with open(filepath, 'r') as f:
                data = json.load(f)
            samples = [HumanEvaluationSample(**s) for s in data]
        
        self.samples = samples
        return samples
    
    def compute_inter_rater_reliability(
        self,
        evaluations: Dict[str, List[HumanEvaluationSample]],  # evaluator_id -> samples
    ) -> Dict[str, float]:
        """
        Compute inter-rater reliability metrics.
        
        Args:
            evaluations: Dict mapping evaluator IDs to their evaluations
            
        Returns:
            Dictionary with Krippendorff's alpha and Fleiss' kappa for each dimension
        """
        dimensions = ['clinical_accuracy', 'completeness', 'clinical_relevance', 
                     'readability', 'actionability']
        
        results = {}
        
        for dim in dimensions:
            # Build rating matrix
            evaluator_ids = list(evaluations.keys())
            sample_ids = list(set(s.sample_id for s in evaluations[evaluator_ids[0]]))
            
            rating_matrix = []
            for sample_id in sample_ids:
                ratings = []
                for eval_id in evaluator_ids:
                    sample = next((s for s in evaluations[eval_id] if s.sample_id == sample_id), None)
                    if sample:
                        ratings.append(getattr(sample, dim))
                rating_matrix.append(ratings)
            
            # Compute Krippendorff's alpha (simplified)
            alpha = self._krippendorff_alpha(rating_matrix)
            results[f"{dim}_alpha"] = alpha
            
            # Compute average correlation
            if len(evaluator_ids) >= 2:
                corrs = []
                for i in range(len(evaluator_ids)):
                    for j in range(i+1, len(evaluator_ids)):
                        r1 = [rating_matrix[k][i] for k in range(len(sample_ids)) if rating_matrix[k][i] is not None]
                        r2 = [rating_matrix[k][j] for k in range(len(sample_ids)) if rating_matrix[k][j] is not None]
                        if r1 and r2:
                            corr, _ = stats.spearmanr(r1, r2)
                            corrs.append(corr)
                results[f"{dim}_spearman"] = np.mean(corrs) if corrs else 0.0
        
        return results
    
    def _krippendorff_alpha(self, rating_matrix: List[List[Optional[int]]]) -> float:
        """Simplified Krippendorff's alpha calculation."""
        # Filter out None values
        valid_ratings = []
        for ratings in rating_matrix:
            valid = [r for r in ratings if r is not None]
            if len(valid) >= 2:
                valid_ratings.append(valid)
        
        if not valid_ratings:
            return 0.0
        
        # Compute observed disagreement
        observed_disagreement = 0
        n_pairs = 0
        for ratings in valid_ratings:
            for i in range(len(ratings)):
                for j in range(i+1, len(ratings)):
                    observed_disagreement += (ratings[i] - ratings[j]) ** 2
                    n_pairs += 1
        
        if n_pairs == 0:
            return 1.0
        
        observed_disagreement /= n_pairs
        
        # Compute expected disagreement
        all_ratings = [r for ratings in valid_ratings for r in ratings]
        expected_disagreement = np.var(all_ratings)
        
        if expected_disagreement == 0:
            return 1.0
        
        alpha = 1 - (observed_disagreement / expected_disagreement)
        return max(0, min(1, alpha))
    
    def compute_model_rankings(self) -> pd.DataFrame:
        """
        Compute model rankings based on human evaluation scores.
        
        Returns:
            DataFrame with model rankings and scores
        """
        if not self.samples:
            raise ValueError("No samples loaded")
        
        # Load model codes
        codes_path = self.output_dir / "model_codes_SECRET.json"
        if codes_path.exists():
            with open(codes_path, 'r') as f:
                model_codes = json.load(f)
            # Reverse mapping
            code_to_model = {v: k for k, v in model_codes.items()}
        else:
            code_to_model = {}
        
        # Aggregate scores by model
        model_scores = {}
        dimensions = ['clinical_accuracy', 'completeness', 'clinical_relevance', 
                     'readability', 'actionability']
        
        for sample in self.samples:
            model = sample.model_name
            if model not in model_scores:
                model_scores[model] = {dim: [] for dim in dimensions}
            
            for dim in dimensions:
                score = getattr(sample, dim)
                if score is not None:
                    model_scores[model][dim].append(score)
        
        # Compute statistics
        results = []
        for model, scores in model_scores.items():
            result = {
                'Model Code': model,
                'Model Name': code_to_model.get(model, 'Unknown'),
            }
            
            overall_scores = []
            for dim in dimensions:
                if scores[dim]:
                    mean_score = np.mean(scores[dim])
                    std_score = np.std(scores[dim])
                    result[f'{dim}_mean'] = mean_score
                    result[f'{dim}_std'] = std_score
                    overall_scores.extend(scores[dim])
            
            result['overall_mean'] = np.mean(overall_scores) if overall_scores else 0
            result['overall_std'] = np.std(overall_scores) if overall_scores else 0
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('overall_mean', ascending=False)
        
        return df
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        rankings = self.compute_model_rankings()
        
        latex = """
\\begin{table}[h]
\\centering
\\caption{Human Evaluation Results (1-5 Likert Scale)}
\\label{tab:human_eval}
\\begin{tabular}{l|ccccc|c}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Complete.} & \\textbf{Relevance} & \\textbf{Readability} & \\textbf{Action.} & \\textbf{Overall} \\\\
\\hline
"""
        
        for _, row in rankings.iterrows():
            latex += f"{row['Model Name']} & "
            latex += f"{row.get('clinical_accuracy_mean', 0):.2f} & "
            latex += f"{row.get('completeness_mean', 0):.2f} & "
            latex += f"{row.get('clinical_relevance_mean', 0):.2f} & "
            latex += f"{row.get('readability_mean', 0):.2f} & "
            latex += f"{row.get('actionability_mean', 0):.2f} & "
            latex += f"\\textbf{{{row['overall_mean']:.2f}}} \\\\\n"
        
        latex += """\\hline
\\end{tabular}
\\end{table}
"""
        return latex


def setup_human_evaluation(
    predictions: Dict[str, List[str]],
    references: List[str],
    image_paths: List[str],
    output_dir: str = "../data/human_evaluation",
    num_samples: int = 100,
) -> str:
    """
    Quick setup for human evaluation study.
    
    Returns path to exported evaluation form.
    """
    manager = HumanEvaluationManager(output_dir)
    manager.prepare_evaluation_set(predictions, references, image_paths, num_samples)
    
    # Export both CSV and HTML forms
    csv_path = manager.export_evaluation_forms('csv')
    html_path = manager.export_evaluation_forms('html')
    
    print(f"Human evaluation forms created:")
    print(f"  CSV: {csv_path}")
    print(f"  HTML: {html_path}")
    print(f"\nInstructions:")
    print("1. Distribute forms to radiologist evaluators")
    print("2. Collect completed evaluations")
    print("3. Use HumanEvaluationManager.load_completed_evaluations() to analyze results")
    
    return csv_path
