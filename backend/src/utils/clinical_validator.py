"""
Clinical Validation Framework for XR2Text

NOVEL: Comprehensive Clinical Validation System with Negation-Aware NER

This module provides tools for clinical validation of generated reports:
1. Clinical entity extraction with NEGATION DETECTION (NOVEL)
2. Anatomical region coverage analysis
3. Severity assessment with uncertainty handling
4. Comparison with radiologist reports
5. Error categorization by clinical significance
6. Per-finding precision/recall metrics

Key Innovations:
- Negation-aware entity extraction (handles "no cardiomegaly" vs "cardiomegaly")
- Uncertainty detection ("possible", "cannot exclude", "likely")
- Context-aware severity inference
- Critical error weighting for clinical safety

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict


class ClinicalSeverity(Enum):
    """Clinical severity levels."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class FindingStatus(Enum):
    """Status of a clinical finding."""
    PRESENT = "present"      # Finding is present
    ABSENT = "absent"        # Finding is explicitly negated
    UNCERTAIN = "uncertain"  # Finding is uncertain (possible, cannot exclude)


class ErrorType(Enum):
    """Types of clinical errors."""
    MISSING_FINDING = "missing_finding"
    FALSE_POSITIVE = "false_positive"
    SEVERITY_MISMATCH = "severity_mismatch"
    LOCATION_ERROR = "location_error"
    NEGATION_ERROR = "negation_error"  # NOVEL: Incorrect negation
    NORMAL = "normal"  # No error


@dataclass
class ClinicalFinding:
    """Represents a clinical finding with status awareness."""
    entity: str
    location: str
    severity: ClinicalSeverity
    status: FindingStatus = FindingStatus.PRESENT  # NOVEL: track if negated
    confidence: float = 1.0
    modifiers: List[str] = field(default_factory=list)


@dataclass
class ClinicalReport:
    """Structured clinical report."""
    findings: List[ClinicalFinding]
    impression: str
    anatomical_regions: List[str]
    severity: ClinicalSeverity


class ClinicalValidator:
    """
    NOVEL: Clinical Validation System with Negation-Aware NER

    Validates generated reports for clinical accuracy and completeness.
    Includes negation detection to distinguish "no cardiomegaly" from "cardiomegaly".

    Configuration:
        negation_window: Character window size to search for negation patterns (default: 60)
        uncertainty_window: Character window size for uncertainty patterns (default: 40)
        severity_window: Character window size for severity keywords (default: 50)
        location_window: Character window size for location keywords (default: 50)
    """

    # Configuration constants (previously magic numbers)
    DEFAULT_NEGATION_WINDOW = 60  # Characters before entity to check for negation
    DEFAULT_UNCERTAINTY_WINDOW = 40  # Characters around entity for uncertainty
    DEFAULT_SEVERITY_WINDOW = 50  # Characters around entity for severity
    DEFAULT_LOCATION_WINDOW = 50  # Characters around entity for location

    def __init__(
        self,
        negation_window: int = DEFAULT_NEGATION_WINDOW,
        uncertainty_window: int = DEFAULT_UNCERTAINTY_WINDOW,
        severity_window: int = DEFAULT_SEVERITY_WINDOW,
        location_window: int = DEFAULT_LOCATION_WINDOW,
    ):
        # Configurable window sizes
        self.negation_window = negation_window
        self.uncertainty_window = uncertainty_window
        self.severity_window = severity_window
        self.location_window = location_window
        # Clinical entities and their synonyms
        self.clinical_entities = {
            'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'heart enlargement'],
            'pneumonia': ['pneumonia', 'pneumonic', 'pulmonary infection', 'infectious infiltrate'],
            'effusion': ['effusion', 'pleural effusion', 'fluid collection'],
            'edema': ['edema', 'pulmonary edema', 'fluid overload', 'congestion'],
            'consolidation': ['consolidation', 'consolidated', 'airspace opacity'],
            'atelectasis': ['atelectasis', 'collapse', 'volume loss'],
            'pneumothorax': ['pneumothorax', 'collapsed lung'],
            'mass': ['mass', 'lesion', 'tumor', 'neoplasm'],
            'nodule': ['nodule', 'nodular', 'pulmonary nodule'],
            'fracture': ['fracture', 'broken', 'rib fracture'],
            'emphysema': ['emphysema', 'hyperinflation', 'copd'],
            'fibrosis': ['fibrosis', 'scarring', 'interstitial'],
            'hernia': ['hernia', 'hiatal hernia'],
            'scoliosis': ['scoliosis', 'curvature'],
        }

        # NOVEL: Negation patterns for clinical text
        self.negation_patterns = [
            'no ', 'no evidence of', 'without', 'negative for',
            'absence of', 'absent', 'not ', 'denies', 'ruled out',
            'no acute', 'no significant', 'no definite', 'no obvious',
            'unremarkable', 'clear of', 'free of', 'resolution of',
            'resolved', 'improved', 'no longer', 'cleared',
        ]

        # NOVEL: Uncertainty patterns
        self.uncertainty_patterns = [
            'possible', 'probable', 'suspected', 'cannot exclude',
            'may represent', 'could be', 'likely', 'suggestive of',
            'consistent with', 'compatible with', 'concerning for',
            'question', 'questionable', 'versus', 'vs',
        ]

        # NOVEL: Positive assertion patterns
        self.positive_patterns = [
            'present', 'noted', 'seen', 'identified', 'demonstrates',
            'shows', 'reveals', 'confirmed', 'evident', 'obvious',
            'new', 'worsening', 'increased', 'developing', 'progression',
            'persistent', 'unchanged', 'stable',  # These indicate presence
        ]

        # Anatomical regions
        self.anatomical_regions = [
            'right_lung', 'left_lung', 'heart', 'mediastinum',
            'spine', 'diaphragm', 'costophrenic_angles'
        ]

        # Severity indicators
        self.severity_keywords = {
            ClinicalSeverity.NORMAL: ['normal', 'clear', 'no acute', 'unremarkable', 'negative'],
            ClinicalSeverity.MILD: ['mild', 'slight', 'minimal', 'trace', 'small', 'subtle'],
            ClinicalSeverity.MODERATE: ['moderate', 'some', 'present', 'modest'],
            ClinicalSeverity.SEVERE: ['severe', 'extensive', 'large', 'marked', 'significant'],
            ClinicalSeverity.CRITICAL: ['massive', 'critical', 'life-threatening', 'emergent', 'tension'],
        }

        # Critical findings that are clinically dangerous to miss or falsely report
        self.critical_findings = ['pneumothorax', 'mass', 'fracture', 'emphysema']

    def _is_negated(self, text: str, entity_pos: int) -> bool:
        """
        NOVEL: Check if entity at given position is negated.

        Args:
            text: Full text (lowercase)
            entity_pos: Position of entity in text

        Returns:
            True if the entity is negated
        """
        # Check window before entity (configurable, default 60 chars)
        window_start = max(0, entity_pos - self.negation_window)
        context_before = text[window_start:entity_pos]

        # Check if any negation pattern appears before entity
        for neg_pattern in self.negation_patterns:
            if neg_pattern in context_before:
                # Make sure there's no positive pattern after negation
                neg_pos = context_before.rfind(neg_pattern)
                text_after_neg = context_before[neg_pos + len(neg_pattern):]

                # Check for positive patterns between negation and entity
                has_positive = any(pos in text_after_neg for pos in self.positive_patterns)
                if not has_positive:
                    return True

        return False

    def _is_uncertain(self, text: str, entity_pos: int) -> bool:
        """
        NOVEL: Check if entity at given position has uncertainty.

        Args:
            text: Full text (lowercase)
            entity_pos: Position of entity in text

        Returns:
            True if the entity has uncertainty markers
        """
        window_start = max(0, entity_pos - self.uncertainty_window)
        window_end = min(len(text), entity_pos + self.uncertainty_window)
        context = text[window_start:window_end]

        return any(unc in context for unc in self.uncertainty_patterns)

    def extract_findings(self, text: str) -> List[ClinicalFinding]:
        """
        NOVEL: Extract structured clinical findings with negation awareness.

        Args:
            text: Report text

        Returns:
            List of ClinicalFinding objects with status (present/absent/uncertain)
        """
        text_lower = text.lower()
        findings = []
        found_entities = set()  # Track to avoid duplicates

        # Extract entities
        for entity, synonyms in self.clinical_entities.items():
            if entity in found_entities:
                continue

            for synonym in synonyms:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(synonym, start)
                    if pos == -1:
                        break

                    # Determine status (negated, uncertain, or present)
                    if self._is_negated(text_lower, pos):
                        status = FindingStatus.ABSENT
                    elif self._is_uncertain(text_lower, pos):
                        status = FindingStatus.UNCERTAIN
                    else:
                        status = FindingStatus.PRESENT

                    # Determine severity
                    severity = self._determine_severity(text_lower, synonym)

                    # Determine location
                    location = self._determine_location(text_lower, synonym)

                    finding = ClinicalFinding(
                        entity=entity,
                        location=location,
                        severity=severity,
                        status=status,
                    )
                    findings.append(finding)
                    found_entities.add(entity)
                    break  # Only count first occurrence per entity

                if entity in found_entities:
                    break

        return findings

    def extract_findings_detailed(self, text: str) -> Dict[str, Set[str]]:
        """
        NOVEL: Extract findings categorized by status.

        Returns:
            Dict with 'present', 'absent', 'uncertain' sets of entity names
        """
        findings = self.extract_findings(text)

        result = {
            'present': set(),
            'absent': set(),
            'uncertain': set(),
        }

        for f in findings:
            if f.status == FindingStatus.PRESENT:
                result['present'].add(f.entity)
            elif f.status == FindingStatus.ABSENT:
                result['absent'].add(f.entity)
            else:
                result['uncertain'].add(f.entity)

        return result
    
    def _determine_severity(self, text: str, entity: str) -> ClinicalSeverity:
        """Determine severity of a finding."""
        # Check for severity keywords near the entity
        entity_pos = text.find(entity)
        if entity_pos == -1:
            return ClinicalSeverity.MODERATE

        # Look in a window around the entity (configurable)
        window_start = max(0, entity_pos - self.severity_window)
        window_end = min(len(text), entity_pos + len(entity) + self.severity_window)
        context = text[window_start:window_end]
        
        # Check severity keywords
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in context:
                    return severity
        
        return ClinicalSeverity.MODERATE
    
    def _determine_location(self, text: str, entity: str) -> str:
        """Determine anatomical location of a finding."""
        location_keywords = {
            'right': ['right', 'rt'],
            'left': ['left', 'lt'],
            'bilateral': ['bilateral', 'both', 'bilaterally'],
            'upper': ['upper', 'superior'],
            'lower': ['lower', 'inferior'],
            'middle': ['middle', 'mid'],
        }
        
        entity_pos = text.find(entity)
        if entity_pos == -1:
            return 'unspecified'

        # Look in a window around the entity (configurable)
        window_start = max(0, entity_pos - self.location_window)
        window_end = min(len(text), entity_pos + len(entity) + self.location_window)
        context = text[window_start:window_end]

        locations = []
        for loc, keywords in location_keywords.items():
            if any(kw in context for kw in keywords):
                locations.append(loc)

        return ' '.join(locations) if locations else 'unspecified'
    
    def validate_report(
        self,
        generated_text: str,
        reference_text: str,
    ) -> Dict:
        """
        NOVEL: Validate generated report against reference with negation awareness.

        Args:
            generated_text: Generated report
            reference_text: Reference (radiologist) report

        Returns:
            Validation results dictionary with negation-aware metrics
        """
        # Extract findings with status
        gen_detailed = self.extract_findings_detailed(generated_text)
        ref_detailed = self.extract_findings_detailed(reference_text)

        gen_present = gen_detailed['present']
        gen_absent = gen_detailed['absent']
        ref_present = ref_detailed['present']
        ref_absent = ref_detailed['absent']

        # NOVEL: Negation-aware metrics
        # True positives: correctly identified PRESENT findings
        tp_present = gen_present & ref_present

        # True negatives: correctly identified ABSENT findings
        tn_absent = gen_absent & ref_absent

        # False positives: said present but actually absent or not mentioned
        fp = gen_present - ref_present

        # False negatives: reference present but not in generated
        fn = ref_present - gen_present

        # CRITICAL: Negation errors (said present when actually negated - DANGEROUS)
        negation_errors = gen_present & ref_absent

        # Compute precision/recall
        total_gen_positive = len(gen_present)
        total_ref_positive = len(ref_present)

        precision = len(tp_present) / total_gen_positive if total_gen_positive > 0 else 1.0
        recall = len(tp_present) / total_ref_positive if total_ref_positive > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Categorize errors
        errors = []

        # Missing findings (false negatives)
        for entity in fn:
            is_critical = entity in self.critical_findings
            errors.append({
                'type': ErrorType.MISSING_FINDING.value,
                'entity': entity,
                'severity': 'high' if is_critical else 'medium',
                'clinical_impact': 'May miss important diagnosis' if is_critical else 'Minor omission',
            })

        # False positives (excluding negation errors)
        for entity in (fp - negation_errors):
            is_critical = entity in self.critical_findings
            errors.append({
                'type': ErrorType.FALSE_POSITIVE.value,
                'entity': entity,
                'severity': 'high' if is_critical else 'medium',
                'clinical_impact': 'May cause unnecessary follow-up' if is_critical else 'Minor overcall',
            })

        # NOVEL: Negation errors (most critical - said positive when negated)
        for entity in negation_errors:
            errors.append({
                'type': ErrorType.NEGATION_ERROR.value,
                'entity': entity,
                'severity': 'critical',  # Always critical - dangerous clinical error
                'clinical_impact': f'CRITICAL: Reported {entity} as present when explicitly negated',
            })

        # Count errors by severity
        critical_errors = sum(1 for e in errors if e['severity'] in ['critical', 'high'])
        negation_error_count = sum(1 for e in errors if e['type'] == ErrorType.NEGATION_ERROR.value)

        # Clinical accuracy penalizes negation errors heavily
        clinical_accuracy = 1.0 - (
            (critical_errors + 2 * negation_error_count) /
            max(total_ref_positive + len(ref_absent), 1)
        )
        clinical_accuracy = max(0.0, clinical_accuracy)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(tp_present),
            'true_negatives': len(tn_absent),
            'false_positives': len(fp),
            'false_negatives': len(fn),
            'negation_errors': negation_error_count,  # NOVEL metric
            'errors': errors,
            'critical_errors': critical_errors,
            'total_errors': len(errors),
            'clinical_accuracy': clinical_accuracy,
            'negation_accuracy': 1.0 - (negation_error_count / max(len(ref_absent), 1)),  # NOVEL
        }
    
    def batch_validate(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> Dict:
        """
        Validate a batch of reports.
        
        Args:
            generated_texts: List of generated reports
            reference_texts: List of reference reports
            
        Returns:
            Aggregate validation metrics
        """
        all_results = [
            self.validate_report(gen, ref)
            for gen, ref in zip(generated_texts, reference_texts)
        ]
        
        # Aggregate metrics
        avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
        avg_clinical_accuracy = sum(r['clinical_accuracy'] for r in all_results) / len(all_results)
        
        total_errors = sum(r['total_errors'] for r in all_results)
        total_critical_errors = sum(r['critical_errors'] for r in all_results)
        
        # Error distribution
        error_types = {}
        for result in all_results:
            for error in result['errors']:
                error_type = error['type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1': avg_f1,
            'average_clinical_accuracy': avg_clinical_accuracy,
            'total_errors': total_errors,
            'total_critical_errors': total_critical_errors,
            'error_distribution': error_types,
            'per_sample_results': all_results,
        }

