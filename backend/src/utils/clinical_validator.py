"""
Clinical Validation Framework for XR2Text

NOVEL: Comprehensive Clinical Validation System

This module provides tools for clinical validation of generated reports:
1. Clinical entity extraction and validation
2. Anatomical region coverage analysis
3. Severity assessment
4. Comparison with radiologist reports
5. Error categorization by clinical significance

This is novel because:
- Most work focuses on NLG metrics (BLEU, ROUGE) but not clinical accuracy
- We provide structured clinical validation tools
- Enables real-world deployment assessment

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ClinicalSeverity(Enum):
    """Clinical severity levels."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of clinical errors."""
    MISSING_FINDING = "missing_finding"
    FALSE_POSITIVE = "false_positive"
    SEVERITY_MISMATCH = "severity_mismatch"
    LOCATION_ERROR = "location_error"
    NORMAL = "normal"  # No error


@dataclass
class ClinicalFinding:
    """Represents a clinical finding."""
    entity: str
    location: str
    severity: ClinicalSeverity
    confidence: float = 1.0
    modifiers: List[str] = None


@dataclass
class ClinicalReport:
    """Structured clinical report."""
    findings: List[ClinicalFinding]
    impression: str
    anatomical_regions: List[str]
    severity: ClinicalSeverity


class ClinicalValidator:
    """
    NOVEL: Clinical Validation System
    
    Validates generated reports for clinical accuracy and completeness.
    """
    
    def __init__(self):
        # Clinical entities and their synonyms
        self.clinical_entities = {
            'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
            'pneumonia': ['pneumonia', 'pneumonic', 'pulmonary infection'],
            'effusion': ['effusion', 'pleural effusion', 'fluid'],
            'edema': ['edema', 'pulmonary edema', 'fluid overload'],
            'consolidation': ['consolidation', 'consolidated', 'opacity'],
            'atelectasis': ['atelectasis', 'collapse', 'collapsed'],
            'pneumothorax': ['pneumothorax', 'air', 'collapsed lung'],
            'mass': ['mass', 'lesion', 'tumor'],
            'nodule': ['nodule', 'nodular'],
            'fracture': ['fracture', 'broken', 'break'],
        }
        
        # Anatomical regions
        self.anatomical_regions = [
            'right_lung', 'left_lung', 'heart', 'mediastinum',
            'spine', 'diaphragm', 'costophrenic_angles'
        ]
        
        # Severity indicators
        self.severity_keywords = {
            ClinicalSeverity.NORMAL: ['normal', 'clear', 'no acute', 'unremarkable'],
            ClinicalSeverity.MILD: ['mild', 'slight', 'minimal', 'trace'],
            ClinicalSeverity.MODERATE: ['moderate', 'some', 'present'],
            ClinicalSeverity.SEVERE: ['severe', 'extensive', 'large', 'marked', 'significant'],
            ClinicalSeverity.CRITICAL: ['massive', 'critical', 'life-threatening', 'emergent'],
        }
    
    def extract_findings(self, text: str) -> List[ClinicalFinding]:
        """
        Extract structured clinical findings from text.
        
        Args:
            text: Report text
            
        Returns:
            List of ClinicalFinding objects
        """
        text_lower = text.lower()
        findings = []
        
        # Extract entities
        for entity, synonyms in self.clinical_entities.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    # Determine severity
                    severity = self._determine_severity(text_lower, synonym)
                    
                    # Determine location
                    location = self._determine_location(text_lower, synonym)
                    
                    finding = ClinicalFinding(
                        entity=entity,
                        location=location,
                        severity=severity,
                    )
                    findings.append(finding)
                    break  # Only count once per entity
        
        return findings
    
    def _determine_severity(self, text: str, entity: str) -> ClinicalSeverity:
        """Determine severity of a finding."""
        # Check for severity keywords near the entity
        entity_pos = text.find(entity)
        if entity_pos == -1:
            return ClinicalSeverity.MODERATE
        
        # Look in a window around the entity
        window_start = max(0, entity_pos - 50)
        window_end = min(len(text), entity_pos + len(entity) + 50)
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
        
        window_start = max(0, entity_pos - 50)
        window_end = min(len(text), entity_pos + len(entity) + 50)
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
        Validate generated report against reference.
        
        Args:
            generated_text: Generated report
            reference_text: Reference (radiologist) report
            
        Returns:
            Validation results dictionary
        """
        gen_findings = self.extract_findings(generated_text)
        ref_findings = self.extract_findings(reference_text)
        
        # Extract entity sets
        gen_entities = {f.entity for f in gen_findings}
        ref_entities = {f.entity for f in ref_findings}
        
        # Compute metrics
        true_positives = gen_entities & ref_entities
        false_positives = gen_entities - ref_entities
        false_negatives = ref_entities - gen_entities
        
        precision = len(true_positives) / len(gen_entities) if gen_entities else 0.0
        recall = len(true_positives) / len(ref_entities) if ref_entities else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Categorize errors
        errors = []
        
        # Missing findings (false negatives)
        for entity in false_negatives:
            errors.append({
                'type': ErrorType.MISSING_FINDING.value,
                'entity': entity,
                'severity': 'high' if entity in ['pneumothorax', 'mass', 'fracture'] else 'medium',
            })
        
        # False positives
        for entity in false_positives:
            errors.append({
                'type': ErrorType.FALSE_POSITIVE.value,
                'entity': entity,
                'severity': 'high' if entity in ['pneumothorax', 'mass'] else 'medium',
            })
        
        # Severity mismatches
        for ref_finding in ref_findings:
            gen_finding = next(
                (f for f in gen_findings if f.entity == ref_finding.entity),
                None
            )
            if gen_finding and gen_finding.severity != ref_finding.severity:
                errors.append({
                    'type': ErrorType.SEVERITY_MISMATCH.value,
                    'entity': ref_finding.entity,
                    'reference_severity': ref_finding.severity.value,
                    'generated_severity': gen_finding.severity.value,
                    'severity': 'medium',
                })
        
        # Count critical errors
        critical_errors = sum(1 for e in errors if e['severity'] == 'high')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'errors': errors,
            'critical_errors': critical_errors,
            'total_errors': len(errors),
            'clinical_accuracy': 1.0 - (critical_errors / max(len(ref_entities), 1)),
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

