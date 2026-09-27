"""
Khmer Content Moderation & Toxicity Detection Engine
Scans Khmer comments for profanity, hate speech, threats, harassment, and scam patterns.
Provides toxicity scoring, violation categorization, and keyword hotlist extraction.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Set, Tuple
import pandas as pd


class KhmerToxicityDetector:
    """
    Detects toxic, inappropriate, and harmful content in Khmer social media comments.
    Combines rule-based category dictionaries, severity weighting, and sentiment thresholds.
    """

    # --- CATEGORIZED KHMER BLOCKLIST ---
    PROFANITY_KEYWORDS: Set[str] = {
        "ចុយ", "ចុយម្រាយ", "អាចុយ", "មីចុយ", "ក្បាលកណ្ដួយ", "កណ្ដួយ", "ក្ដ", "អាចុយម្រាយ",
        "អាឆ្កែ", "មីឆ្កែ", "អាល្ងង់", "មីល្ងង់", "ល្ងង់ដូចក្របី", "ថោកទាប", "អាថោក", "មីថោក",
        "អាព្រៃផ្សៃ", "អាឆ្កួត", "មីឆ្កួត", "អាខ្វាក់", "ក្បាលខូច", "ក្បាលខ្វាក់", "អាសំរាម",
        "អាអត់កំណើត", "អាគម្រក់", "មីគម្រក់", "ចង្រៃ", "អាចង្រៃ", "មីចង្រៃ", "អាចោរ", "មីចោរ",
        # English slurs commonly used in Cambodia
        "fuck", "fucking", "bitch", "bastard", "dick", "asshole", "shit"
    }

    HATE_SPEECH_KEYWORDS: Set[str] = {
        "អាសត្វ", "អាតិរច្ឆាន", "អាពូជអន់", "ពូជអន់", "អាមនុស្សថោក", "មីថោកទាប", "អត់ពូជ",
        "មនុស្សអត់ខួរ", "គ្មានខួរ", "ខួរក្បាលស្មើនឹងកែងជើង", "មុខដូចខ្មោច", "មុខអាក្រក់ដូចខ្មោច",
        "ខ្ទើយចង្រៃ", "រើសអើង", "អាជនជាតិក្បត់", "អាឈ្លានពាន", "អាអាគក",
        # English hate terms
        "idiot", "moron", "stupid", "retard", "faggot", "scumbag"
    }

    THREAT_VIOLENCE_KEYWORDS: Set[str] = {
        "សម្លាប់", "សម្លាប់ចោល", "វ៉ៃបំបែកក្បាល", "កាប់ចោល", "កាប់ឲ្យងាប់", "បាញ់ចោល",
        "ចាំមើលប្រយ័ត្នងាប់", "អាងាប់", "ចង់ងាប់មែន", "វាយឲ្យងាប់", "យកឡានបុកឲ្យងាប់",
        "ដុតផ្ទះចោល", "អាងាប់តៃហោង", "ទៅងាប់ទៅ", "ងាប់ឲ្យបាត់ទៅ", "អាប្រកាច់",
        # English threats
        "kill you", "die", "go die", "burn in hell"
    }

    SCAM_FRAUD_KEYWORDS: Set[str] = {
        "បោកប្រាស់", "បោកលុយ", "ឆបោក", "កុហកភូតភរ", "ផ្ទេរប្រាក់មុនគេចខ្លួន", "កុំជឿគេបោក",
        "ឈ្នះរង្វាន់ក្លែងក្លាយ", "ផ្ទេរលុយមកលេខ", "លុយកក់រួចគេច", "ហាងបោកគេ", "លួចលុយ",
        "ផ្ញើសារយកកូដ", "ក្លែងបន្លំ", "fake news", "scammer", "scam"
    }

    def __init__(self, toxicity_threshold: float = 0.35):
        """
        Initialize the detector with confidence/severity weights.
        
        Args:
            toxicity_threshold: Score (0.0 - 1.0) above which a comment is flagged.
        """
        self.toxicity_threshold = toxicity_threshold
        
        # Sort terms by length descending so longer compound terms match first
        self._profanity_sorted = sorted(self.PROFANITY_KEYWORDS, key=len, reverse=True)
        self._hate_sorted = sorted(self.HATE_SPEECH_KEYWORDS, key=len, reverse=True)
        self._threat_sorted = sorted(self.THREAT_VIOLENCE_KEYWORDS, key=len, reverse=True)
        self._scam_sorted = sorted(self.SCAM_FRAUD_KEYWORDS, key=len, reverse=True)

    def scan_text(
        self,
        text: str,
        sentiment_score: float = 0.0,
        sentiment_label: str = "Neutral"
    ) -> Dict[str, Any]:
        """
        Scans a comment for toxic terms and returns moderation diagnostics.
        
        Returns:
            Dict containing:
            - 'is_flagged': bool
            - 'toxicity_score': float (0.0 to 1.0)
            - 'severity': 'Clean' | 'Low Risk' | 'Moderate Risk' | 'High Risk / Severe'
            - 'violation_category': 'None' | 'Profanity / Vulgarity' | 'Hate Speech' | 'Threat / Violence' | 'Scam / Fraud' | 'Severe Toxicity'
            - 'flagged_keywords': List[str]
            - 'highlighted_text': str (with HTML <mark> tags)
        """
        if not text or not isinstance(text, str):
            return {
                "is_flagged": False,
                "toxicity_score": 0.0,
                "severity": "Clean",
                "violation_category": "None",
                "flagged_keywords": [],
                "highlighted_text": text or ""
            }

        text_lower = text.lower()
        flagged_terms: List[str] = []
        categories_detected: List[Tuple[str, float]] = []

        # 1. Check Threat / Violence (Highest Severity: weight 1.0)
        for term in self._threat_sorted:
            if term in text_lower:
                flagged_terms.append(term)
                categories_detected.append(("Threat / Violence", 1.0))

        # 2. Check Hate Speech / Harassment (Severity weight: 0.85)
        for term in self._hate_sorted:
            if term in text_lower:
                flagged_terms.append(term)
                categories_detected.append(("Hate Speech / Harassment", 0.85))

        # 3. Check Profanity / Vulgarity (Severity weight: 0.75)
        for term in self._profanity_sorted:
            if term in text_lower:
                flagged_terms.append(term)
                categories_detected.append(("Profanity / Vulgarity", 0.75))

        # 4. Check Scam / Fraud (Severity weight: 0.70)
        for term in self._scam_sorted:
            if term in text_lower:
                flagged_terms.append(term)
                categories_detected.append(("Scam / Fraud", 0.70))

        # Deduplicate terms while preserving order
        unique_flagged = list(dict.fromkeys(flagged_terms))

        # Base toxicity calculation
        max_cat_weight = max([w for _, w in categories_detected], default=0.0)
        match_count = len(unique_flagged)

        if match_count > 0:
            # Term-based toxicity calculation
            # Starts at primary category weight and scales with match count
            base_score = max_cat_weight + (min(match_count - 1, 3) * 0.1)
            
            # Boost if sentiment is also strongly negative
            if sentiment_score < -1.5 or sentiment_label == "Negative":
                base_score += 0.15
            
            toxicity_score = min(1.0, max(0.40, base_score))
            is_flagged = True

            # Determine dominant category
            dominant_category = categories_detected[0][0] if categories_detected else "Inappropriate Content"

        else:
            # No explicit blocklist keyword match.
            # Check for extreme hostility based on very negative sentiment score
            if sentiment_score <= -3.5:
                toxicity_score = 0.45
                is_flagged = True
                dominant_category = "Hostile Sentiment"
            elif sentiment_score <= -2.0:
                toxicity_score = 0.25
                is_flagged = False
                dominant_category = "None"
            else:
                toxicity_score = max(0.0, -sentiment_score * 0.05) if sentiment_score < 0 else 0.0
                is_flagged = False
                dominant_category = "None"

        # Assign Severity Label
        if toxicity_score >= 0.70:
            severity = "High Risk / Severe"
        elif toxicity_score >= 0.40:
            severity = "Moderate Risk"
        elif toxicity_score >= 0.20:
            severity = "Low Risk"
        else:
            severity = "Clean"

        # Generate HTML highlighted text for UI rendering
        highlighted = text
        for term in unique_flagged:
            # Case-insensitive replacement with styled tag
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span style="background-color: rgba(239, 68, 68, 0.25); color: #DC2626; padding: 1px 4px; border-radius: 4px; font-weight: 600;">{term}</span>',
                highlighted
            )

        return {
            "is_flagged": is_flagged,
            "toxicity_score": round(float(toxicity_score), 2),
            "severity": severity,
            "violation_category": dominant_category,
            "flagged_keywords": unique_flagged,
            "highlighted_text": highlighted
        }

    def scan_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "comment",
        sentiment_score_col: str = "Sentiment_Score",
        sentiment_label_col: str = "Sentiment"
    ) -> pd.DataFrame:
        """
        Batch scans a DataFrame of comments and appends moderation columns:
        - 'Is_Flagged': bool
        - 'Toxicity_Score': float
        - 'Severity': str
        - 'Violation_Category': str
        - 'Flagged_Keywords': str
        - 'Moderation_Status': 'Pending Review' | 'Clean'
        """
        if df.empty or text_col not in df.columns:
            return df

        result_df = df.copy()

        is_flagged_list = []
        toxicity_scores = []
        severities = []
        categories = []
        flagged_kw_list = []
        highlighted_list = []
        statuses = []

        for _, row in result_df.iterrows():
            text = str(row.get(text_col, ""))
            score = float(row.get(sentiment_score_col, 0.0)) if sentiment_score_col in row else 0.0
            sentiment = str(row.get(sentiment_label_col, "Neutral")) if sentiment_label_col in row else "Neutral"

            diag = self.scan_text(text, sentiment_score=score, sentiment_label=sentiment)

            is_flagged_list.append(diag["is_flagged"])
            toxicity_scores.append(diag["toxicity_score"])
            severities.append(diag["severity"])
            categories.append(diag["violation_category"])
            flagged_kw_list.append(", ".join(diag["flagged_keywords"]))
            highlighted_list.append(diag["highlighted_text"])
            statuses.append("Pending Review" if diag["is_flagged"] else "Approved")

        result_df["Is_Flagged"] = is_flagged_list
        result_df["Toxicity_Score"] = toxicity_scores
        result_df["Severity"] = severities
        result_df["Violation_Category"] = categories
        result_df["Flagged_Keywords"] = flagged_kw_list
        result_df["Highlighted_Comment"] = highlighted_list
        result_df["Moderation_Status"] = statuses

        return result_df
