"""
Khmer Sentiment Analysis Engine
Combines rule-based lexical matching with contextual negation and confidence scoring,
with integrated support for scikit-learn TF-IDF + Logistic Regression / SVM models.
"""

from __future__ import annotations
import os
import math
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessor import KhmerPreprocessor


class KhmerSentimentEngine:
    """
    Production-ready Sentiment Analysis Engine for Khmer Social Media Text.
    Features:
    - Rule-based lexicon with contextual negators, intensifiers, and emoji signals.
    - Confidence scoring (0.50 - 1.00).
    - Scikit-learn TF-IDF model integration with automatic training/persistence.
    - Hybrid classification mode combining Lexicon + ML.
    """

    # --- KHMER LEXICON ---
    # Positive words with base polarity weights (1.0 to 2.5)
    POSITIVE_LEXICON: Dict[str, float] = {
        # Core Positive
        "ល្អ": 1.5,
        "ល្អណាស់": 2.2,
        "ល្អខ្លាំង": 2.2,
        "ល្អឥតខ្ចោះ": 2.5,
        "ស្រលាញ់": 1.8,
        "ចូលចិត្ត": 1.5,
        "ពេញចិត្ត": 1.7,
        "សប្បាយ": 1.4,
        "សប្បាយរីករាយ": 1.8,
        "អរគុណ": 1.6,
        "អរគុណច្រើន": 2.0,
        "ពូកែ": 1.5,
        "អស្ចារ្យ": 2.0,
        "ត្រឹមត្រូវ": 1.3,
        "គាំទ្រ": 1.5,
        "ឡូយ": 1.4,
        "ឡូយណាស់": 1.8,
        "ស្អាត": 1.5,
        "ស្អាតណាស់": 1.9,
        "អែម": 1.3,
        "អបអរ": 1.5,
        "អបអរសាទរ": 1.8,
        "ជោគជ័យ": 1.8,
        "មានប្រយោជន៍": 1.7,
        "ច្បាស់": 1.2,
        "ច្បាស់លាស់": 1.5,
        "កោតសរសើរ": 1.9,
        "អភិវឌ្ឍន៍": 1.3,
        "រួសរាយ": 1.4,
        "ស្មោះត្រង់": 1.6,
        "គុណភាព": 1.2,
        "គុណភាពល្អ": 2.0,
        "សេវាកម្មល្អ": 2.0,
        "តម្លៃសមរម្យ": 1.8,
        "លឿន": 1.2,
        "លឿនរហ័ស": 1.6,
        "ពេញបេះដូង": 2.0,
        "គំរូ": 1.2,
        "គំរូល្អ": 1.8,
        "ឆ្លាត": 1.5,
        "ពិត": 1.0,
        "ពិតជាល្អ": 2.0,
        "ញាក់សាច់": 1.5,
        "ចូលចិត្តណាស់": 2.2,
        "រីករាយ": 1.5,
        "រំភើប": 1.6,
        "ជួយបានច្រើន": 1.8,
        "ពេញបេះដូង❤️": 2.2,
        
        # English social slang commonly used in Cambodia
        "good": 1.2,
        "great": 1.6,
        "best": 2.0,
        "love": 1.8,
        "nice": 1.3,
        "awesome": 1.8,
        "congrats": 1.5,
        "cool": 1.3,
        "perfect": 2.0,
        "fast": 1.1,
    }

    # Negative words with base polarity weights (-1.0 to -2.5)
    NEGATIVE_LEXICON: Dict[str, float] = {
        # Core Negative
        "អាក្រក់": -1.8,
        "អាក្រក់ណាស់": -2.4,
        "ស្អប់": -2.0,
        "ស្អប់ណាស់": -2.5,
        "ខឹង": -1.8,
        "ខកចិត្ត": -2.0,
        "ខកចិត្តណាស់": -2.4,
        "មិនល្អ": -1.8,
        "មិនចូលចិត្ត": -1.6,
        "មិនពេញចិត្ត": -1.7,
        "មិនសម": -1.3,
        "មិនកើត": -1.6,
        "មិនត្រូវ": -1.2,
        "អន់": -1.6,
        "អន់ណាស់": -2.2,
        "ខាត": -1.4,
        "ខាតលុយ": -2.0,
        "ខាតពេល": -1.8,
        "ស្អុយ": -1.6,
        "ឆ្កួត": -1.8,
        "កុហក": -2.2,
        "បោក": -2.2,
        "បោកប្រាស់": -2.5,
        "យឺត": -1.3,
        "យឺតណាស់": -1.8,
        "ថ្លៃ": -1.2,
        "ថ្លៃណាស់": -1.6,
        "ថ្លៃហួសហេតុ": -2.2,
        "គ្មានប្រយោជន៍": -2.0,
        "គ្មានគុណភាព": -2.0,
        "គ្មានការទទួលខុសត្រូវ": -2.3,
        "ពិបាក": -1.2,
        "ពិបាកចិត្ត": -1.7,
        "ពិបាកប្រើ": -1.6,
        "អត់បានការ": -2.0,
        "អត់កើត": -1.5,
        "អត់ល្អ": -1.6,
        "បរាជ័យ": -1.8,
        "ខូច": -1.5,
        "ខូចចិត្ត": -1.9,
        "រិះគន់": -1.3,
        "ឈឺចាប់": -1.7,
        "លួចចម្លង": -2.0,
        "គួរអោយខ្លាច": -1.6,
        "ធុញ": -1.4,
        "ធុញណាស់": -1.8,
        "ខ្ជិល": -1.3,
        "បាត់បង់": -1.5,
        "គ្រោះថ្នាក់": -1.8,
        
        # English negative terms
        "bad": -1.4,
        "worst": -2.2,
        "hate": -2.0,
        "scam": -2.5,
        "fake": -2.2,
        "slow": -1.3,
        "terrible": -2.0,
        "expensive": -1.3,
        "poor": -1.4,
        "useless": -2.0,
    }

    # Negators that flip sentiment of the following word
    NEGATORS: Set[str] = {
        "មិន", "អត់", "គ្មាន", "ពុំ", "កុំ", "មិនមែន", "មិនដែល"
    }

    # Intensifiers that multiply the polarity
    INTENSIFIERS: Dict[str, float] = {
        "ណាស់": 1.4,
        "ខ្លាំង": 1.4,
        "ខ្លាំងណាស់": 1.7,
        "ពិតជា": 1.3,
        "បំផុត": 1.5,
        "លើសលប់": 1.5,
        "មែនទែន": 1.3,
        "very": 1.4,
        "really": 1.3,
        "so": 1.2,
    }

    # Emoji Polarity Dictionary
    EMOJI_SENTIMENT: Dict[str, float] = {
        # Positive Emojis
        "❤️": 1.6, "💖": 1.6, "💕": 1.5, "😍": 1.8, "🥰": 1.8,
        "👍": 1.4, "👏": 1.5, "🔥": 1.5, "🎉": 1.6, "🎊": 1.5,
        "💯": 1.8, "😊": 1.3, "😁": 1.3, "😃": 1.3, "😄": 1.3,
        "🌟": 1.4, "⭐": 1.3, "✨": 1.4, "🥳": 1.6, "👌": 1.3,
        "🙏": 1.2, "💪": 1.4, "💐": 1.3, "🌸": 1.2, "🏆": 1.5,
        # Negative Emojis
        "😡": -1.8, "😠": -1.6, "🤬": -2.2, "👎": -1.6, "💔": -1.8,
        "😢": -1.5, "😭": -1.6, "🤮": -2.0, "🤢": -1.7, "💩": -1.8,
        "😤": -1.5, "🤦": -1.4, "🤦‍♂️": -1.4, "🤦‍♀️": -1.4, "🙄": -1.3,
        "😞": -1.4, "😔": -1.3, "😒": -1.4, "👿": -1.8, "💀": -1.2,
        # Neutral Emojis
        "🤔": 0.0, "😐": 0.0, "😑": 0.0, "🧐": 0.0, "👀": 0.0,
        "🤷": 0.0, "🤷‍♂️": 0.0, "🤷‍♀️": 0.0, "ℹ️": 0.0, "📌": 0.0,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the sentiment engine.
        
        Args:
            model_path: Path to serialized scikit-learn model (.joblib).
                        If None, uses default path `models/khmer_sentiment_model.joblib`.
        """
        self.preprocessor = KhmerPreprocessor()
        
        default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.model_path = model_path or os.path.join(default_dir, "khmer_sentiment_model.joblib")
        self.ml_pipeline: Optional[Pipeline] = None
        
        # Load or train ML model if available
        self._load_or_initialize_ml_model()

    def _load_or_initialize_ml_model(self):
        """Attempts to load a trained sklearn pipeline, or trains a baseline if missing."""
        if os.path.exists(self.model_path):
            try:
                self.ml_pipeline = joblib.load(self.model_path)
                return
            except Exception:
                pass
        
        # Auto-train baseline model and persist
        try:
            self.train_baseline_model()
        except Exception:
            self.ml_pipeline = None

    def predict_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Classifies sentiment using lexicon matching, negator windowing, and emoji analysis.
        
        Returns:
            Dict containing:
            - 'sentiment': 'Positive', 'Negative', or 'Neutral'
            - 'confidence': float between 0.50 and 1.00
            - 'score': raw polarity score (positive > 0, negative < 0)
            - 'matched_terms': list of detected sentiment tokens
        """
        cleaned = self.preprocessor.clean_text(text, keep_emojis=True)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 0.50,
                "score": 0.0,
                "matched_terms": []
            }

        tokens = self.preprocessor.tokenize(cleaned)
        if not tokens:
            return {
                "sentiment": "Neutral",
                "confidence": 0.50,
                "score": 0.0,
                "matched_terms": []
            }

        total_score = 0.0
        matched_terms: List[str] = []
        negator_active = False
        negator_window = 0
        intensifier_mult = 1.0

        for i, token in enumerate(tokens):
            lower_token = token.lower()
            
            # Check for intensifiers
            if token in self.INTENSIFIERS or lower_token in self.INTENSIFIERS:
                mult = self.INTENSIFIERS.get(token, self.INTENSIFIERS.get(lower_token, 1.0))
                intensifier_mult = mult
                continue

            # Check for negators
            if token in self.NEGATORS:
                negator_active = True
                negator_window = 2  # negates next 2 tokens
                continue

            term_polarity = 0.0
            found = False

            # Check Emoji sentiment
            if token in self.EMOJI_SENTIMENT:
                term_polarity = self.EMOJI_SENTIMENT[token]
                found = True
            # Check Positive Lexicon
            elif token in self.POSITIVE_LEXICON or lower_token in self.POSITIVE_LEXICON:
                term_polarity = self.POSITIVE_LEXICON.get(token, self.POSITIVE_LEXICON.get(lower_token, 0.0))
                found = True
            # Check Negative Lexicon
            elif token in self.NEGATIVE_LEXICON or lower_token in self.NEGATIVE_LEXICON:
                term_polarity = self.NEGATIVE_LEXICON.get(token, self.NEGATIVE_LEXICON.get(lower_token, 0.0))
                found = True

            if found:
                # Apply intensifier
                term_polarity *= intensifier_mult
                intensifier_mult = 1.0  # Reset intensifier

                # Apply negation flip
                if negator_active:
                    # Invert polarity
                    term_polarity = -term_polarity * 0.9
                    matched_terms.append(f"[Negated]{token}")
                else:
                    matched_terms.append(token)

                total_score += term_polarity

            # Decrement negator window
            if negator_active:
                negator_window -= 1
                if negator_window <= 0:
                    negator_active = False

        # Classification decision based on cumulative polarity score
        threshold = 0.35
        if total_score > threshold:
            sentiment = "Positive"
        elif total_score < -threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Confidence calculation
        # Uses sigmoid-like function based on absolute polarity and match count
        abs_score = abs(total_score)
        match_count = len(matched_terms)
        
        if sentiment == "Neutral":
            if match_count == 0:
                confidence = 0.65  # Clean neutral
            else:
                confidence = max(0.50, 0.70 - (abs_score * 0.2))  # Ambivalent/conflicted neutral
        else:
            # Scaled between 0.60 and 0.99
            confidence = min(0.99, 0.60 + (1.0 - math.exp(-0.7 * abs_score)) * 0.38)

        return {
            "sentiment": sentiment,
            "confidence": round(float(confidence), 2),
            "score": round(float(total_score), 2),
            "matched_terms": matched_terms
        }

    def predict_ml(self, text: str) -> Dict[str, Any]:
        """
        Classifies sentiment using the scikit-learn TF-IDF model.
        Falls back to rule-based if ML pipeline is unavailable.
        """
        if not self.ml_pipeline:
            return self.predict_rule_based(text)

        cleaned_token_str = self.preprocessor.get_tokenized_string(text)
        if not cleaned_token_str.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.50,
                "score": 0.0,
                "matched_terms": []
            }

        try:
            probas = self.ml_pipeline.predict_proba([cleaned_token_str])[0]
            classes = self.ml_pipeline.classes_
            
            # Map probabilities
            proba_dict = dict(zip(classes, probas))
            pred_sentiment = self.ml_pipeline.predict([cleaned_token_str])[0]
            confidence = float(proba_dict.get(pred_sentiment, 0.50))

            # Approximate polarity score from probabilities
            pos_p = proba_dict.get("Positive", 0.0)
            neg_p = proba_dict.get("Negative", 0.0)
            score = (pos_p - neg_p) * 2.0

            return {
                "sentiment": str(pred_sentiment),
                "confidence": round(confidence, 2),
                "score": round(score, 2),
                "matched_terms": []
            }
        except Exception:
            return self.predict_rule_based(text)

    def predict_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Combines Rule-based Lexical matching and ML Classifier predictions.
        Weights clear lexical matches highly while leveraging statistical ML
        for complex or subtle context.
        """
        rule_res = self.predict_rule_based(text)
        
        # If no ML model exists, use rule-based
        if not self.ml_pipeline:
            return rule_res

        ml_res = self.predict_ml(text)

        # If rule-based has high confidence and strong matches, prioritize it
        if len(rule_res["matched_terms"]) >= 2 and rule_res["confidence"] >= 0.85:
            return rule_res

        # If both agree, boost confidence
        if rule_res["sentiment"] == ml_res["sentiment"]:
            boosted_conf = min(0.99, max(rule_res["confidence"], ml_res["confidence"]) + 0.05)
            return {
                "sentiment": rule_res["sentiment"],
                "confidence": round(boosted_conf, 2),
                "score": rule_res["score"],
                "matched_terms": rule_res["matched_terms"]
            }

        # If disagreement, weigh by confidence
        if rule_res["confidence"] > ml_res["confidence"] + 0.15:
            return rule_res
        elif ml_res["confidence"] > rule_res["confidence"] + 0.15:
            return ml_res
        else:
            # If close tie between opposing polarities, designate as Neutral or rule-based
            if rule_res["sentiment"] != "Neutral" and ml_res["sentiment"] == "Neutral":
                return rule_res
            return ml_res

    def analyze_comment(self, text: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Analyze a single comment text using chosen mode.
        Modes: 'hybrid', 'lexicon', 'ml'.
        """
        if mode == "lexicon":
            return self.predict_rule_based(text)
        elif mode == "ml":
            return self.predict_ml(text)
        else:
            return self.predict_hybrid(text)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "comment",
        mode: str = "hybrid",
        include_moderation: bool = True
    ) -> pd.DataFrame:
        """
        Runs batch sentiment analysis and optional content moderation on a pandas DataFrame.
        Adds sentiment columns: 'Cleaned_Tokens', 'Sentiment', 'Confidence', 'Sentiment_Score', 'Matched_Terms'.
        Adds moderation columns if include_moderation=True:
        'Is_Flagged', 'Toxicity_Score', 'Severity', 'Violation_Category', 'Flagged_Keywords', 'Highlighted_Comment', 'Moderation_Status'.
        """
        if df.empty or text_column not in df.columns:
            return df

        result_df = df.copy()
        
        sentiments = []
        confidences = []
        scores = []
        tokens_list = []
        matched_terms_list = []

        for text in result_df[text_column]:
            res = self.analyze_comment(str(text), mode=mode)
            token_str = self.preprocessor.get_tokenized_string(str(text))
            
            sentiments.append(res["sentiment"])
            confidences.append(res["confidence"])
            scores.append(res["score"])
            tokens_list.append(token_str)
            matched_terms_list.append(", ".join(res["matched_terms"]))

        result_df["Cleaned_Tokens"] = tokens_list
        result_df["Sentiment"] = sentiments
        result_df["Confidence"] = confidences
        result_df["Sentiment_Score"] = scores
        result_df["Matched_Terms"] = matched_terms_list

        if include_moderation:
            from moderation.toxicity_detector import KhmerToxicityDetector
            detector = KhmerToxicityDetector()
            result_df = detector.scan_dataframe(
                result_df,
                text_col=text_column,
                sentiment_score_col="Sentiment_Score",
                sentiment_label_col="Sentiment"
            )

        return result_df


    def train_baseline_model(self) -> Pipeline:
        """
        Trains a TF-IDF + LogisticRegression baseline model on a curated
        authentic Khmer sentiment training set and saves it to disk.
        """
        from .sample_data import TRAINING_DATA

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        texts = [self.preprocessor.get_tokenized_string(item[0]) for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=42))
        ])

        pipeline.fit(texts, labels)
        self.ml_pipeline = pipeline

        # Save to disk
        joblib.dump(pipeline, self.model_path)
        return pipeline
