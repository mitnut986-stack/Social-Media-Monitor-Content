"""
Khmer Text Preprocessor
Provides text cleaning, normalization, Unicode filtering, and tokenization (word segmentation)
tailored for Khmer social media comments (Facebook & YouTube).
"""

from __future__ import annotations
import re
from typing import List, Set, Optional


class KhmerPreprocessor:
    """
    Cleans and tokenizes Khmer social media text.
    Maintains Khmer Unicode ranges (\u1780-\u17FF), emoji sentiment signals,
    and cleans URLs, handles, and disruptive markup.
    """

    # Khmer Unicode ranges
    # Consonants & Independent Vowels: \u1780-\u17B3
    # Dependent Vowels: \u17B6-\u17C5
    # Diacritics & Signs: \u17C6-\u17D3
    # Punctuation: \u17D4-\u17DA
    # Riel Currency & Symbols: \u17DB, \u17DC-\u17DF
    # Khmer Digits: \u17E0-\u17E9
    # Khmer Divination Lore / Astrological: \u17F0-\u17F9
    KHMER_CHAR_PATTERN = r"[\u1780-\u17FF]"
    
    # Khmer Syllable Regex
    # An onset consonant/independent vowel, followed by optional Coeng (\u17D2) subscript clusters,
    # followed by optional vowel signs, diacritics, and repetition signs.
    KHMER_SYLLABLE_REGEX = re.compile(
        r"(?:[\u1780-\u17B3](?:\u17D2[\u1780-\u17B3])*(?:[\u17B6-\u17D3\u17D7-\u17D9])*)|"
        r"(?:[\u17E0-\u17E9]+)|"
        r"(?:[a-zA-Z0-9]+)|"
        r"(?:[^\s\u1780-\u17FFa-zA-Z0-9]+)"
    )

    # Emoji regex pattern
    EMOJI_PATTERN = re.compile(
        r"[\U00010000-\U0010ffff"
        r"\u2600-\u26FF\u2700-\u27BF"
        r"\u2300-\u23FF\u2B50\u200D\uFE0F]+",
        flags=re.UNICODE
    )

    # Core multi-syllable vocabulary for Khmer compound recognition & maximum matching
    DEFAULT_COMPOUND_WORDS: Set[str] = {
        # Positive compounds
        "អរគុណ", "អរគុណច្រើន", "ស្រលាញ់", "ចូលចិត្ត", "ពេញចិត្ត", "សប្បាយ", "សប្បាយរីករាយ",
        "អស្ចារ្យ", "ត្រឹមត្រូវ", "គាំទ្រ", "កោតសរសើរ", "ជោគជ័យ", "មានប្រយោជន៍", "អភិវឌ្ឍន៍",
        "រួសរាយ", "ស្មោះត្រង់", "ពូកែ", "ឡូយណាស់", "ល្អណាស់", "ល្អខ្លាំង", "ល្អឥតខ្ចោះ",
        "ពេញបេះដូង", "គំរូល្អ", "គុណភាពល្អ", "សេវាកម្មល្អ", "តម្លៃសមរម្យ", "លឿនរហ័ស",
        "គួរអោយស្រលាញ់", "គួរឲ្យស្រលាញ់", "ជួយចែករំលែក", "ផ្ដល់កម្លាំងចិត្ត",
        
        # Negative compounds
        "មិនល្អ", "មិនចូលចិត្ត", "មិនពេញចិត្ត", "មិនសម", "មិនត្រូវ", "មិនច្បាស់", "មិនកើត",
        "អត់ល្អ", "អត់ចូលចិត្ត", "អត់កើត", "អត់បានការ", "គ្មានប្រយោជន៍", "គ្មានគុណភាព",
        "គ្មានការទទួលខុសត្រូវ", "បោកប្រាស់", "កុហក", "ខកចិត្ត", "ខូចចិត្ត", "ឈឺចាប់",
        "ពិបាកចិត្ត", "ពិបាកប្រើ", "យឺតណាស់", "យឺតខ្លាំង", "ថ្លៃហួសហេតុ", "ខាតលុយ",
        "ខាតពេល", "រិះគន់", "ស្អប់ណាស់", "អាក្រក់ណាស់", "អន់ណាស់", "លួចចម្លង",
        "សេវាអន់", "សេវាអាក្រក់", "អាកប្បកិរិយាអន់",
        
        # Negators & Modifiers
        "មិនដែល", "មិនដែលឃើញ", "មិនទាន់", "ពុំដែល", "កុំធ្វើ", "កុំជឿ",
        
        # Social Media / Common terms
        "វីដេអូ", "ផេក", "ប៉ុស្តិ៍", "ចែករំលែក", "ខមមិន", "មេដៃ", "បងប្អូន", "ប្រិយមិត្ត",
        "ព័ត៌មាន", "ទស្សនា", "ទស្សនិកជន", "អតិថិជន", "កម្មវិធី", "ទូរស័ព្ទ", "កម្ពុជា"
    }

    def __init__(self, custom_words: Optional[Set[str]] = None):
        """Initialize preprocessor with base vocabulary and compiled regexes."""
        self.compound_words = set(self.DEFAULT_COMPOUND_WORDS)
        if custom_words:
            self.compound_words.update(custom_words)
        
        # Sort compounds by length descending for longest-matching
        sorted_compounds = sorted(self.compound_words, key=len, reverse=True)
        # Escape for regex safely
        escaped_compounds = [re.escape(w) for w in sorted_compounds]
        self.compound_pattern = re.compile(f"({'|'.join(escaped_compounds)})") if escaped_compounds else None

        # Check for optional khmer_nltk
        self._has_khmer_nltk = False
        try:
            from khmer_nltk import word_tokenize  # type: ignore
            self._khmer_nltk_tokenize = word_tokenize
            self._has_khmer_nltk = True
        except ImportError:
            self._khmer_nltk_tokenize = None

    def clean_text(self, text: str, keep_emojis: bool = True) -> str:
        """
        Cleans raw social media text:
        - Removes URLs and hyperlinks
        - Removes user mentions (@username)
        - Removes hashtags symbols while preserving words
        - Normalizes multiple spaces, tabs, and newlines
        - Strips zero-width space characters (\u200B, \u200C, \u200D, \uFEFF)
        - Optionally preserves or filters emojis
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # Remove User Mentions (@someone)
        text = re.sub(r"@[\w_]+", " ", text)

        # Remove hashtag '#' sign but keep word
        text = re.sub(r"#([^\s#]+)", r"\1", text)

        # Remove zero-width spaces / joiners / byte order marks
        text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

        # Normalize repeated punctuation like ???, !!!, ..., ---
        text = re.sub(r"([!?.,\-~])\1+", r"\1", text)

        # Remove unwanted control characters while keeping Khmer, alphanumeric, and emojis
        if not keep_emojis:
            text = self.EMOJI_PATTERN.sub(" ", text)

        # Clean multiple spaces and whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def contains_khmer(self, text: str) -> bool:
        """Check if the text contains any Khmer characters."""
        return bool(re.search(self.KHMER_CHAR_PATTERN, text))

    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis present in the text."""
        return self.EMOJI_PATTERN.findall(text)

    def tokenize(self, text: str, use_nltk_if_available: bool = True) -> List[str]:
        """
        Tokenizes Khmer text into segmented words/tokens.
        Uses compound dictionary matching + syllable fallback segmenter,
        or khmer_nltk if available and selected.
        
        Args:
            text: Input text (will be cleaned first).
            use_nltk_if_available: Whether to use khmer_nltk if installed.
            
        Returns:
            List of string tokens.
        """
        cleaned = self.clean_text(text, keep_emojis=True)
        if not cleaned:
            return []

        # If khmer_nltk is available and preferred
        if use_nltk_if_available and self._has_khmer_nltk and self._khmer_nltk_tokenize:
            try:
                tokens = self._khmer_nltk_tokenize(cleaned)
                return [t.strip() for t in tokens if t.strip()]
            except Exception:
                pass  # Fall back to built-in segmenter

        # High-performance built-in segmentation
        # Step 1: Split into whitespace/punctuation chunks
        chunks = cleaned.split()
        all_tokens: List[str] = []

        for chunk in chunks:
            # Check for pure emojis
            emojis = self.extract_emojis(chunk)
            if emojis and len("".join(emojis)) == len(chunk):
                # Split multi-emojis
                for char in chunk:
                    all_tokens.append(char)
                continue

            # Process Khmer / mixed chunk
            all_tokens.extend(self._segment_chunk(chunk))

        return [t for t in all_tokens if t and not t.isspace()]

    def _segment_chunk(self, chunk: str) -> List[str]:
        """Segments an individual non-spaced string into word/syllable tokens."""
        tokens: List[str] = []

        # Step A: If compound pattern exists, split by compound dictionary first
        if self.compound_pattern:
            parts = self.compound_pattern.split(chunk)
        else:
            parts = [chunk]

        for part in parts:
            if not part:
                continue
            # If this part is one of our compounds, keep it as single token!
            if part in self.compound_words:
                tokens.append(part)
            elif self.contains_khmer(part):
                # Segment Khmer characters by syllable regex
                syllables = self.KHMER_SYLLABLE_REGEX.findall(part)
                # Filter empty syllables
                tokens.extend([s for s in syllables if s.strip()])
            else:
                # Non-Khmer (English word, number, emoji, symbol)
                tokens.append(part)

        return tokens

    def get_tokenized_string(self, text: str) -> str:
        """Returns space-separated tokens string (convenient for TF-IDF / vectorizers)."""
        tokens = self.tokenize(text)
        return " ".join(tokens)
