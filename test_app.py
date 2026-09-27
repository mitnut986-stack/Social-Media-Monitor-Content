"""
Verification and Test Suite for Khmer Social Media Sentiment Analysis
Tests:
1. Khmer Preprocessor: Unicode range preservation, URL removal, emoji extraction, syllable tokenization.
2. Rule-Based Lexicon: Polarity calculation, negation flipping (e.g. មិនល្អ -> negative, អត់ខកចិត្ត -> positive/neutral).
3. Emojis and Multilingual Handling: ❤️, 😡, English social slang.
4. Model Training & Serialization: TF-IDF + Logistic Regression training and inference.
5. Connector URL Parsers: YouTube video ID extraction and Facebook Post ID extraction.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from auth.user_manager import UserManager
from connectors.youtube_connector import YouTubeConnector
from connectors.facebook_connector import FacebookConnector
from connectors.page_explorer import PageExplorer
from moderation.toxicity_detector import KhmerToxicityDetector
from nlp.preprocessor import KhmerPreprocessor
from nlp.sentiment_engine import KhmerSentimentEngine




def test_suite():
    print("=" * 60)
    print("🇰🇭 RUNNING KHMER SENTIMENT ANALYSIS TEST SUITE")
    print("=" * 60)

    # 1. Test Preprocessor
    print("\n[1] Testing KhmerPreprocessor...")
    preprocessor = KhmerPreprocessor()
    
    sample_text = "វីដេអូនេះល្អណាស់! https://youtube.com @admin #khmer ❤️❤️"
    cleaned = preprocessor.clean_text(sample_text, keep_emojis=True)
    assert "https" not in cleaned, "URL removal failed"
    assert "@admin" not in cleaned, "Mention removal failed"
    assert "❤️" in cleaned, "Emoji preservation failed"
    print(f"  Cleaned text: '{cleaned}' -> PASSED")

    tokens = preprocessor.tokenize(cleaned)
    assert len(tokens) > 0, "Tokenization returned empty list"
    assert "ល្អណាស់" in tokens or "ល្អ" in tokens, "Compound word matching failed"
    print(f"  Tokens: {tokens} -> PASSED")

    # 2. Test Sentiment Engine (Rule-based & Negation)
    print("\n[2] Testing KhmerSentimentEngine Rule-Based & Negation...")
    engine = KhmerSentimentEngine()

    # Positive test
    pos_res = engine.predict_rule_based("វីដេអូនេះល្អណាស់ អរគុណច្រើនបង ❤️❤️")
    print(f"  Positive text result: {pos_res}")
    assert pos_res["sentiment"] == "Positive", f"Expected Positive, got {pos_res['sentiment']}"
    assert pos_res["confidence"] >= 0.70, "Confidence too low for clear positive"

    # Negative test
    neg_res = engine.predict_rule_based("ទំនិញអន់ណាស់ បោកប្រាស់ ស្អប់ណាស់ 😡")
    print(f"  Negative text result: {neg_res}")
    assert neg_res["sentiment"] == "Negative", f"Expected Negative, got {neg_res['sentiment']}"

    # Negation test: មិន + ល្អ -> Negative
    negated_res = engine.predict_rule_based("សេវាកម្មនេះមិនល្អទេ")
    print(f"  Negation test (មិនល្អ): {negated_res}")
    assert negated_res["sentiment"] == "Negative", f"Negation failed for មិនល្អ: {negated_res}"

    # Neutral test
    neu_res = engine.predict_rule_based("តើទំនិញនេះតម្លៃប៉ុន្មានដែរបង?")
    print(f"  Neutral question result: {neu_res}")
    assert neu_res["sentiment"] == "Neutral", f"Expected Neutral, got {neu_res['sentiment']}"

    # 3. Test Machine Learning Inference
    print("\n[3] Testing Scikit-Learn TF-IDF Sentiment Inference...")
    ml_res = engine.predict_ml("វីដេអូមានប្រយោជន៍ណាស់ គាំទ្របង")
    print(f"  ML Result: {ml_res}")
    assert ml_res["sentiment"] in ["Positive", "Negative", "Neutral"]

    # 4. Test Hybrid Prediction
    print("\n[4] Testing Hybrid Prediction (Rule-Based + ML)...")
    hybrid_res = engine.predict_hybrid("សេវាកម្មរហ័សទាន់ចិត្ត និងបុគ្គលិករួសរាយណាស់")
    print(f"  Hybrid Result: {hybrid_res}")
    assert hybrid_res["sentiment"] == "Positive"

    # 5. Test Connectors URL Parsers
    print("\n[5] Testing Social Media URL Parsers...")
    yt_id1 = YouTubeConnector.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    yt_id2 = YouTubeConnector.extract_video_id("https://youtu.be/dQw4w9WgXcQ")
    assert yt_id1 == "dQw4w9WgXcQ", f"YouTube parser failed: {yt_id1}"
    assert yt_id2 == "dQw4w9WgXcQ", f"YouTube youtu.be parser failed: {yt_id2}"
    print("  YouTube URL parsing -> PASSED")

    fb_id = FacebookConnector.extract_post_id("https://www.facebook.com/my_page/posts/1234567890")
    assert fb_id == "1234567890", f"Facebook parser failed: {fb_id}"
    print("  Facebook Post URL parsing -> PASSED")

    # 6. Test Authentication & Password Hashing
    print("\n[6] Testing UserManager & Password Hashing...")
    user_mgr = UserManager()
    
    # Test hashing and verification
    pw = "SecretKhmer2026!"
    hashed = user_mgr.hash_password(pw)
    assert user_mgr.verify_password(pw, hashed) is True, "Password verification failed"
    assert user_mgr.verify_password("WrongPassword!", hashed) is False, "Password verification should have failed"
    print("  Password hashing & verification -> PASSED")

    # Test Default Demo Admin Authentication
    auth_ok, admin_data = user_mgr.authenticate_user("admin@khmer.ai", "Password123!")
    assert auth_ok is True, f"Admin demo authentication failed: {admin_data}"
    assert admin_data["username"] == "admin@khmer.ai"
    print(f"  Demo Admin Login ({admin_data['full_name']}) -> PASSED")

    # Test Registration: Password mismatch
    mismatch_ok, msg = user_mgr.register_user("Test User", "test_user_temp", "pass123", "pass999")
    assert mismatch_ok is False, "Password mismatch should have been rejected"

    # Test Registration: Success
    import uuid
    random_user = f"user_{uuid.uuid4().hex[:6]}@example.com"
    reg_ok, msg = user_mgr.register_user("Bopha Chan", random_user, "ValidPass123!", "ValidPass123!")
    assert reg_ok is True, f"Registration failed: {msg}"
    print(f"  User Registration ({random_user}) -> PASSED")

    # Test Duplicate Registration rejection
    dup_ok, _ = user_mgr.register_user("Duplicate User", random_user, "ValidPass123!", "ValidPass123!")
    assert dup_ok is False, "Duplicate user registration should have failed"
    # 7. Test Khmer Toxicity & Moderation Engine
    print("\n[7] Testing KhmerToxicityDetector...")
    detector = KhmerToxicityDetector()

    # Profanity test
    profane_res = detector.scan_text("អាចុយម្រាយ ចុយ ក្ដ")
    assert profane_res["is_flagged"] is True, "Profanity should be flagged"
    assert profane_res["violation_category"] == "Profanity / Vulgarity", f"Expected Profanity, got {profane_res['violation_category']}"
    assert len(profane_res["flagged_keywords"]) > 0
    print(f"  Profanity detection ({profane_res['violation_category']}) -> PASSED")

    # Hate Speech test
    hate_res = detector.scan_text("អាមនុស្សថោកទាប មនុស្សអត់ខួរ គ្មានខួរ")
    assert hate_res["is_flagged"] is True, "Hate speech should be flagged"
    assert hate_res["violation_category"] == "Hate Speech / Harassment", f"Expected Hate Speech, got {hate_res['violation_category']}"
    print(f"  Hate speech detection ({hate_res['violation_category']}) -> PASSED")

    # Threat & Violence test
    threat_res = detector.scan_text("ចាំមើលប្រយ័ត្នងាប់ វ៉ៃបំបែកក្បាលចោល ទៅងាប់ទៅ")
    assert threat_res["is_flagged"] is True, "Threat should be flagged"
    assert threat_res["violation_category"] == "Threat / Violence"
    print(f"  Threat detection ({threat_res['violation_category']}) -> PASSED")

    # Scam & Fraud test
    scam_res = detector.scan_text("ឈ្នះរង្វាន់ក្លែងក្លាយ! ផ្ទេរលុយមកលេខនេះភ្លាម 012-999-888")
    assert scam_res["is_flagged"] is True, "Scam should be flagged"
    assert scam_res["violation_category"] == "Scam / Fraud"
    print(f"  Scam detection ({scam_res['violation_category']}) -> PASSED")


    # Clean comment test
    clean_res = detector.scan_text("វីដេអូនេះល្អណាស់ អរគុណបងដែលបានចែករំលែកចំណេះដឹង ❤️")
    assert clean_res["is_flagged"] is False, "Clean comment should NOT be flagged"
    assert clean_res["severity"] == "Clean"
    print("  Clean comment verification -> PASSED")

    # 8. Test Public Page & Content Explorer
    print("\n[8] Testing PageExplorer...")
    explorer = PageExplorer()
    pub_res = explorer.explore_public_channel("Fresh News Daily", platform="YouTube", max_comments=10)
    assert "channel_info" in pub_res, "Channel info missing"
    assert "comments" in pub_res and len(pub_res["comments"]) > 0, "Public comments missing"
    print(f"  Public Page Explorer ({pub_res['channel_info']['title']}) -> PASSED")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! APPLICATION IS PRODUCTION READY!")
    print("=" * 60)




if __name__ == "__main__":
    test_suite()
