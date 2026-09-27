# 🛡️ Khmer Social Media Monitor & Content Moderation Dashboard

A production-grade Web Application that provides real-time **AI Sentiment Intelligence**, **Automated Content Moderation**, and **Public Brand Tracking** for **YouTube Channels** and **Facebook Pages**.

Built with **Streamlit**, **Python 3.10+**, **Plotly**, **Scikit-learn**, **Google API Client**, and **Meta Graph API v19.0**.

---

## 🌟 Key Features

1. **Dual Monitoring Modes**:
   - **Mode A: 🛡️ My Channel / Page Monitor (Authenticated)**:
     - Connect own Facebook Page (via Page Access Token) and YouTube Channel (via YouTube Data API v3).
     - Ingest live comments and automatically detect toxicity (Profanity, Harassment, Hate Speech, Threats, Scams).
     - **🚨 Red Alert Moderation Queue**: Real-time alert banner, review triage workflow (Mark Reviewed / Re-flag), and actionable flagged CSV report export.
   - **Mode B: 🌐 Public Page & Content Explorer (External Tracking)**:
     - Track external public Facebook Pages and YouTube Channels without requiring owner privileges.
     - View channel reach, monitored posts, public reaction sentiment, and frequent Khmer discussion topics.
2. **Khmer Toxicity & Content Moderation Engine**:
   - Categorized Khmer blocklists for **Profanity / Vulgarities (`ពាក្យអាសអាភាស`)**, **Hate Speech / Harassment (`ការរើសអើង / បៀតបៀន`)**, **Threats & Violence (`ការគំរាមកំហែង`)**, and **Scams & Fraud (`ការបោកប្រាស់`)**.
   - Computes **Toxicity Score** (0.0 to 1.0) and assigns severity levels (Clean, Low Risk, Moderate Risk, High Risk / Severe).
   - Generates an active **Sensitive Keyword Hotlist** and visual highlights inside comments.
3. **User Authentication & Session Management**:
   - Clean landing page with **Sign In** and **Create New Account** tabs.
   - Secure password hashing using **`bcrypt`** (with fallback to NIST-standard **`PBKDF2-HMAC-SHA256`**).
   - Session state tracking with `st.session_state` and **Sign Out** button in the sidebar.
   - Pre-seeded default administrator account: `admin@khmer.ai` / `Password123!`.
4. **Khmer NLP & Sentiment Engine**:
   - Cleans noise, URLs, user mentions, while preserving the full Khmer Unicode range (`\u1780-\u17FF`), Khmer numbers (`\u17E0-\u17E9`), and punctuation signs.
   - Syllable segmentation and compound phrase matching.
   - Contextual negation windowing (`មិន`, `អត់`, `គ្មាន`, `ពុំ`) and Scikit-learn TF-IDF model.


2. **Khmer NLP Engine**:
   - Cleans noise, URLs, user mentions, while preserving the full Khmer Unicode range (`\u1780-\u17FF`), Khmer numbers (`\u17E0-\u17E9`), and punctuation signs.
   - Intelligent syllable segmentation and compound word matching tailored for the unspaced nature of Khmer writing.
   - Preserves emojis and evaluates emoji polarity.
3. **Sentiment Classification Modes**:
   - **Hybrid (Lexicon + ML)** *(Recommended)*: Combines rule-based lexical matching with statistical Machine Learning predictions.
   - **Rule-Based Lexicon**: Uses polarity weights, intensifiers (`ណាស់`, `ខ្លាំង`), and contextual negation windowing (`មិន`, `អត់`, `គ្មាន`, `ពុំ`).
   - **Machine Learning (TF-IDF + Logistic Regression)**: Scikit-learn TF-IDF pipeline trained on authentic Khmer social media comments.
4. **Rich Interactive Analytics**:
   - Summary cards: Total Comments, Positive %, Negative %, Neutral %, and Net Sentiment Score (NSS).
   - Plotly Donut Chart with Emerald / Crimson / Slate color schemes.
   - Temporal activity bar chart and model confidence distribution.
   - Interactive data table with keyword filtering, sentiment filtering, and confidence thresholds.
   - Deep-dive comment inspector showing token breakdown and trigger words.
5. **Windows Excel-Friendly Export**:
   - Export analyzed comments to CSV with **UTF-8 with BOM (`utf-8-sig`)** so Khmer script displays cleanly in Microsoft Excel without mojibake.
   - Export to formatted JSON.
6. **Instant Demo Mode**:
   - Includes realistic Khmer social media comments for instant testing even without live API keys.

---

## 📁 Directory Structure

```text
khmer-sentiment-analysis/
│
├── app.py                      # Main Streamlit web application & dashboard
├── requirements.txt            # Project dependencies
├── README.md                   # Setup guide and documentation
│
├── connectors/                 # Social media API connectors
│   ├── __init__.py
│   ├── youtube_connector.py    # YouTube Data API v3 client with pagination
│   └── facebook_connector.py   # Meta Graph API v19.0 client
│
├── nlp/                        # Khmer NLP and Sentiment Engine
│   ├── __init__.py
│   ├── preprocessor.py         # Khmer Unicode cleaning & syllable/word tokenizer
│   ├── sentiment_engine.py     # Rule-based, ML, and Hybrid sentiment classifier
│   └── sample_data.py          # Benchmark training corpus & realistic mock data
│
└── models/                     # Machine learning models & training scripts
    ├── train_model.py          # Script to train & evaluate TF-IDF ML model
    └── khmer_sentiment_model.joblib # Serialized model (generated automatically)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher installed on your system.

### 2. Setup Virtual Environment
Open PowerShell or your terminal in the project directory:

```bash
# Navigate to the project folder
cd C:\Users\ONEXPC2\.gemini\antigravity-ide\scratch\khmer-sentiment-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train or Re-train the Machine Learning Model
The application will automatically initialize the model on first launch, or you can run:
```bash
python models/train_model.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🔑 How to Obtain API Credentials

### A. YouTube Data API v3 Key
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g. `Khmer-Sentiment-App`).
3. In the sidebar, navigate to **APIs & Services > Library**.
4. Search for **"YouTube Data API v3"** and click **Enable**.
5. Navigate to **APIs & Services > Credentials**.
6. Click **Create Credentials > API Key**.
7. Copy your API Key (starts with `AIzaSy...`) and paste it into the Streamlit sidebar.

> **Tip:** You can input either an 11-character Video ID (e.g., `dQw4w9WgXcQ`) or any full YouTube URL (e.g., `https://www.youtube.com/watch?v=...`, `https://youtu.be/...`, or YouTube Shorts URLs).

---

### B. Meta Graph API (Facebook Page Access Token)
1. Go to [Meta for Developers](https://developers.facebook.com/).
2. Log in and click **My Apps > Create App**.
3. Select **"Business"** or **"Other"** as the app type.
4. Navigate to the **Graph API Explorer** tool (`developers.facebook.com/tools/explorer`).
5. In the **User or Page** dropdown, select your **Facebook Page**.
6. Under **Permissions**, ensure the following are added:
   - `pages_show_list`
   - `pages_read_engagement`
   - `pages_manage_posts` (if accessing private/scheduled posts)
7. Click **Generate Access Token**.
8. For production use, exchange this temporary token for a **Long-Lived Page Access Token** using the Meta Token Debugger.
9. Copy your Page Access Token into the Streamlit sidebar.

---

## 🧠 Khmer NLP Architecture

### 1. Khmer Text Normalization & Syllable Segmentation
Khmer is a continuous script without inter-word spacing. `KhmerPreprocessor` uses a two-tiered tokenization approach:
1. **Compound Lexicon Matching**: Identifies multi-syllable phrases (e.g., `អរគុណច្រើន`, `គ្មានប្រយោជន៍`, `បោកប្រាស់`, `សេវាកម្មល្អ`) using longest-match regex substitution.
2. **Syllable Cluster Regex**: Deconstructs remaining Khmer text by onset consonants, subscript Coeng (`\u17D2`) clusters, dependent/independent vowels, and diacritics.

### 2. Contextual Negator Windowing
In Khmer social media, negators like `មិន` (min), `អត់` (ot), or `គ្មាន` (kmean) directly reverse the polarity of subsequent words:
- `ល្អ` (good) -> Positive (+1.5)
- `មិនល្អ` (not good) -> Negative (-1.35)
- `ខកចិត្ត` (disappointed) -> Negative (-2.0)
- `មិនដែលខកចិត្ត` (never disappointed) -> Positive (+1.8)

### 3. UTF-8 with BOM Export
Standard Excel on Windows often misinterprets raw UTF-8 CSVs containing Southeast Asian scripts. By exporting with Python's `utf-8-sig` codec, a Byte Order Mark (`\xEF\xBB\xBF`) is embedded at the start of the file, allowing Excel to immediately recognize and render Khmer Unicode characters properly.

---

## 🛡️ Error Handling
- **API Quota Exceeded (403)**: Displays clear instructions on checking Google Cloud quota.
- **Comments Disabled (403)**: Informs user that the video owner disabled comments.
- **Invalid Video/Post ID (404/400)**: Alerts user to verify the URL or ID.
- **Invalid Meta Token (190)**: Prompts user to refresh expired Facebook Page Token.
- **Empty / Non-Khmer Comments**: Gracefully classified as Neutral with low confidence.
