# 📰 Fake News Detection System

An AI-powered **news and claim verification system** built with Python and Streamlit. The application analyzes a user's news claim, searches for related trusted reporting using NewsAPI, and uses NLP techniques to find supporting or contradictory evidence.

## 🚀 Features

* 🔎 News claim verification
* 📰 Trusted news retrieval using NewsAPI
* 🧠 NLP-based entity extraction using spaCy
* 📊 TF-IDF based text similarity
* 🤖 Natural Language Inference using RoBERTa-large-MNLI
* ⚠️ Contradiction detection
* 📑 Displays related trusted articles
* 🌐 Interactive Streamlit web interface
* 💾 Caches models and API results for better performance

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **NLTK**
* **spaCy**
* **Scikit-learn**
* **Transformers**
* **PyTorch**
* **Requests**
* **NewsAPI**
* **RoBERTa-large-MNLI**

## 🔄 How It Works

The system follows a multi-step verification pipeline:

```text
User News Claim
       ↓
Text Processing
       ↓
Entity Extraction using spaCy
       ↓
Search Related News using NewsAPI
       ↓
TF-IDF Similarity
       ↓
Relevant Articles
       ↓
RoBERTa NLI Analysis
       ↓
Supporting / Contradictory / Inconclusive Evidence
       ↓
Final Verification Result
```

## 🧠 NLP Pipeline

### 1. Entity Extraction

spaCy is used to identify important entities from the input text, including:

* PERSON
* ORGANIZATION
* GPE / Location

These entities help construct search queries for retrieving related news.

### 2. News Retrieval

The application uses **NewsAPI** to search for related articles from available news sources.

The retrieved articles are used as external evidence for verification.

### 3. TF-IDF Similarity

TF-IDF converts the user claim and retrieved articles into numerical representations.

Cosine similarity is then used to estimate how closely a retrieved article is related to the user's claim.

### 4. Natural Language Inference

The project uses:

**RoBERTa-large-MNLI**

to analyze the relationship between the user claim and retrieved news content.

The NLI model produces scores related to:

* Entailment
* Contradiction
* Neutral

These scores are used as additional evidence for the final result.

## 📊 Verification Results

The system can produce results such as:

### 🟢 Likely Real

Related reporting provides strong supporting evidence for the claim.

### 🔴 Probably Fake

Strong contradictory evidence is detected between the claim and retrieved reporting.

### 🟡 Needs Review

There is not enough reliable or sufficiently related evidence to confidently verify the claim.

> The system should be treated as a claim-verification assistance tool rather than a guaranteed fake-news detector. A "Needs Review" result does not mean that a claim is false.

## 📁 Project Structure

```text
Fake-News-Detection-System/
│
├── FND_demo(1).py
├── test_news_api.py
├── requirements.txt
├── .gitignore
└── .ipynb
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/piyushkumarsingh2511-art/Fake-News-Detection-System.git
```

Move into the project directory:

```bash
cd Fake-News-Detection-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 API Key Configuration

The project uses NewsAPI.

Create a `.env` file in the project directory:

```env
NEWS_API_KEY=your_api_key_here
```

The application loads the API key using environment variables.

**Do not upload your `.env` file or expose your API key publicly.**

## ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run "FND_demo(1).py"
```

The application will open in your browser.

## 🧪 Example

### Input

```text
The Eiffel Tower is located in Paris, France.
```

The application searches for related reporting and evaluates the available evidence using text similarity and NLI.

The result may be:

```text
🟢 Likely Real
```

depending on the retrieved evidence.

## ⚠️ Limitations

* NewsAPI results depend on available news coverage.
* Historical or general factual claims may not have suitable current-news evidence.
* Lack of retrieved evidence does not automatically mean that a claim is fake.
* TF-IDF measures textual similarity and does not understand facts by itself.
* NLI predictions depend on the quality and relevance of the retrieved evidence.
* The system requires an internet connection for NewsAPI-based verification.

## 🔮 Future Improvements

* Add a dedicated factual knowledge base
* Improve evidence ranking
* Add better claim-type detection
* Use semantic embeddings instead of only TF-IDF
* Add source credibility scoring
* Improve historical fact verification
* Add multilingual news verification
* Add a more advanced retrieval pipeline using vector databases
* Improve explainability of verification results

## 👨‍💻 Author

**Piyush Kumar Singh**

B.Tech Computer Science & Engineering Student

Interested in:

* Machine Learning
* Artificial Intelligence
* Data Science
* Software Engineering
* NLP

## 📌 Project Purpose

This project was developed to explore how **NLP, information retrieval, text similarity, and Natural Language Inference** can be combined to assist with news and claim verification.
