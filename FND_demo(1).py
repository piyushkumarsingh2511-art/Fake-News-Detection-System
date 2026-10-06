import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline
import requests
import os
import streamlit as st
import spacy
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detection Demo")
st.info("Loading AI models... Please wait.")
# -----------------------------
# Cache the Hugging Face model
# -----------------------------
@st.cache_resource
def load_nli_model():
    return pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

nli_model = load_nli_model()
_ = nli_model("Warmup headline", text_pair="Warmup trusted article")  # warm-up

# -----------------------------
# -----------------------------
# Load spaCy for entity extraction (with fallback)
# -----------------------------
import subprocess, sys

def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = get_spacy_model()

# -----------------------------
# Cache trusted articles
# -----------------------------

@st.cache_data
def fetch_trusted_news(query):
    API_KEY = "your_actual_api_key"

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    print("NewsAPI status:", response.status_code)
    print("Total results:", data.get("totalResults"))

    if response.status_code != 200:
        print("NewsAPI error:", data)
        return []

    trusted_articles = []

    for article in data.get("articles", []):
        text = (
            (article.get("title") or "")
            + " "
            + (article.get("description") or "")
        ).strip()

        if text:
            trusted_articles.append(text)

    pd.DataFrame(
        trusted_articles,
        columns=["article"]
    ).to_csv("trusted_news.csv", index=False)

    return trusted_articles

# -----------------------------
# Exact similarity checking
# -----------------------------
def check_exact_match(input_news, trusted_articles):
    for trusted in trusted_articles:
        if input_news.strip().lower() == trusted.strip().lower():
            return True
    return False

# -----------------------------
# Entity & event extraction
# -----------------------------
def extract_entities_events(text: str) -> dict:
    doc = nlp(text)
    entities = {"PERSON": set(), "ORG": set(), "GPE": set()}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].add(ent.text)

    lower = text.lower()
    events = {
        "death": any(k in lower for k in ["died", "death", "passed away", "last rite", "funeral"]),
        "appearance": any(k in lower for k in ["appearance", "spotted", "photo", "picture", "video", "interview", "birthday"]),
        "attack": any(k in lower for k in ["attack", "missile", "strike", "war", "bomb", "drone"]),
        "absurd_duration": bool(re.search(r"\b\d{4,}\s+years?\b", lower)),
    }
    return {"entities": {k: list(v) for k, v in entities.items()}, "events": events}

# -----------------------------
# Build entity state from trusted sources
# -----------------------------
def build_entity_state(trusted_articles: list) -> dict:
    state = {}
    for art in trusted_articles:
        info = extract_entities_events(art)
        persons = info["entities"].get("PERSON", [])
        has_appearance = info["events"]["appearance"]

        for p in persons:
            if p not in state:
                state[p] = {"alive_recent": False, "evidence": []}
            if has_appearance:
                state[p]["alive_recent"] = True
            state[p]["evidence"].append(art)
    return state

# -----------------------------
# Logical contradiction checks
# -----------------------------
def detect_logical_contradiction(input_text: str, entity_state: dict) -> tuple:
    info = extract_entities_events(input_text)
    persons = info["entities"].get("PERSON", [])
    events = info["events"]

    if events["absurd_duration"]:
        return True, "🔴 Fake: Impossible duration detected"

    if events["death"] and persons:
        for p in persons:
            st = entity_state.get(p)
            if st and st.get("alive_recent"):
                return True, f"🔴 Fake: Logical contradiction — {p} reported alive recently vs death claim"

    return False, ""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words="english")

def semantic_similarity(a, b):
    vecs = tfidf.fit_transform([a, b])
    return cosine_similarity(vecs[0], vecs[1])[0][0]



# ----------------------------------
# Relevence checking
# ----------------------------------
# Relevance checking
# ----------------------------------
def is_relevant(input_text, trusted_text, min_sim=0.15):
    """
    Check whether a trusted article is actually related
    to the user's claim.
    """

    sim = semantic_similarity(input_text, trusted_text)

    # Semantic similarity is the main signal.
    # Lower threshold helps short factual claims.
    return sim >= min_sim  # this threshold works well for headlines



#-----------------------------------
# -----------------------------
# NLI comparison (confidence)
# -----------------------------
# -----------------------------------
# NLI comparison
# -----------------------------------
def nli_confidence_verdict(
    input_text,
    trusted_articles,
    top_n=3,
    min_sim=0.15
):

    results = []

    # First calculate similarity for all articles
    candidates = []

    for trusted in trusted_articles:
        sim = semantic_similarity(input_text, trusted)

        if sim >= min_sim:
            candidates.append((trusted, sim))

    # Highest similarity articles first
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Only send the best articles to NLI
    candidates = candidates[:10]

    for trusted, sim in candidates:

        # Trusted article = premise
        # User claim = hypothesis
        result = nli_model(
            trusted,
            text_pair=input_text
        )

        if isinstance(result, list):
            scores = result
        else:
            scores = [result]

        contradiction_score = 0.0
        entailment_score = 0.0
        neutral_score = 0.0

        for score in scores:

            label = str(score.get("label", "")).lower()
            value = score.get("score", 0.0)

            if "contradiction" in label:
                contradiction_score = value

            elif "entailment" in label:
                entailment_score = value

            elif "neutral" in label:
                neutral_score = value

        results.append({
            "article": trusted,
            "similarity": sim,
            "contradiction": contradiction_score,
            "entailment": entailment_score,
            "neutral": neutral_score
        })

    # --------------------------------
    # No relevant article
    # --------------------------------
    if not results:
        return (
            "🟡 Needs Review (no sufficiently related trusted article found)",
            []
        )

    # --------------------------------
    # Find strongest contradiction
    # --------------------------------
    strongest_contradiction = max(
        results,
        key=lambda x: x["contradiction"]
    )

    # --------------------------------
    # Find strongest supporting evidence
    # --------------------------------
    strongest_entailment = max(
        results,
        key=lambda x: x["entailment"]
    )

    # --------------------------------
    # Fake verdict
    # --------------------------------
    if strongest_contradiction["contradiction"] >= 0.80:

        verdict = (
            f"🔴 Probably Fake: "
            f"{int(strongest_contradiction['contradiction'] * 100)}% "
            f"contradiction confidence"
        )

        top_matches = [
            (
                r["article"],
                r["similarity"],
                r["contradiction"]
            )
            for r in results
            if r["contradiction"] >= 0.50
        ][:top_n]

        return verdict, top_matches

    # --------------------------------
    # Real / supported verdict
    # --------------------------------
    if strongest_entailment["entailment"] >= 0.80:

        verdict = (
            f"🟢 Likely Real: "
            f"{int(strongest_entailment['entailment'] * 100)}% "
            f"support confidence"
        )

        top_matches = [
            (
                r["article"],
                r["similarity"],
                r["contradiction"]
            )
            for r in results
            if r["entailment"] >= 0.50
        ][:top_n]

        return verdict, top_matches

    # --------------------------------
    # Related but uncertain
    # --------------------------------
    best_similarity = max(
        r["similarity"] for r in results
    )

    verdict = (
        f"🟡 Needs Review "
        f"(related reporting found, but evidence is inconclusive)"
    )

    top_matches = [
        (
            r["article"],
            r["similarity"],
            r["contradiction"]
        )
        for r in results
        if r["similarity"] == best_similarity
    ][:top_n]

    return verdict, top_matches

   


# -----------------------------
# Combined system (Hybrid)
# -----------------------------
def full_check(news_text):
    # Search ke liye important entity/topic nikalo
    nlp = get_spacy_model()
    doc = nlp(news_text)

    entities = [
        ent.text
        for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE"]
    ]

    # Example:
    # "Narendra Modi died yesterday"
    # -> "Narendra Modi"
    # Targeted search for important claim types
    lower_text = news_text.lower()

    # --------------------------------
# Create better search query
# --------------------------------

    if any(word in lower_text for word in [
        "died", "dead", "death", "killed"
    ]):
        search_query = " ".join(entities) + " died death"

    elif any(word in lower_text for word in [
        "resigned", "resign", "stepped down"
    ]):
        search_query = " ".join(entities) + " resigned resignation"

    elif any(word in lower_text for word in [
        "arrested", "arrest"
    ]):
        search_query = " ".join(entities) + " arrested arrest"

    else:
    # IMPORTANT:
    # Search the complete claim instead of only locations/entities.
        search_query = news_text

    if not search_query.strip():
        search_query = news_text

    print("NewsAPI search query:", search_query)

    trusted_articles = fetch_trusted_news(search_query)
    print("Trusted articles found:", len(trusted_articles))

    for i, article in enumerate(trusted_articles):
        print(f"{i+1}. {article}")
    # Exact match check
    if check_exact_match(news_text, trusted_articles):
        return {
            "ContradictionCheck":
            "🟢 Likely Real : Exact match found in trusted dataset (100% similarity)"
        }

    # Build entity state
    entity_state = build_entity_state(trusted_articles)

    # Logical contradiction check
    # is_contra, reason = detect_logical_contradiction(
    #     news_text, entity_state
    # )

    # if is_contra:
    #     return {
    #         "ContradictionCheck_3": reason,
    #         "Matches": []
    #     }

    # NLI fallback
    verdict, matches = nli_confidence_verdict(
        news_text, trusted_articles
    )

    return {
        "ContradictionCheck_3": verdict,
        "Matches": matches
    }
# -------------------------------
# Streamlit UI
# -------------------------------

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection Demo</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns([1, 1])

# Left column
with col1:
    user_input = st.text_area(
        "Enter a news headline or article:",
        height=200
    )

    check_button = st.button("Check")

# Right column
with col2:

    if check_button:

        if not user_input.strip():

            st.warning("Please enter some text.")

        else:

            with st.spinner("Analyzing news..."):

                result = full_check(user_input.strip())

            # Show verdict
            verdict = (
                result.get("ContradictionCheck")
                or result.get("ContradictionCheck_3")
                or "🟡 Needs Review"
            )

            if "Real" in verdict:
                st.success(verdict)

            elif "Fake" in verdict:
                st.error(verdict)

            else:
                st.warning(verdict)

            # Show matching trusted articles
            matches = result.get("Matches", [])

            if matches:

                st.subheader("Related Trusted News")

                for article, similarity, contradiction in matches:

                    st.markdown(
                        f"**Similarity:** {similarity:.2f}  \n"
                        f"**Contradiction Score:** {contradiction:.2f}"
                    )

                    st.write(article)

                    st.divider()

            else:

                st.info("No related trusted news found.")


# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    if st.button("Clear Cache"):

        st.cache_data.clear()
        st.cache_resource.clear()

        st.success("Cache cleared! Please rerun the app.")


# Footer
st.markdown("---")