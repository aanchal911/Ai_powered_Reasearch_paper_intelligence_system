# app.py — AI Research Paper Intelligence System Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import json, os, pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
import re

st.set_page_config(
    page_title="AI Research Paper Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    .stExpander { border: 1px solid #333 !important; }
    .score-high { color: #00c853; font-weight: bold; }
    .score-med  { color: #ffab00; font-weight: bold; }
    .score-low  { color: #ff5252; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def data_ready():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in [
        "cleaned_papers.csv", "arxiv_embeddings.npy", "paper_faiss.index",
        "lstm_classifier.h5", "trend_report.csv"
    ])

if not data_ready():
    st.error("Pipeline data not found. Please run `python run_pipeline.py` first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_models():
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline
    from keybert import KeyBERT
    import spacy
    from tensorflow.keras.models import load_model

    df           = pd.read_csv(f"{DATA_DIR}/cleaned_papers.csv")
    search_model = SentenceTransformer("all-MiniLM-L6-v2")
    index        = faiss.read_index(f"{DATA_DIR}/paper_faiss.index")
    summarizer   = hf_pipeline("summarization", model="t5-small", framework="pt")
    kw_model     = KeyBERT(model=search_model)
    nlp          = spacy.load("en_core_web_sm")
    lstm         = load_model(f"{DATA_DIR}/lstm_classifier.h5")

    with open(f"{DATA_DIR}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return df, search_model, index, summarizer, kw_model, nlp, lstm, tokenizer, le

with st.spinner("🔄 Loading AI models..."):
    df, search_model, index, summarizer, kw_model, nlp, lstm, tokenizer, le = load_models()

import faiss
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

def search_papers(query, k=5, threshold=None):
    qe = search_model.encode([query]).astype("float32")
    faiss.normalize_L2(qe)
    D, I = index.search(qe, k)
    results = []
    for r in range(k):
        score = float(D[0][r])
        if threshold and score < threshold:
            continue
        results.append({"rank": r+1, "score": round(score, 4),
                         "title": df.iloc[I[0][r]]["title"],
                         "abstract": df.iloc[I[0][r]]["abstract"]})
    return results

def summarize(text):
    return summarizer("summarize: " + text[:512], max_length=80, min_length=30,
                      do_sample=False)[0]["summary_text"]

def predict_topic(text):
    seq  = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding="post")
    pred = lstm.predict(seq, verbose=0)[0]
    topic = le.classes_[np.argmax(pred)]
    conf  = float(np.max(pred)) * 100
    all_probs = {le.classes_[i]: round(float(pred[i])*100, 1) for i in range(len(le.classes_))}
    return topic, conf, all_probs

def score_color(score):
    if score >= 0.75: return "🟢"
    if score >= 0.55: return "🟡"
    return "🔴"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "🏠 Overview",
    "🔍 Search Papers",
    "🔀 Compare Queries",
    "📊 Topic Trends",
    "🧠 Classify Paper",
    "📈 EDA & Insights",
    "📄 Final Report",
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} papers")
st.sidebar.markdown("**Model:** all-MiniLM-L6-v2")
st.sidebar.markdown("**Classifier:** LSTM (~64% acc)")

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🤖 AI Research Paper Intelligence System")
    st.markdown("An end-to-end NLP pipeline that searches, summarizes, extracts keywords, identifies entities, and classifies ArXiv ML papers.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Total Papers", f"{len(df):,}")
    c2.metric("🏷️ Topics", len(le.classes_))
    c3.metric("🔍 Search Model", "MiniLM-L6")
    c4.metric("🧠 Classifier", "LSTM")

    st.markdown("---")
    st.subheader("🏗️ Pipeline Architecture")
    st.markdown("""
    ```
    Query → FAISS Search → T5 Summarizer → KeyBERT Keywords → spaCy NER → LSTM Classifier → Report
    ```
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛠️ Tech Stack")
        tech = {
            "FAISS": "Semantic vector search",
            "Sentence Transformers": "384-dim embeddings",
            "T5-small": "Text summarization",
            "KeyBERT": "Keyword extraction",
            "spaCy NER": "Entity recognition",
            "LSTM (Keras)": "Topic classification",
            "Streamlit": "Dashboard UI",
        }
        for k, v in tech.items():
            st.markdown(f"- **{k}** — {v}")

    with col2:
        st.subheader("📊 Topic Distribution")
        trend_df = pd.read_csv(f"{DATA_DIR}/trend_report.csv").sort_values("Count", ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = cm.Set3(np.linspace(0, 1, len(trend_df)))
        ax.barh(trend_df["Topic"], trend_df["Count"], color=colors)
        ax.set_xlabel("Paper Count")
        plt.tight_layout()
        st.pyplot(fig)

# ── Page: Search Papers ───────────────────────────────────────────────────────
elif page == "🔍 Search Papers":
    st.title("🔍 Semantic Paper Search")
    st.markdown("Uses FAISS + sentence embeddings to find the most relevant papers for your query.")

    col1, col2, col3 = st.columns([4, 1, 1])
    query     = col1.text_input("Enter your research query", "deep learning for medical image segmentation")
    k         = col2.slider("Top K results", 1, 10, 5)
    threshold = col3.slider("Min similarity", 0.0, 1.0, 0.0, 0.05)

    if st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            results = search_papers(query, k=k, threshold=threshold if threshold > 0 else None)

        if not results:
            st.warning("No results above threshold. Try lowering the minimum similarity.")
        else:
            # Score bar chart
            titles_short = [r["title"][:40] + "..." for r in results]
            scores       = [r["score"] for r in results]
            fig, ax = plt.subplots(figsize=(8, 2.5))
            bars = ax.barh(titles_short[::-1], scores[::-1],
                           color=["#00c853" if s >= 0.75 else "#ffab00" if s >= 0.55 else "#ff5252" for s in scores[::-1]])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Similarity Score")
            ax.set_title("Search Result Scores")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"**Found {len(results)} results**")
            st.markdown("---")

            for r in results:
                color = score_color(r["score"])
                with st.expander(f"{color} #{r['rank']} | Score: {r['score']} — {r['title']}"):
                    tab1, tab2, tab3 = st.tabs(["📝 Summary & Abstract", "🔑 Keywords & NER", "🏷️ Classification"])

                    with tab1:
                        st.markdown("**Abstract:**")
                        st.write(r["abstract"])
                        st.markdown("**AI Summary:**")
                        with st.spinner("Summarizing..."):
                            st.info(summarize(r["abstract"]))

                    with tab2:
                        kws = kw_model.extract_keywords(r["abstract"],
                              keyphrase_ngram_range=(1, 2), stop_words="english", top_n=8)
                        kw_df = pd.DataFrame(kws, columns=["Keyword", "Score"])
                        kw_df["Score"] = kw_df["Score"].round(3)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Top Keywords:**")
                            st.dataframe(kw_df, use_container_width=True, hide_index=True)
                        with col_b:
                            doc    = nlp(r["abstract"])
                            orgs   = list(set(e.text for e in doc.ents if e.label_ == "ORG"))
                            people = list(set(e.text for e in doc.ents if e.label_ == "PERSON"))
                            locs   = list(set(e.text for e in doc.ents if e.label_ in ["GPE","LOC"]))
                            st.markdown("**Named Entities:**")
                            if people: st.write("👤 People:", ", ".join(people[:5]))
                            if orgs:   st.write("🏢 Orgs:", ", ".join(orgs[:5]))
                            if locs:   st.write("📍 Locations:", ", ".join(locs[:5]))
                            if not any([people, orgs, locs]):
                                st.write("No named entities found.")

                    with tab3:
                        topic, conf, all_probs = predict_topic(r["abstract"])
                        st.success(f"**Predicted Topic: {topic}**")
                        st.metric("Confidence", f"{conf:.1f}%")
                        prob_df = pd.DataFrame(list(all_probs.items()), columns=["Topic", "Probability %"]).sort_values("Probability %", ascending=False)
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        ax2.barh(prob_df["Topic"], prob_df["Probability %"], color="steelblue")
                        ax2.set_xlabel("Probability %")
                        plt.tight_layout()
                        st.pyplot(fig2)

# ── Page: Compare Queries ─────────────────────────────────────────────────────
elif page == "🔀 Compare Queries":
    st.title("🔀 Multi-Query Comparison")
    st.markdown("Compare two research queries side by side to find overlapping and unique papers.")

    col1, col2 = st.columns(2)
    q1 = col1.text_input("Query 1", "transformer models for NLP")
    q2 = col2.text_input("Query 2", "BERT for text classification")
    k  = st.slider("Top K per query", 3, 10, 5)

    if st.button("🔀 Compare", type="primary"):
        with st.spinner("Running both queries..."):
            r1 = search_papers(q1, k=k)
            r2 = search_papers(q2, k=k)

        titles1 = set(r["title"] for r in r1)
        titles2 = set(r["title"] for r in r2)
        overlap = titles1 & titles2

        c1, c2, c3 = st.columns(3)
        c1.metric("Query 1 Results", len(r1))
        c2.metric("Query 2 Results", len(r2))
        c3.metric("🔁 Overlapping Papers", len(overlap))

        if overlap:
            st.success("**Papers appearing in BOTH queries:**")
            for t in overlap:
                st.markdown(f"- {t}")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(f"Query 1: *{q1}*")
            for r in r1:
                color = score_color(r["score"])
                st.markdown(f"{color} **{r['score']}** — {r['title']}")

        with col_b:
            st.subheader(f"Query 2: *{q2}*")
            for r in r2:
                color = score_color(r["score"])
                st.markdown(f"{color} **{r['score']}** — {r['title']}")

# ── Page: Topic Trends ────────────────────────────────────────────────────────
elif page == "📊 Topic Trends":
    st.title("📊 Topic Trend Analysis")
    st.markdown("Frequency analysis of research topics across the dataset.")

    trend_df = pd.read_csv(f"{DATA_DIR}/trend_report.csv").sort_values("Count", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = cm.tab10(np.linspace(0, 1, len(trend_df)))
        ax.bar(trend_df["Topic"], trend_df["Count"], color=colors, edgecolor="black")
        ax.set_xticklabels(trend_df["Topic"], rotation=45, ha="right")
        ax.set_title("Topic Frequency in ArXiv Papers")
        ax.set_ylabel("Number of Papers")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(trend_df["Count"], labels=trend_df["Topic"],
                autopct="%1.1f%%", startangle=140, colors=colors)
        ax2.set_title("Topic Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("📋 Raw Counts")
    trend_df["% Share"] = (trend_df["Count"] / trend_df["Count"].sum() * 100).round(1)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

    # Top topic insight
    top = trend_df.iloc[0]
    st.info(f"**Most dominant topic:** {top['Topic']} with {top['Count']:,} papers ({top['% Share']}% of dataset)")

# ── Page: Classify Paper ──────────────────────────────────────────────────────
elif page == "🧠 Classify Paper":
    st.title("🧠 LSTM Topic Classifier")
    st.markdown("Paste any paper title or abstract and the LSTM model will predict its research topic.")

    text_input = st.text_area("Paper text", height=180,
                               value="This paper proposes a convolutional neural network for detecting tumors in MRI scans using deep learning and transfer learning techniques.")

    col1, col2 = st.columns([1, 3])
    if col1.button("🧠 Classify", type="primary"):
        topic, conf, all_probs = predict_topic(text_input)

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"**Predicted Topic:** {topic}")
            st.metric("Confidence", f"{conf:.2f}%")
            if conf >= 75:
                st.markdown("🟢 High confidence prediction")
            elif conf >= 50:
                st.markdown("🟡 Medium confidence — result may vary")
            else:
                st.markdown("🔴 Low confidence — text may be ambiguous")

        with col_b:
            st.markdown("**All Topic Probabilities:**")
            prob_df = pd.DataFrame(list(all_probs.items()),
                                   columns=["Topic", "Probability %"]).sort_values("Probability %", ascending=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_bar = ["#00c853" if t == topic else "steelblue" for t in prob_df["Topic"]]
            ax.barh(prob_df["Topic"], prob_df["Probability %"], color=colors_bar)
            ax.set_xlabel("Probability %")
            ax.set_title("Topic Probability Distribution")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Available Topics")
        for topic in le.classes_:
            st.markdown(f"- {topic}")
    with col2:
        report_path = f"{DATA_DIR}/classification_report.json"
        if os.path.exists(report_path):
            st.subheader("📊 Model Performance")
            cr = load_json(report_path)
            st.metric("Overall Accuracy", f"{cr['accuracy']*100:.1f}%")
            st.caption("Labels generated via TF-IDF + KMeans clustering — data-driven, not hand-coded rules")
            rows = []
            for label, vals in cr.items():
                if isinstance(vals, dict) and "precision" in vals:
                    rows.append({"Topic": label, "Precision": round(vals["precision"],2),
                                 "Recall": round(vals["recall"],2), "F1": round(vals["f1-score"],2)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Page: EDA & Insights ──────────────────────────────────────────────────────
elif page == "📈 EDA & Insights":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Deep dive into the dataset statistics and patterns.")

    tab1, tab2, tab3 = st.tabs(["📊 Word Stats", "🔤 Top Words", "📅 Year Trends"])

    with tab1:
        df["word_count"] = df["abstract"].apply(lambda x: len(str(x).split()))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Words/Abstract", f"{df['word_count'].mean():.0f}")
        c2.metric("Max Words", df["word_count"].max())
        c3.metric("Min Words", df["word_count"].min())
        c4.metric("Total Papers", len(df))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["word_count"], bins=50, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title("Distribution of Abstract Word Counts")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Number of Papers")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        all_words = " ".join(df["title"].dropna().tolist()).lower()
        all_words = re.sub(r"[^a-z\s]", "", all_words)
        stopwords = {"a","an","the","of","in","for","on","with","and","to","is",
                     "are","using","based","via","from","by","as","at","we","our",
                     "this","that","be","its","towards","learning"}
        word_list = [w for w in all_words.split() if w not in stopwords and len(w) > 2]
        common = Counter(word_list).most_common(20)
        words, counts = zip(*common)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(words, counts, color="coral", edgecolor="black")
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_title("Top 20 Most Common Words in Paper Titles")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Top 20 words:**")
        word_df = pd.DataFrame(common, columns=["Word", "Count"])
        st.dataframe(word_df, use_container_width=True, hide_index=True)

    with tab3:
        df["year"] = df["abstract"].str.extract(r'(20[0-2][0-9])')
        year_counts = df["year"].value_counts().sort_index().dropna()

        if len(year_counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(year_counts.index, year_counts.values, color="mediumseagreen", edgecolor="black")
            ax.set_title("Papers Mentioning Each Year in Abstract")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No year data found in abstracts.")

# ── Page: Final Report ────────────────────────────────────────────────────────
elif page == "📄 Final Report":
    st.title("📄 Final Intelligence Report")
    st.markdown("Pre-generated report combining search, summarization, keywords, and NER.")

    report_path = f"{DATA_DIR}/final_report.json"
    csv_path    = f"{DATA_DIR}/final_report.csv"

    # Option to generate new report
    st.subheader("Generate New Report")
    new_query = st.text_input("Query for report", "deep learning for medical image analysis")
    if st.button("⚡ Generate Report", type="primary"):
        with st.spinner("Generating full intelligence report..."):
            results = search_papers(new_query, k=5)
            report  = []
            for r in results:
                r["summary"]  = summarize(r["abstract"])
                kws           = kw_model.extract_keywords(r["abstract"], keyphrase_ngram_range=(1,2), stop_words="english", top_n=5)
                r["keywords"] = [kw for kw, _ in kws]
                doc           = nlp(r["abstract"])
                r["organizations"] = list(set(e.text for e in doc.ents if e.label_ == "ORG"))
                r["people"]        = list(set(e.text for e in doc.ents if e.label_ == "PERSON"))
                topic, conf, _     = predict_topic(r["abstract"])
                r["topic"]         = topic
                r["confidence"]    = round(conf, 1)
                report.append(r)

            with open(report_path, "w") as f:
                json.dump({"query": new_query, "results": report}, f, indent=2)
            rdf = pd.DataFrame(report)
            rdf["keywords"]      = rdf["keywords"].apply(lambda x: ", ".join(x))
            rdf["organizations"] = rdf["organizations"].apply(lambda x: ", ".join(x))
            rdf.to_csv(csv_path, index=False)
        st.success("Report generated!")

    st.markdown("---")

    if os.path.exists(report_path):
        report = load_json(report_path)
        st.subheader(f"📋 Report for: *{report['query']}*")

        for r in report["results"]:
            color = score_color(r["score"])
            with st.expander(f"{color} #{r['rank']} | Score: {r['score']} — {r['title']}"):
                if r.get("genai_brief"):
                    st.markdown("**🤖 AI-Generated Intelligence Brief:**")
                    st.success(r["genai_brief"])
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Summary:**")
                    st.info(r.get("summary", "N/A"))
                    st.markdown("**Keywords:**")
                    st.write(", ".join(r.get("keywords", [])))
                with col_b:
                    st.markdown("**Organizations:**")
                    st.write(", ".join(r.get("organizations", [])) or "None found")
                    st.markdown("**People:**")
                    st.write(", ".join(r.get("people", [])) or "None found")
                    if r.get("topic"):
                        st.markdown(f"**Topic:** {r['topic']} ({r.get('confidence', '')}%)")

        if os.path.exists(csv_path):
            st.download_button(
                "⬇️ Download Full Report as CSV",
                data=open(csv_path).read(),
                file_name="intelligence_report.csv",
                mime="text/csv"
            )
    else:
        st.info("No report yet. Generate one above.")
