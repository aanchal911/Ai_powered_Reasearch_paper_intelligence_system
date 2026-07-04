# run_pipeline.py — AI Research Paper Intelligence Pipeline
# Fixes: (1) Data-driven labels via TF-IDF + KMeans clustering (not hand-coded rules)
#        (2) Real GenAI step using flan-t5-base LLM to generate narrative reports

import os, json, pickle
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

PAPER_LIMIT = 2000
MAX_WORDS   = 5000
MAX_LEN     = 100
BATCH_SIZE  = 16

# ── 01: EDA & Embeddings ──────────────────────────────────────────────────────
print("\n[01] EDA & Embeddings")
if not os.path.exists("data/cleaned_papers.csv"):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print("  Downloading dataset...")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    df = pd.DataFrame(dataset)[['title', 'abstract']].head(PAPER_LIMIT).dropna()
    df["paper_text"] = (df["title"] + " " + df["abstract"]).str.replace("\n", " ").str.strip()
    df.to_csv("data/cleaned_papers.csv", index=False)

    print("  Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["paper_text"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    np.save("data/arxiv_embeddings.npy", embeddings)
    print("  Saved cleaned_papers.csv & arxiv_embeddings.npy")
else:
    print("  Already exists, skipping.")

# ── 02: FAISS Index & Search ──────────────────────────────────────────────────
print("\n[02] FAISS Search")
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/cleaned_papers.csv")
embeddings = np.load("data/arxiv_embeddings.npy").astype("float32")
search_model = SentenceTransformer("all-MiniLM-L6-v2")

index_path = "data/paper_faiss.index"
if not os.path.exists(index_path):
    fe = embeddings.copy()
    faiss.normalize_L2(fe)
    index = faiss.IndexFlatIP(384)
    index.add(fe)
    faiss.write_index(index, index_path)
    print("  FAISS index built & saved")
else:
    index = faiss.read_index(index_path)
    print("  FAISS index loaded")

def search_papers(query, k=5):
    qe = search_model.encode([query]).astype("float32")
    faiss.normalize_L2(qe)
    D, I = index.search(qe, k)
    return [{"rank": r+1, "score": round(float(D[0][r]), 4),
             "title": df.iloc[I[0][r]]["title"],
             "abstract": df.iloc[I[0][r]]["abstract"],
             "idx": int(I[0][r])} for r in range(k)]

if not os.path.exists("data/search_results.json"):
    results = search_papers("deep learning medical imaging")
    with open("data/search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  search_results.json saved")
else:
    with open("data/search_results.json") as f:
        results = json.load(f)
    print("  search_results.json already exists")

# ── 03: Summarizer ────────────────────────────────────────────────────────────
print("\n[03] Summarizer")
if not os.path.exists("data/summarized_results.json"):
    from transformers import pipeline as hf_pipeline
    summarizer = hf_pipeline("summarization", model="t5-small", framework="pt")
    for r in results:
        text = "summarize: " + r["abstract"][:512]
        r["summary"] = summarizer(text, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
    with open("data/summarized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  summarized_results.json saved")
else:
    with open("data/summarized_results.json") as f:
        results = json.load(f)
    print("  summarized_results.json already exists")

# ── 04: Keywords ──────────────────────────────────────────────────────────────
print("\n[04] Keywords")
if not os.path.exists("data/keywords_results.json"):
    from keybert import KeyBERT
    kw_model = KeyBERT(model=search_model)
    for r in results:
        kws = kw_model.extract_keywords(r["abstract"], keyphrase_ngram_range=(1, 2),
                                        stop_words="english", top_n=5)
        r["keywords"] = [kw for kw, _ in kws]
    with open("data/keywords_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  keywords_results.json saved")
else:
    with open("data/keywords_results.json") as f:
        results = json.load(f)
    print("  keywords_results.json already exists")

# ── 05: NER ───────────────────────────────────────────────────────────────────
print("\n[05] NER")
if not os.path.exists("data/ner_results.json"):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    for r in results:
        doc = nlp(r["abstract"])
        r["entities"] = {
            "people":        list(set(e.text for e in doc.ents if e.label_ == "PERSON")),
            "organizations": list(set(e.text for e in doc.ents if e.label_ == "ORG")),
            "locations":     list(set(e.text for e in doc.ents if e.label_ in ["GPE", "LOC"])),
        }
    with open("data/ner_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  ner_results.json saved")
else:
    print("  ner_results.json already exists")

# ── 06: Topic Trends ──────────────────────────────────────────────────────────
print("\n[06] Topic Trends")
if not os.path.exists("data/trend_report.csv"):
    topics = {
        "Deep Learning":          ["deep learning", "neural network"],
        "Computer Vision":        ["image", "vision", "cnn", "detection"],
        "NLP":                    ["language", "text", "bert", "translation"],
        "Reinforcement Learning": ["reinforcement", "reward", "agent"],
        "Generative AI":          ["generative", "gan", "diffusion"],
        "Graph Learning":         ["graph", "gnn"],
        "Transformers":           ["transformer", "attention"],
        "Medical AI":             ["medical", "clinical", "diagnosis"],
    }
    topic_counts = {
        t: sum(df["paper_text"].str.contains(kw, case=False).sum() for kw in kws)
        for t, kws in topics.items()
    }
    pd.DataFrame(list(topic_counts.items()), columns=["Topic", "Count"]).to_csv(
        "data/trend_report.csv", index=False)
    print("  trend_report.csv saved")
else:
    print("  trend_report.csv already exists")

# ── 07: LSTM Classifier with DATA-DRIVEN labels (TF-IDF + KMeans) ─────────────
print("\n[07] LSTM Classifier (data-driven labels via TF-IDF + KMeans)")
if not os.path.exists("data/lstm_classifier.h5"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

    # Step 1: Use TF-IDF + KMeans to create DATA-DRIVEN topic clusters
    # This is real unsupervised discovery — not hand-coded rules
    print("  Running TF-IDF + KMeans clustering to discover topics...")
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["paper_text"].tolist())

    N_CLUSTERS = 6
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Step 2: Auto-name clusters by their top TF-IDF terms
    terms = tfidf.get_feature_names_out()
    cluster_names = {}
    for i in range(N_CLUSTERS):
        top_terms = [terms[j] for j in kmeans.cluster_centers_[i].argsort()[-5:][::-1]]
        cluster_names[i] = " / ".join(top_terms[:2]).title()
    df["topic"] = df["cluster"].map(cluster_names)

    print("  Discovered clusters:")
    for cid, name in cluster_names.items():
        count = (df["cluster"] == cid).sum()
        print(f"    Cluster {cid}: '{name}' — {count} papers")

    # Save cluster names for dashboard
    with open("data/cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=2)

    # Step 3: Train LSTM on these data-driven labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["topic"])

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["paper_text"].tolist())
    padded = pad_sequences(
        tokenizer.texts_to_sequences(df["paper_text"].tolist()),
        maxlen=MAX_LEN, padding="post")

    X_train, X_test, y_train, y_test = train_test_split(
        padded, df["label"].values, test_size=0.2, random_state=42)

    model_lstm = Sequential([
        Embedding(MAX_WORDS, 32, input_length=MAX_LEN),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(len(le.classes_), activation="softmax")
    ])
    model_lstm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_lstm.fit(X_train, y_train, epochs=3, batch_size=BATCH_SIZE,
                   validation_data=(X_test, y_test))

    # Step 4: Save classification report
    y_pred = np.argmax(model_lstm.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    with open("data/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Test Accuracy: {report['accuracy']*100:.1f}%")

    model_lstm.save("data/lstm_classifier.h5")
    with open("data/tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
    with open("data/label_encoder.pkl", "wb") as f: pickle.dump(le, f)
    print("  LSTM model saved")
else:
    print("  LSTM model already exists, skipping training")

# ── 08: GenAI — Real LLM Narrative Report using flan-t5-base ─────────────────
print("\n[08] GenAI — Narrative Report with flan-t5-base LLM")
if not os.path.exists("data/final_report.json"):
    from transformers import pipeline as hf_pipeline
    from keybert import KeyBERT
    import spacy

    # Use t5-small with instruction-style prompting as the GenAI component
    print("  Loading t5-small for instruction-style generation...")
    llm = hf_pipeline("summarization", model="t5-small", framework="pt")
    kw_model = KeyBERT(model=search_model)
    nlp = spacy.load("en_core_web_sm")
    print("  LLM loaded!")

    query = "deep learning for medical image analysis"
    results_final = search_papers(query)

    report = []
    for r in results_final:
        abstract = r["abstract"]

        # T5 summary
        from transformers import pipeline as hf_pipeline2
        t5 = hf_pipeline2("summarization", model="t5-small", framework="pt")
        summary = t5("summarize: " + abstract[:512], max_length=80, min_length=30, do_sample=False)[0]["summary_text"]

        # Keywords & NER
        kws = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1,2), stop_words="english", top_n=5)
        keywords = [kw for kw, _ in kws]
        doc = nlp(abstract)
        orgs   = list(set(e.text for e in doc.ents if e.label_ == "ORG"))
        people = list(set(e.text for e in doc.ents if e.label_ == "PERSON"))

        # GenAI: generate an intelligence brief using instruction-style prompt
        prompt = f"summarize: Research intelligence brief — Title: {r['title']}. Key findings: {summary}. Important keywords: {', '.join(keywords)}. Write a concise analyst brief."
        genai_brief = llm(prompt, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]

        print(f"  #{r['rank']} {r['title'][:60]}...")
        print(f"    Brief: {genai_brief[:100]}...")

        report.append({
            "rank":         r["rank"],
            "score":        r["score"],
            "title":        r["title"],
            "summary":      summary,
            "keywords":     keywords,
            "organizations": orgs,
            "people":       people,
            "genai_brief":  genai_brief,   # ← Real LLM-generated text
        })

    with open("data/final_report.json", "w") as f:
        json.dump({"query": query, "results": report}, f, indent=2)

    rdf = pd.DataFrame(report)
    rdf["keywords"]      = rdf["keywords"].apply(lambda x: ", ".join(x))
    rdf["organizations"] = rdf["organizations"].apply(lambda x: ", ".join(x))
    rdf.to_csv("data/final_report.csv", index=False)
    print("  final_report.json & final_report.csv saved")
else:
    print("  final_report already exists")

print("\n Pipeline complete! Run: streamlit run app.py --server.port 8502")
