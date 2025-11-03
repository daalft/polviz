import os, json, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
import pickle

agg = []
for yy in (16, 20, 24):
    d = pickle.load(open("./polviz/data/prob_20{}.pickle".format(yy), "rb"))
    for line in d:
        agg.append(("20{}".format(yy), line.rstrip()))

df = pd.DataFrame(agg, columns=["year", "text"])

TEXT_COL = "text"                        # <-- column with responses
COLOR_META = "year"                      # <-- column to color points ('year' works nicely)
OPTIONAL_HOVER = ["year"]                # <-- extra columns to show on hover; add eg 'party' if present
MODEL_NAME = "all-MiniLM-L6-v2"          # small, fast embeddings
CACHE_DIR = Path("./cache_semantic_atlas")
EXPORT_HTML = "./viz/semantic_atlas.html"
SEED = 42
N_COMPONENTS = 2
N_NEIGHBORS = 15
MIN_DIST = 0.1

CACHE_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(SEED)

assert TEXT_COL in df.columns, f"Missing column: {TEXT_COL}"
df = df.dropna(subset=[TEXT_COL]).copy()
keep = [TEXT_COL] +  OPTIONAL_HOVER
keep = [c for c in keep if c in df.columns]
df = df[keep].reset_index(drop=True)

def basic_clean(s: str) -> str:
    s = str(s)
    return " ".join(s.lower().split())

df["_clean"] = df[TEXT_COL].map(basic_clean)

emb_path = CACHE_DIR / "embeddings.npy"
idx_path = CACHE_DIR / "emb_index.json"

if emb_path.exists() and idx_path.exists():
    X = np.load(emb_path)
    with open(idx_path, "r") as f:
        cached_idx = json.load(f)
    # Guard: if dataset changed length, re-embed
    if len(cached_idx) != len(df):
        print("Cache size mismatch, recomputing embeddings…")
        model = SentenceTransformer(MODEL_NAME)
        X = model.encode(df["_clean"].tolist(), show_progress_bar=True, normalize_embeddings=True)
        np.save(emb_path, X)
        json.dump(list(range(len(df))), open(idx_path, "w"))
else:
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(df["_clean"].tolist(), show_progress_bar=True, normalize_embeddings=True)
    np.save(emb_path, X)
    json.dump(list(range(len(df))), open(idx_path, "w"))

try:
    import umap
    reducer = umap.UMAP(n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS,
                        min_dist=MIN_DIST, random_state=SEED, metric="cosine")
    XY = reducer.fit_transform(X)
    method_used = "UMAP"
except Exception as e:
    print("UMAP not available, falling back to t-SNE:", e)
    from sklearn.manifold import TSNE
    perplexity = max(5, min(30, len(df)//3))
    XY = TSNE(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=SEED, metric="cosine").fit_transform(X)
    method_used = "t-SNE"

df["x"], df["y"] = XY[:,0], XY[:,1]

USE_HDBSCAN = True  # set False to use KMeans below

if USE_HDBSCAN:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, metric="euclidean")  # tweak if needed
    labels = clusterer.fit_predict(XY)  # cluster in 2D space for intuitive groups
    df["cluster"] = labels.astype(int)
else:
    from sklearn.cluster import KMeans
    K = 10
    df["cluster"] = KMeans(n_clusters=K, random_state=SEED, n_init="auto").fit_predict(X)

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from itertools import chain

def top_terms_for_cluster(
    df, 
    cluster_col="cluster", 
    text_col="_clean", 
    k=5, 
    min_docs=5,             # clusters with < min_docs get generic labels
    skip_noise=True         # skip HDBSCAN noise label -1
):
    """
    Returns {cluster_int: 'term1, term2, ...'} with multiple fallbacks
    to avoid empty vocabulary errors.
    """
    labels = {}

    # utility: basic tokenization (fallback)
    token_re = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
    def tokenize(s):
        return token_re.findall(str(s).lower())

    # cluster iteration
    clusters = sorted(df[cluster_col].dropna().unique())
    for cl in clusters:
        if skip_noise and int(cl) == -1:
            labels[int(cl)] = "(noise)"
            continue

        sub = df.loc[(df[cluster_col] == cl) & df[text_col].notna(), text_col].astype(str)
        # remove rows that are blank or have no word characters
        sub = sub[sub.str.contains(r"\w", regex=True)]
        n_docs = len(sub)

        if n_docs < min_docs:
            labels[int(cl)] = f"(cluster {int(cl)}; {n_docs} docs)"
            continue

        texts = sub.tolist()

        # try a sequence of vectorizers from stricter -> looser
        vectorizers = [
            # 1) Typical TF-IDF (unigram+bigram), english stopwords, min_df=2
            TfidfVectorizer(ngram_range=(1,2), max_features=5000, 
                            stop_words="english", min_df=2),
            # 2) Unigram only, english stopwords, min_df=1
            TfidfVectorizer(ngram_range=(1,1), max_features=5000, 
                            stop_words="english", min_df=1),
            # 3) Unigram only, no stopwords (very loose), min_df=1
            TfidfVectorizer(ngram_range=(1,1), max_features=5000, 
                            stop_words=None, min_df=1),
        ]

        label = None
        for vec in vectorizers:
            try:
                Xtf = vec.fit_transform(texts)
                if Xtf.shape[1] == 0:
                    # empty vocabulary or all filtered
                    continue
                means = np.asarray(Xtf.mean(axis=0)).ravel()
                if means.size == 0:
                    continue
                top_idx = means.argsort()[::-1][:k]
                vocab = np.array(vec.get_feature_names_out())[top_idx]
                label = ", ".join(vocab)
                break
            except ValueError:
                # empty vocabulary; try next vectorizer
                continue

        # final fallback: simple token frequency
        if not label:
            tokens = list(chain.from_iterable(tokenize(t) for t in texts))
            if tokens:
                most = [w for w, _ in Counter(tokens).most_common(k)]
                label = ", ".join(most)
            else:
                label = f"(cluster {int(cl)}; no salient terms)"

        labels[int(cl)] = label

    return labels

cluster_labels = top_terms_for_cluster(df, cluster_col="cluster", text_col="_clean", k=5)
df["cluster_label"] = df["cluster"].map(cluster_labels).fillna("cluster")
cluster_labels

# If needed once: !pip -q install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df["sentiment_compound"] = df["_clean"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

def sent_label(c):
    # VADER's conventional thresholds
    if c > 0.05: return "positive"
    if c < -0.05: return "negative"
    return "neutral"

df["sentiment_label"] = df["sentiment_compound"].apply(sent_label)
df["sentiment_label"] = pd.Categorical(df["sentiment_label"], categories=["negative","neutral","positive"], ordered=True)

# --- Sentiment-annotated Semantic Atlas (categorical) ---
COLOR_BY = "sentiment_label"  # negative / neutral / positive
assert COLOR_BY in df.columns, "Run the VADER block first to create 'sentiment_label'."

hover_cols = (
    [TEXT_COL]
    + [c for c in OPTIONAL_HOVER if c in df.columns]
    + ["cluster", "cluster_label"] * int("cluster" in df.columns)
    + ["sentiment_compound", "sentiment_label"]
)

fig_sent_cat = px.scatter(
    df, x="x", y="y",
    color=COLOR_BY,
    category_orders={"sentiment_label": ["negative","neutral","positive"]},
    hover_data=hover_cols,
    title=f"Semantic Atlas of Public Concerns — sentiment (categorical) [{method_used}]",
    opacity=0.9,
    height=800
)
fig_sent_cat.update_traces(marker=dict(size=5))
fig_sent_cat.show()

out_cat = "./viz/semantic_atlas_sentiment_categorical.html"
pio.write_html(fig_sent_cat, file=out_cat, auto_open=False, include_plotlyjs="cdn")

# --- Sentiment-annotated Semantic Atlas (continuous) ---
fig_sent_cont = px.scatter(
    df, x="x", y="y",
    color="sentiment_compound",               # continuous in [-1, 1]
    color_continuous_scale="RdBu",
    range_color=[-1, 1],
    hover_data=hover_cols,
    title=f"Semantic Atlas of Public Concerns — sentiment (continuous) [{method_used}]",
    opacity=0.9,
    height=800
)
fig_sent_cont.update_traces(marker=dict(size=5))
fig_sent_cont.show()

out_cont = "./viz/semantic_atlas_sentiment_continuous.html"
pio.write_html(fig_sent_cont, file=out_cont, auto_open=False, include_plotlyjs="cdn")

COLOR_BY = "cluster"
hover_cols = [TEXT_COL] + [c for c in OPTIONAL_HOVER if c in df.columns] + ["cluster_label"]
fig = px.scatter(
    df, x="x", y="y",
    color=COLOR_BY,
    hover_data=hover_cols,
    title=f"Semantic Atlas of Public Concerns ({method_used})",
    opacity=0.9,
    height=800
)
fig.update_traces(marker=dict(size=5))
fig.show()

pio.write_html(fig, file="./viz/semantic_atlas_cluster.html", auto_open=False, include_plotlyjs="cdn")

if COLOR_META in df.columns:
    fig2 = px.scatter(
        df, x="x", y="y",
        color="cluster",
        facet_col=COLOR_META, facet_col_wrap=3,
        hover_data=hover_cols,
        title=f"Semantic Atlas by Year (colored by cluster)",
        opacity=0.9,
        height=900
    )
    fig2.update_traces(marker=dict(size=4))
    fig2.for_each_annotation(lambda a: a.update(text=a.text.replace("=", ": ")))
    fig2.show()
    pio.write_html(fig2, file="semantic_atlas_by_year.html", auto_open=False, include_plotlyjs="cdn")

assert "year" in df.columns, "Column 'year' required for animation."
df["year"] = df["year"].astype(str)
year_order = sorted(df["year"].unique())

hover_cols = [c for c in [TEXT_COL] + OPTIONAL_HOVER + ["cluster","cluster_label","sentiment_compound","sentiment_label"] if c in df.columns]

fig_anim = px.scatter(
    df, x="x", y="y",
    color="sentiment_label",                 # categorical colors
    animation_frame="year",
    category_orders={"year": year_order, "sentiment_label": ["negative","neutral","positive"]},
    hover_data=hover_cols,
    title="Semantic Atlas by Year — colored by Sentiment",
    opacity=0.9,
    height=800
)
fig_anim.update_traces(marker=dict(size=5))
fig_anim.update_layout(transition={"duration": 500})
fig_anim.show()

pio.write_html(fig_anim, file="./viz/semantic_atlas_by_year_sentiment.html", auto_open=False, include_plotlyjs="cdn")

fig_facets = px.scatter(
    df, x="x", y="y",
    color="sentiment_label",
    facet_col="year", facet_col_wrap=3,
    category_orders={"year": year_order, "sentiment_label": ["negative","neutral","positive"]},
    hover_data=hover_cols,
    title="Semantic Atlas — small multiples by year (sentiment colors)",
    opacity=0.9,
    height=900
)
fig_facets.update_traces(marker=dict(size=4))
fig_facets.for_each_annotation(lambda a: a.update(text=a.text.replace("=", ": ")))
fig_facets.show()

pio.write_html(fig_facets, file="./viz/semantic_atlas_facets_by_year_sentiment.html", auto_open=False, include_plotlyjs="cdn")