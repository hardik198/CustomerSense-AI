from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CustomerSense AI | Intelligence Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# Design tokens
INK = "#13221F"
MUTED = "#5E706B"
PAPER = "#F2F5F3"
CARD = "#FFFFFF"
LINE = "#DCE5E1"
ACCENT = "#0E7C6B"
POS = "#1E9E73"
NEG = "#D9505F"
NEU = "#E2A93B"
SENTIMENT_COLORS = {"Positive": POS, "Negative": NEG, "Neutral": NEU}
CLUSTER_COLORS = [
    "#0E7C6B", "#3D6FB6", "#E2A93B", "#D9505F", "#7C5CBF",
    "#2FA3B8", "#8A9A3B", "#C46A3A", "#5B7B7A", "#B0457A"
]

# =========================================================
# THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Source Sans 3', sans-serif;
    color: #13221F;
}

h1, h2, h3, h4, .hero h1, .section-title, .metric-value, .brand {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #F2F5F3;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0D2623;
    border-right: 1px solid #16403B;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {
    color: #E4F0EC;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #8FB0A8 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1B4741;
}

.brand {
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.3px;
    margin-bottom: 2px;
}

.brand-sub {
    color: #8FB0A8;
    font-size: 13px;
    margin-bottom: 6px;
}

/* sidebar radio as nav */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    padding: 9px 12px;
    border-radius: 10px;
    border: 1px solid transparent;
    width: 100%;
    transition: background .15s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #143730;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #17483F;
    border-color: #23695C;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 600;
    font-size: 15px;
}

/* sidebar inputs: keep text readable */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #12332E;
    border-color: #23695C;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #0E7C6B;
}

/* ---------- Hero ---------- */
.hero {
    background: #0D2623;
    background-image:
        radial-gradient(circle at 92% 20%, rgba(30,158,115,.35), transparent 38%),
        radial-gradient(circle at 70% 110%, rgba(61,111,182,.28), transparent 40%);
    padding: 36px 40px;
    border-radius: 18px;
    margin-bottom: 26px;
    border: 1px solid #16403B;
}

.hero .eyebrow {
    color: #7FD1B9;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .3px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.8px;
    line-height: 1.15;
    margin: 8px 0 10px 0;
    padding: 0;
}

.hero p {
    color: #BCD6CF;
    font-size: 16px;
    line-height: 1.55;
    max-width: 640px;
    margin: 0;
}

/* ---------- Sections ---------- */
.section-title {
    color: #13221F;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 30px 0 4px 0;
}

.section-sub {
    color: #5E706B;
    font-size: 14px;
    margin-bottom: 14px;
}

/* ---------- Metrics ---------- */
.metric-box {
    background: #FFFFFF;
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid #DCE5E1;
    border-left: 4px solid var(--accent, #0E7C6B);
    min-height: 116px;
}

.metric-label {
    color: #5E706B;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #13221F;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.6px;
    margin-top: 6px;
}

.metric-note {
    color: #6F817C;
    font-size: 13px;
    margin-top: 4px;
}

/* ---------- Insight + info ---------- */
.insight {
    background: #FFFFFF;
    border: 1px solid #DCE5E1;
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
}

.insight .k {
    color: #5E706B;
    font-size: 13px;
    font-weight: 600;
}

.insight .v {
    font-family: 'Sora', sans-serif;
    color: #13221F;
    font-size: 19px;
    font-weight: 700;
    margin: 4px 0 2px 0;
}

.insight .d {
    color: #6F817C;
    font-size: 13px;
}

.info-box {
    background: #FFFFFF;
    padding: 20px 22px;
    border: 1px solid #DCE5E1;
    border-radius: 12px;
    margin-bottom: 14px;
}

.info-box h4 {
    font-size: 16px;
    margin: 0 0 6px 0;
    color: #13221F;
}

.info-box p {
    color: #5E706B;
    margin: 0;
    line-height: 1.55;
}

.step {
    display: flex;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid #E6EDEA;
}

.step:last-child {
    border-bottom: none;
}

.step-n {
    flex: 0 0 30px;
    height: 30px;
    border-radius: 50%;
    background: #E3F2EE;
    color: #0E7C6B;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.step h4 {
    margin: 0 0 2px 0;
    font-size: 15px;
}

.step p {
    margin: 0;
    color: #5E706B;
    font-size: 14px;
}

.pill {
    display: inline-block;
    background: #E3F2EE;
    color: #0B5C4F;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 600;
}

/* ---------- Streamlit widgets ---------- */
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #DCE5E1;
    border-radius: 12px;
    padding: 6px;
    overflow: hidden;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #DCE5E1;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    border: 1px solid #0E7C6B;
    background: #0E7C6B;
    color: white;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #0B5C4F;
    border-color: #0B5C4F;
    color: white;
}

.stTextInput input {
    border-radius: 10px;
}

hr {
    border-color: #DCE5E1;
}

*:focus-visible {
    outline: 2px solid #0E7C6B;
    outline-offset: 2px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            return pd.DataFrame()


def load_first_available(*filenames):
    for filename in filenames:
        df = load_csv(str(OUTPUTS / filename))
        if not df.empty:
            return df, filename

    return pd.DataFrame(), None


reviews, review_source = load_first_available(
    "sentiment_reviews.csv",
    "clustered_reviews.csv"
)

keywords = load_csv(str(OUTPUTS / "cluster_keywords.csv"))

evaluation, evaluation_source = load_first_available(
    "model_evaluation_metrics.csv",
    "cluster_evaluation.csv",
    "clustering_metrics.csv"
)

sentiment_summary = load_csv(
    str(OUTPUTS / "sentiment_summary.csv")
)

sentiment_cluster = load_csv(
    str(OUTPUTS / "sentiment_by_cluster.csv")
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, possible_names):
    if df.empty:
        return None

    lookup = {str(c).lower().strip(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def number(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def metric_card(label, value, note="Project insight", color=ACCENT):
    st.markdown(
        f"""
        <div class="metric-box" style="--accent:{color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def insight_card(label, value, detail):
    st.markdown(
        f"""
        <div class="insight">
            <div class="k">{label}</div>
            <div class="v">{value}</div>
            <div class="d">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<div class="section-sub">{subtitle}</div>',
            unsafe_allow_html=True
        )


def hero(eyebrow, title, text):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Source Sans 3, sans-serif",
            color="#2B3D39",
            size=13
        ),
        title=dict(
            font=dict(family="Sora, sans-serif", size=15, color=INK),
            x=0.02
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
        colorway=CLUSTER_COLORS,
        hoverlabel=dict(font_family="Source Sans 3, sans-serif")
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=LINE
    )

    fig.update_yaxes(
        gridcolor="#EAF0ED",
        zeroline=False
    )

    return fig


def display_empty(message):
    st.info(message)


# =========================================================
# VALIDATE DATA
# =========================================================

if reviews.empty:
    st.error(
        "Customer review data was not found. "
        "Run your project pipeline first, then reload this page."
    )

    st.write("Expected file:")
    st.code("outputs/sentiment_reviews.csv")

    st.stop()


cluster_col = find_column(
    reviews,
    ["cluster", "cluster_id", "clusters"]
)

sentiment_col = find_column(
    reviews,
    ["sentiment", "sentiment_label", "sentiment_category"]
)

text_col = find_column(
    reviews,
    ["text", "review", "review_text", "cleaned_text", "original_text"]
)

rating_col = find_column(
    reviews,
    ["rating", "score", "stars"]
)

compound_col = find_column(
    reviews,
    ["compound_score", "compound"]
)

# Work on a separate dataframe
data = reviews.copy()

if cluster_col:
    data[cluster_col] = data[cluster_col].fillna("Unknown").astype(str)

if sentiment_col:
    data[sentiment_col] = (
        data[sentiment_col]
        .fillna("Unknown")
        .astype(str)
        .str.title()
    )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown('<div class="brand">✦ CustomerSense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Customer intelligence studio</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Customer Segments",
            "Sentiment Intelligence",
            "Review Explorer",
            "Model Performance"
        ],
        index=0
    )

    st.markdown("---")

    st.markdown("### Filters")

    filtered = data.copy()

    if cluster_col:
        cluster_options = sorted(
            filtered[cluster_col].dropna().unique().tolist()
        )

        selected_clusters = st.multiselect(
            "Customer clusters",
            cluster_options,
            default=cluster_options
        )

        filtered = filtered[
            filtered[cluster_col].isin(selected_clusters)
        ]

    if sentiment_col:
        sentiment_options = sorted(
            filtered[sentiment_col].dropna().unique().tolist()
        )

        selected_sentiments = st.multiselect(
            "Sentiment",
            sentiment_options,
            default=sentiment_options
        )

        filtered = filtered[
            filtered[sentiment_col].isin(selected_sentiments)
        ]

    if rating_col:
        numeric_ratings = pd.to_numeric(
            filtered[rating_col],
            errors="coerce"
        )

        if numeric_ratings.notna().any():
            min_rating = int(numeric_ratings.min())
            max_rating = int(numeric_ratings.max())

            if min_rating < max_rating:
                rating_range = st.slider(
                    "Rating range",
                    min_rating,
                    max_rating,
                    (min_rating, max_rating)
                )

                filtered = filtered[
                    pd.to_numeric(
                        filtered[rating_col],
                        errors="coerce"
                    ).between(*rating_range)
                ]

    st.markdown("---")

    st.caption(f"Showing {len(filtered):,} of {len(data):,} reviews")
    st.caption("NLP • TF-IDF • MiniBatch K-Means")
    st.caption("VADER sentiment • Streamlit")


# =========================================================
# COMMON METRICS
# =========================================================

total_reviews = len(filtered)

total_clusters = (
    filtered[cluster_col].nunique()
    if cluster_col else 0
)

total_sentiments = (
    filtered[sentiment_col].nunique()
    if sentiment_col else 0
)

positive_count = 0
negative_count = 0
neutral_count = 0

if sentiment_col:
    sentiment_values = filtered[sentiment_col].str.lower()

    positive_count = int(
        sentiment_values.str.contains("positive", na=False).sum()
    )

    negative_count = int(
        sentiment_values.str.contains("negative", na=False).sum()
    )

    neutral_count = int(
        sentiment_values.str.contains("neutral", na=False).sum()
    )

positive_percentage = (
    positive_count / total_reviews * 100
    if total_reviews else 0
)

negative_percentage = (
    negative_count / total_reviews * 100
    if total_reviews else 0
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    hero(
        "Customer experience analytics",
        "Understand every customer.",
        "Discover hidden customer segments, understand review sentiment, "
        "and turn feedback into decisions."
    )

    section("Your business at a glance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total reviews",
            number(total_reviews),
            "In the current selection",
            ACCENT
        )

    with c2:
        metric_card(
            "Customer segments",
            number(total_clusters),
            "Detected clusters",
            "#3D6FB6"
        )

    with c3:
        metric_card(
            "Positive reviews",
            f"{positive_percentage:.1f}%",
            f"{number(positive_count)} positive reviews",
            POS
        )

    with c4:
        metric_card(
            "Negative reviews",
            number(negative_count),
            f"{negative_percentage:.1f}% of reviews need attention",
            NEG
        )

    # ---------- Quick insights ----------
    if cluster_col and not filtered.empty:

        section("Quick insights", "Generated from your current filters.")

        i1, i2, i3 = st.columns(3)

        size = filtered[cluster_col].value_counts()
        top_cluster = size.index[0]
        top_share = size.iloc[0] / total_reviews * 100

        with i1:
            insight_card(
                "Largest segment",
                f"Cluster {top_cluster}",
                f"{number(size.iloc[0])} reviews, {top_share:.1f}% of the selection"
            )

        if sentiment_col:
            tmp = filtered.assign(
                _neg=filtered[sentiment_col].str.lower().str.contains("negative", na=False)
            )
            neg_rate = tmp.groupby(cluster_col)["_neg"].mean().sort_values(ascending=False)

            with i2:
                insight_card(
                    "Most negative segment",
                    f"Cluster {neg_rate.index[0]}",
                    f"{neg_rate.iloc[0] * 100:.1f}% of its reviews are negative"
                )

            with i3:
                insight_card(
                    "Happiest segment",
                    f"Cluster {neg_rate.index[-1]}",
                    f"Only {neg_rate.iloc[-1] * 100:.1f}% of its reviews are negative"
                )

    section("Customer feedback overview")

    left, right = st.columns([1.2, 1])

    with left:

        if sentiment_col and not filtered.empty:

            sentiment_counts = (
                filtered[sentiment_col]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Reviews")
            )

            fig = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Customer sentiment distribution"
            )
            fig.update_layout(showlegend=False)

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        else:
            display_empty("Sentiment information is not available.")

    with right:

        if cluster_col and not filtered.empty:

            cluster_counts = (
                filtered[cluster_col]
                .value_counts()
                .reset_index()
            )

            cluster_counts.columns = ["Cluster", "Reviews"]

            fig = px.pie(
                cluster_counts,
                names="Cluster",
                values="Reviews",
                hole=0.62,
                title="Customer segment distribution"
            )
            fig.update_traces(textposition="outside", textinfo="percent")

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        else:
            display_empty("Cluster information is not available.")

    section("How it works")

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("""
        <div class="info-box">
            <h4>Customer segmentation</h4>
            <p>
                TF-IDF features and MiniBatch K-Means identify groups
                of reviews with similar text patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with info2:
        st.markdown("""
        <div class="info-box">
            <h4>Sentiment intelligence</h4>
            <p>
                VADER sentiment analysis identifies positive,
                negative and neutral customer feedback.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Review data source: {review_source}")


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

elif page == "Customer Segments":

    hero(
        "Segment discovery",
        "Customer segments",
        "Explore groups of similar reviews and the themes that define them."
    )

    if not cluster_col:
        display_empty("Cluster column is not available in the review dataset.")
    else:

        cluster_counts = (
            filtered[cluster_col]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = ["Cluster", "Review Count"]

        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                cluster_counts,
                x="Cluster",
                y="Review Count",
                color="Review Count",
                color_continuous_scale=["#CFE8E1", "#0E7C6B"],
                title="Reviews per customer segment"
            )
            fig.update_coloraxes(showscale=False)

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        with c2:
            fig = px.pie(
                cluster_counts,
                names="Cluster",
                values="Review Count",
                hole=0.55,
                title="Segment share"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        section(
            "Segment keywords",
            "The words that best describe each segment."
        )

        if not keywords.empty:

            st.dataframe(
                keywords,
                use_container_width=True,
                hide_index=True
            )

        else:
            display_empty("Cluster keyword report was not found.")

        section("Segment summary")

        st.dataframe(
            cluster_counts,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# SENTIMENT INTELLIGENCE
# =========================================================

elif page == "Sentiment Intelligence":

    hero(
        "Customer emotion analysis",
        "Sentiment intelligence",
        "Measure customer satisfaction and spot negative feedback patterns."
    )

    if not sentiment_col:

        display_empty("Sentiment column was not found in the review dataset.")

    else:

        a, b, c = st.columns(3)

        with a:
            metric_card(
                "Positive",
                number(positive_count),
                "Positive customer feedback",
                POS
            )

        with b:
            metric_card(
                "Neutral",
                number(neutral_count),
                "Neutral customer feedback",
                NEU
            )

        with c:
            metric_card(
                "Negative",
                number(negative_count),
                "Negative customer feedback",
                NEG
            )

        sentiment_counts = (
            filtered[sentiment_col]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = ["Sentiment", "Reviews"]

        section("Sentiment breakdown")

        left, right = st.columns(2)

        with left:

            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Reviews",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment share"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        with right:

            fig = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment comparison"
            )
            fig.update_layout(showlegend=False)

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        if compound_col and not filtered.empty:

            section(
                "Sentiment score distribution",
                "VADER compound score: -1 is most negative, +1 is most positive."
            )

            score_df = filtered.copy()
            score_df[compound_col] = pd.to_numeric(
                score_df[compound_col],
                errors="coerce"
            )

            fig = px.histogram(
                score_df.dropna(subset=[compound_col]),
                x=compound_col,
                nbins=40,
                color=sentiment_col,
                color_discrete_map=SENTIMENT_COLORS,
                title="Compound score distribution"
            )

            st.plotly_chart(
                chart_layout(fig, 380),
                use_container_width=True
            )

        section("Sentiment by customer segment")

        if (
            cluster_col
            and not filtered.empty
        ):

            cross_data = pd.crosstab(
                filtered[cluster_col],
                filtered[sentiment_col]
            )

            fig = px.bar(
                cross_data.reset_index().melt(
                    id_vars=cluster_col,
                    var_name="Sentiment",
                    value_name="Reviews"
                ),
                x=cluster_col,
                y="Reviews",
                color="Sentiment",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment distribution across clusters"
            )

            st.plotly_chart(
                chart_layout(fig, 420),
                use_container_width=True
            )

        elif not sentiment_cluster.empty:

            st.dataframe(
                sentiment_cluster,
                use_container_width=True,
                hide_index=True
            )

        else:
            display_empty("Cluster-level sentiment information is unavailable.")


# =========================================================
# REVIEW EXPLORER
# =========================================================

elif page == "Review Explorer":

    hero(
        "Customer voice",
        "Review explorer",
        "Search and read individual customer reviews."
    )

    if not text_col:

        display_empty("Review text column was not found.")

    else:

        search_text = st.text_input(
            "Search customer reviews",
            placeholder="Type a word or phrase..."
        )

        review_data = filtered.copy()

        if search_text.strip():

            review_data = review_data[
                review_data[text_col]
                .fillna("")
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    regex=False
                )
            ]

        head_l, head_r = st.columns([3, 1])

        with head_l:
            st.markdown(
                f'<span class="pill">{number(len(review_data))} reviews found</span>',
                unsafe_allow_html=True
            )

        display_columns = []

        for col in [
            text_col,
            cluster_col,
            sentiment_col,
            rating_col
        ]:

            if col and col not in display_columns:
                display_columns.append(col)

        with head_r:
            st.download_button(
                "Download results",
                review_data.to_csv(index=False).encode("utf-8"),
                file_name="filtered_reviews.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.write("")

        if display_columns:

            st.dataframe(
                review_data[display_columns],
                use_container_width=True,
                hide_index=True,
                height=480
            )

        section("Review length analysis")

        review_lengths = (
            review_data[text_col]
            .fillna("")
            .astype(str)
            .str.len()
        )

        length_df = pd.DataFrame({
            "Review Length": review_lengths
        })

        if not length_df.empty:

            fig = px.histogram(
                length_df,
                x="Review Length",
                nbins=35,
                title="Customer review length distribution",
                color_discrete_sequence=[ACCENT]
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    hero(
        "Machine learning lab",
        "Model performance",
        "Check clustering quality and inspect evaluation results."
    )

    section("Model configuration")

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_card("Feature extraction", "TF-IDF", "Text vectorization", ACCENT)

    with m2:
        metric_card("Clustering model", "MiniBatch K-Means", "Unsupervised learning", "#3D6FB6")

    with m3:
        metric_card("Sentiment engine", "VADER", "Lexicon-based sentiment analysis", NEU)

    section("Evaluation results")

    if not evaluation.empty:

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        metric_col = find_column(
            evaluation,
            ["Metric", "metric", "Name"]
        )

        value_col = find_column(
            evaluation,
            ["Value", "value", "Score"]
        )

        if metric_col and value_col:

            chart_data = evaluation[[metric_col, value_col]].copy()

            chart_data[value_col] = pd.to_numeric(
                chart_data[value_col],
                errors="coerce"
            )

            chart_data = chart_data.dropna()

            if not chart_data.empty:

                fig = px.bar(
                    chart_data,
                    x=metric_col,
                    y=value_col,
                    color=value_col,
                    color_continuous_scale=["#CFE8E1", "#0E7C6B"],
                    title="Model evaluation metrics"
                )
                fig.update_coloraxes(showscale=False)

                st.plotly_chart(
                    chart_layout(fig, 400),
                    use_container_width=True
                )

    else:

        display_empty(
            "No model evaluation report found. "
            "Expected outputs/model_evaluation_metrics.csv"
        )

    section("Clustering analysis")

    col1, col2 = st.columns(2)

    with col1:

        elbow_path = OUTPUTS / "elbow_method.png"

        if elbow_path.exists():
            st.image(
                str(elbow_path),
                caption="Elbow method",
                use_container_width=True
            )
        else:
            display_empty("Elbow method image not found.")

    with col2:

        silhouette_path = OUTPUTS / "silhouette_analysis.png"

        if silhouette_path.exists():
            st.image(
                str(silhouette_path),
                caption="Silhouette analysis",
                use_container_width=True
            )
        else:
            display_empty("Silhouette analysis image not found.")

    section("Project methodology")

    st.markdown("""
    <div class="info-box">
        <div class="step"><div class="step-n">1</div><div>
            <h4>Text preprocessing</h4>
            <p>Clean and normalize customer feedback using NLP techniques.</p></div></div>
        <div class="step"><div class="step-n">2</div><div>
            <h4>TF-IDF feature extraction</h4>
            <p>Convert text documents into numerical feature vectors.</p></div></div>
        <div class="step"><div class="step-n">3</div><div>
            <h4>MiniBatch K-Means</h4>
            <p>Group similar customer reviews into clusters.</p></div></div>
        <div class="step"><div class="step-n">4</div><div>
            <h4>Sentiment analysis</h4>
            <p>Classify feedback as positive, negative or neutral using VADER.</p></div></div>
        <div class="step"><div class="step-n">5</div><div>
            <h4>Evaluation</h4>
            <p>Inspect clustering metrics and understand model performance.</p></div></div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:12px; color:#6F817C; font-size:14px;">
        <b>CustomerSense AI</b> &nbsp;|&nbsp;
        Customer feedback intelligence platform<br>
        <small>Built with Python, NLP, machine learning and Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)