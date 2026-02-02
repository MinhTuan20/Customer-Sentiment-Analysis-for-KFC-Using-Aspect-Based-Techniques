\
import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# ============================================
# CONFIG
# ============================================
DEFAULT_LABEL_MAP = {0: "None", 1: "Positive", 2: "Negative", 3: "Neutral"}

st.set_page_config(page_title="KFC Hanoi — ABSA + Map", layout="wide")
st.title("🍗 KFC Hanoi — Aspect-Based Sentiment & Branch Map")

# ============================================
# HELPER FOR review_time_en (relative strings)
# ============================================
def build_time_order():
    """
    Map '1 ngày trước', '2 tuần trước', '3 tháng trước', ... -> số ngày tương ứng.
    Dùng để sort & filter theo kiểu: từ mốc được chọn cho tới hiện tại.
    """
    order = {}
    # 1–6 ngày
    for i in range(1, 7):
        order[f"{i} ngày trước"] = i
    # 1–4 tuần (~7 ngày/tuần)
    for i in range(1, 5):
        order[f"{i} tuần trước"] = i * 7
    # 1–6 tháng (~30 ngày/tháng)
    for i in range(1, 7):
        order[f"{i} tháng trước"] = i * 30
    return order

TIME_ORDER = build_time_order()

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource(show_spinner=False)
def load_models(p: str):
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(f"Model not found: {pth.resolve()}")

    obj = joblib.load(p)
    pipelines = obj["pipelines"]
    aspects = obj.get("aspects", list(pipelines.keys()))
    label_map = obj.get("label_map", DEFAULT_LABEL_MAP)

    clean_map = {int(k): str(v) for k, v in label_map.items() if str(k).isdigit()}
    for k, v in DEFAULT_LABEL_MAP.items():
        clean_map.setdefault(k, v)

    return pipelines, aspects, clean_map


# ========================
# SIDEBAR CONFIG
# ========================
st.sidebar.header("⚙️ Configuration")

excel_path = st.sidebar.text_input(
    "Excel file path",
    value=r"D:\Du lieu\Tài Liệu Đại Học\Trí Tuệ Nhân Tạo\AI FINAL GR2\FINAL (Code + Data)\Data\KFC HN Customer Feedback .xlsx"#Data=============================================================================
)

model_path = st.sidebar.text_input(
    "Model path",
    value="kfc_absa_aspects.pkl"
)

try:
    PIPELINES, ASPECTS, LABEL_MAP = load_models(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# ============================================
# LOAD EXCEL
# ============================================
@st.cache_data(show_spinner=False)
def load_excel(fp: str) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

try:
    df = load_excel(excel_path)
except Exception as e:
    st.error(f"❌ Excel read error: {e}")
    st.stop()


# ============================================
# VALIDATION
# ============================================
required_cols = ["address", "location"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns in Excel file: {missing}")
    st.stop()

def split_lat_long(val):
    try:
        lat, lon = map(float, str(val).split(","))
        return lat, lon
    except Exception:
        return None, None

df["latitude"], df["longitude"] = zip(*df["location"].apply(split_lat_long))
data_map = df.dropna(subset=["latitude", "longitude", "address"]).copy()

if data_map.empty:
    st.error("❌ No valid branch data found.")
    st.stop()


# ============================================
# SENTIMENT PREDICT
# ============================================
def predict_aspect(text, aspect_key):
    if aspect_key not in PIPELINES:
        return "None"
    pred = PIPELINES[aspect_key].predict([text])[0]
    try:
        pred = int(pred)
    except Exception:
        pass
    return LABEL_MAP.get(pred, "None")


# ============================================
# SUMMARY SENTIMENT
# ============================================
def summarize_branch(df_branch, aspects):
    text_col = (
        "translated_text" if "translated_text" in df_branch.columns else
        "review_text" if "review_text" in df_branch.columns else None
    )
    if not text_col:
        return pd.DataFrame()

    rows = []
    texts = df_branch[text_col].astype(str).tolist()

    for a in aspects:
        counter = {"Positive": 0, "Negative": 0, "Neutral": 0, "None": 0}
        for t in texts:
            s = predict_aspect(t, a)
            counter[s] += 1

        rows.append({
            "Aspect": a.replace("_", " ").title(),
            "Positive": counter["Positive"],
            "Negative": counter["Negative"],
            "Neutral": counter["Neutral"],
            "None": counter["None"],
            "Total": counter["Positive"] + counter["Negative"] + counter["Neutral"],
        })

    return pd.DataFrame(rows).set_index("Aspect").sort_values("Total", ascending=False)


# ============================================
# KEYWORD EXTRACTION
# ============================================
def extract_keywords(texts, top_k=20):
    if len(texts) == 0:
        return pd.DataFrame(columns=["keyword", "count"])

    stopwords = [
        "the","and","for","with","that","this","was","were","from","but","are","have",
        "has","had","will","would","can","could","i","you","we","they","he","she","it",
        "to","in","on","at","of","a","an","is","am","be"
    ]

    cv = CountVectorizer(stop_words=stopwords, max_features=30)
    mat = cv.fit_transform(texts)
    freq = mat.sum(axis=0)

    words = [(w, int(freq[0, i])) for w, i in cv.vocabulary_.items()]
    df_kw = pd.DataFrame(words, columns=["keyword", "count"])
    return df_kw.sort_values("count", ascending=False)


# ============================================
# UI — TWO TABS
# ============================================
tab1, tab2 = st.tabs(["🗺️ Branch Map", "🔍 Check One Sentence"])



# ==========================================================
# TAB 1 — MAP + FEEDBACK + SENTIMENT CHART + KEYWORDS
# ==========================================================
with tab1:

    st.subheader("📍 Map of 15 KFC Hanoi Branches")

    branches = data_map["address"].value_counts().index.tolist()
    branch = st.selectbox("Select a branch:", ["(All)"] + branches)

    # Auto zoom
    if branch != "(All)":
        df_b = data_map[data_map["address"] == branch]
        if not df_b.empty:
            center = [float(df_b["latitude"].iloc[0]), float(df_b["longitude"].iloc[0])]
            zoom = 16
        else:
            center = [data_map["latitude"].mean(), data_map["longitude"].mean()]
            zoom = 12
    else:
        center = [data_map["latitude"].mean(), data_map["longitude"].mean()]
        zoom = 12

    m = folium.Map(location=center, zoom_start=zoom)
    mc = MarkerCluster().add_to(m)

    for _, row in data_map.iterrows():
        popup = f"<b>{row['address']}</b>"
        review = df[df["address"] == row["address"]]["review_text"].dropna().astype(str)
        if len(review):
            popup += f"<br><i>Example:</i><br>{review.iloc[0][:200]}"
        folium.Marker([row["latitude"], row["longitude"]], popup=popup).add_to(mc)

    st_folium(m, width=900, height=520)

    # Feedback
    st.subheader("📄 Branch feedback")
    df_branch = df if branch == "(All)" else df[df["address"] == branch]
    st.dataframe(df_branch, use_container_width=True)

    # Summary table (không lọc theo date, để tổng quan)
    st.markdown("---")
    st.subheader("📊 Sentiment summary by aspect")

    summary = summarize_branch(df_branch, ASPECTS)
    st.dataframe(summary, use_container_width=True)

    # =============================================
    # Chart + filter Aspect, Sentiment, Review time 
    # =============================================
    st.subheader("📈 Sentiment chart by aspect")

    # Dữ liệu dùng cho chart & keywords (sẽ lọc thêm theo review_time_en)
    df_for_chart = df_branch.copy()
    sel = None

    has_date = "review_time_en" in df_for_chart.columns

    if has_date:
        # Các giá trị review_time_en thực tế có trong data
        all_dates = (
            df_for_chart["review_time_en"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        # Sort theo độ xa (dựa trên TIME_ORDER)
        all_dates = sorted(all_dates, key=lambda x: TIME_ORDER.get(x, 9999))

        # 3 filter: Aspect | Sentiment | Review time
        colA, colB, colC = st.columns([1, 1, 1.6])

        # thêm option None cho date
        date_options = ["(None)"] + all_dates

        selected_date = colC.selectbox(
            "From this time up to now:",
            options=date_options,
            index=0  # mặc định = None, không lọc
        )

        if selected_date != "(None)":
            max_age = TIME_ORDER.get(selected_date, None)
            if max_age is not None:
                ages = df_for_chart["review_time_en"].astype(str).map(TIME_ORDER)
                df_for_chart = df_for_chart[ages <= max_age]
    else:
        colA, colB = st.columns(2)

    # Tính summary cho CHART dựa trên df_for_chart (đã lọc date nếu có)
    summary_chart = summarize_branch(df_for_chart, ASPECTS)

    if summary_chart is None or summary_chart.empty:
        st.info("No data available to plot.")
    else:
        s2 = summary_chart.reset_index()

        # Ensure numeric types
        for c in ["Positive", "Negative", "Neutral"]:
            s2[c] = pd.to_numeric(s2.get(c, 0), errors="coerce").fillna(0).astype(int)

        # ===== Filters Aspect & Sentiment (áp dụng cho CẢ CHART) =====
        asp = colA.selectbox("Aspect:", ["(None)"] + list(s2["Aspect"]))
        senti = colB.selectbox("Sentiment:", ["(None)", "Positive", "Negative", "Neutral"])

        # Lọc dữ liệu theo Aspect
        s2_plot = s2.copy()
        if asp != "(None)":
            s2_plot = s2_plot[s2_plot["Aspect"] == asp]

        # Chọn cột y theo Sentiment
        if senti == "(None)":
            y_cols = ["Positive", "Negative", "Neutral"]
        else:
            y_cols = [senti]

        custom_colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}

        fig = px.bar(
            s2_plot,
            x="Aspect",
            y=y_cols,
            barmode="group",
            title="Sentiment breakdown by aspect",
            color_discrete_map=custom_colors
        )
        st.plotly_chart(fig, use_container_width=True)

        # Nếu cả Aspect & Sentiment đều được chọn cụ thể -> dùng cho keywords
        if asp != "(None)" and senti != "(None)":
            sel = (asp, senti)
    
    # Keywords section
    st.subheader("🔎 Top keywords")

    if sel:
        asp, senti = sel
        st.markdown(f"### Aspect: **{asp}** — Sentiment: **{senti}**")

        asp_key = asp.lower().replace(" ", "_")
        rev_col = "translated_text" if "translated_text" in df_for_chart.columns else "review_text"

        filtered = []
        for _, r in df_for_chart.iterrows():
            t = str(r[rev_col])
            pred = predict_aspect(t, asp_key)
            if pred == senti:
                filtered.append(t)

        if not filtered:
            st.info("No reviews found for this group.")
        else:
            kw_df = extract_keywords(filtered)

            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.markdown("### ⭐ Top Keywords")
                st.dataframe(kw_df, height=350, use_container_width=True)

            with col2:
                st.markdown("### 💬 Sample Reviews")
                for t in filtered[:20]:
                    st.markdown(f"- {t}")


# ==========================================================
# TAB 2 — SINGLE TEXT CHECK
# ==========================================================
with tab2:
    st.subheader("🔍 Check sentiment for each aspect")

    txt = st.text_area("Enter a sentence:", height=140)

    if st.button("Predict sentiment"):
        rows = []
        for a in ASPECTS:
            pred = predict_aspect(txt, a)
            if pred != "None":
                rows.append([a.replace("_"," ").title(), pred])

        if rows:
            st.table(pd.DataFrame(rows, columns=["Aspect","Sentiment"]))
        else:
            st.info("No aspect was detected.")

