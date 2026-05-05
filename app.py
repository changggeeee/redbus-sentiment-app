import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ======================
# CONFIG & STYLING
# ======================
st.set_page_config(
    page_title="RedBus Sentiment Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background: #0f0f13; color: #e8e8f0; }
    [data-testid="stSidebar"] { background: #16161d !important; border-right: 1px solid #2a2a3a; }
    [data-testid="stSidebar"] * { color: #e8e8f0 !important; }
    [data-testid="metric-container"] {
        background: #1a1a24; border: 1px solid #2a2a3a;
        border-radius: 12px; padding: 20px; transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover { border-color: #ff4b4b; }
    [data-testid="stMetricValue"] {
        color: #ff4b4b !important; font-size: 2rem !important;
        font-weight: 800 !important; font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: #888899 !important; font-size: 0.8rem !important;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #ff4b4b;
        text-transform: uppercase; letter-spacing: 0.15em;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #2a2a3a;
    }
    .hero-header {
        background: linear-gradient(135deg, #1a0a0a 0%, #1a1a24 50%, #0a0a1a 100%);
        border: 1px solid #2a2a3a; border-radius: 16px;
        padding: 2rem 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
    }
    .hero-header::before {
        content: ''; position: absolute; top: 0; right: 0; width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(255,75,75,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0; line-height: 1.2; }
    .hero-title span { color: #ff4b4b; }
    .hero-subtitle { color: #888899; margin: 0.5rem 0 0 0; font-size: 0.95rem; }
    .hero-badge {
        display: inline-block; background: rgba(255,75,75,0.12);
        border: 1px solid rgba(255,75,75,0.3); color: #ff4b4b;
        font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem;
    }
    .insight-card {
        background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin-bottom: 0.8rem; border-left: 3px solid #ff4b4b;
    }
    .insight-card p { color: #c8c8d8; margin: 0; font-size: 0.9rem; line-height: 1.6; }
    .insight-card strong { color: #ffffff; }
    .about-card {
        background: linear-gradient(135deg, #1a1a24, #16161d);
        border: 1px solid #2a2a3a; border-radius: 16px; padding: 2rem;
    }
    .about-card h3 { color: #ff4b4b; margin-top: 0; }
    .about-card p, .about-card li { color: #c8c8d8; line-height: 1.7; }
    .skill-badge {
        display: inline-block; background: #1a1a24; border: 1px solid #2a2a3a;
        color: #c8c8d8; padding: 4px 12px; border-radius: 6px;
        font-size: 0.8rem; margin: 3px; font-family: 'JetBrains Mono', monospace;
    }
    hr { border-color: #2a2a3a; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ======================
# PLOTLY THEME
# ======================
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color='#c8c8d8', size=12),
    colorway=['#ff4b4b', '#4b9eff', '#4bff9f', '#ffb84b', '#c44bff'],
    xaxis=dict(gridcolor='#2a2a3a', linecolor='#2a2a3a', tickfont=dict(color='#888899')),
    yaxis=dict(gridcolor='#2a2a3a', linecolor='#2a2a3a', tickfont=dict(color='#888899')),
    title=dict(font=dict(size=14, color='#e8e8f0'), x=0),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c8c8d8')),
    margin=dict(l=20, r=20, t=50, b=20),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

COLORS = {
    'positive':       '#4bff9f',
    'mixed positive': '#a8ffcc',
    'neutral':        '#4b9eff',
    'mixed negative': '#ffaa4b',
    'negative':       '#ff4b4b',
}

# ======================
# HELPER: anti-NaN unique list
# ======================
def safe_unique(series):
    """Kembalikan list unique values tanpa NaN, tanpa string 'nan'."""
    result = []
    for x in series.unique():
        try:
            if x != x:          # NaN float check
                continue
            if str(x) in ('nan', 'NaN', 'None', ''):
                continue
            result.append(x)
        except Exception:
            continue
    return result

# ======================
# LOAD & CLEAN DATA
# ======================
def clean_dataframe(df):
    # Rename kolom ke nama standar
    rename_map = {
        'ReviewText': 'Review', 'reviewtext': 'Review',
        'ReviewDate': 'date',   'reviewdate': 'date',
        'userName':   'userName','username':  'userName',
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # Bersihkan Rating
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df[df['Rating'].notna()]
        df['Rating'] = df['Rating'].astype(int)

    # Normalisasi SentimentCategory
    if 'SentimentCategory' in df.columns:
        df = df[df['SentimentCategory'].notna()]
        df = df[df['SentimentCategory'].astype(str) != 'nan']

        def norm_sentiment(s):
            s = str(s).strip().lower()
            if s == 'mixed negative':  return 'mixed negative'
            if s == 'mixed positive':  return 'mixed positive'
            if s == 'negative':        return 'negative'
            if s == 'positive':        return 'positive'
            if s == 'neutral':         return 'neutral'
            return 'neutral'

        df['SentimentCategory'] = df['SentimentCategory'].apply(norm_sentiment)

    # Normalisasi SentimentBucket
    if 'SentimentBucket' in df.columns:
        bucket_map = {
            '-1.0 to -0.5': 'Very Negative',
            '-0.5 to 0.0':  'Negative',
            '0.0 to 0.49':  'Positive',
            '0.5 to 1.0':   'Very Positive',
        }
        df['SentimentBucket'] = df['SentimentBucket'].apply(
            lambda s: bucket_map.get(str(s).strip(), str(s).strip())
        )

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df.reset_index(drop=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('redbus_sentiment.csv')
        df = clean_dataframe(df)
        return df, None
    except FileNotFoundError:
        np.random.seed(42)
        n = 500
        sentiments = np.random.choice(['positive', 'negative', 'neutral'], n, p=[0.55, 0.30, 0.15])
        ratings = []
        for s in sentiments:
            if s == 'positive':   ratings.append(int(np.random.choice([4, 5], p=[0.3, 0.7])))
            elif s == 'negative': ratings.append(int(np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])))
            else:                 ratings.append(int(np.random.choice([3, 4], p=[0.6, 0.4])))

        score_map = {'positive': (0.3, 1.0), 'negative': (-1.0, -0.1), 'neutral': (-0.1, 0.3)}
        scores = [round(float(np.random.uniform(*score_map[s])), 4) for s in sentiments]

        buckets = []
        for sc in scores:
            if sc >= 0.5:    buckets.append('Very Positive')
            elif sc >= 0.1:  buckets.append('Positive')
            elif sc >= -0.1: buckets.append('Neutral')
            elif sc >= -0.5: buckets.append('Negative')
            else:            buckets.append('Very Negative')

        pos_w = ['great service','comfortable seats','on time','clean bus','friendly driver','good experience','easy booking']
        neg_w = ['late arrival','bad service','dirty bus','rude driver','app crash','slow booking','expensive']
        neu_w = ['average service','okay journey','normal experience','decent bus','fine']
        reviews = [np.random.choice(pos_w if s=='positive' else neg_w if s=='negative' else neu_w) for s in sentiments]
        dates = pd.date_range('2023-01-01', '2024-12-31', periods=n)

        df = pd.DataFrame({
            'userName': ['Demo User'] * n,
            'Rating': ratings,
            'date': dates,
            'Review': reviews,
            'SentimentScore': scores,
            'SentimentCategory': list(sentiments),
            'SentimentBucket': buckets,
        })
        return df, "⚠️ File `redbus_sentiment.csv` tidak ditemukan — menampilkan **data demo**."


df, data_warning = load_data()

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0;'>
        <div style='font-size:1.5rem;font-weight:800;color:#ff4b4b;'>🚌 RedBus</div>
        <div style='font-size:0.75rem;color:#888899;text-transform:uppercase;letter-spacing:0.15em;margin-top:2px;'>Sentiment Dashboard</div>
    </div>
    <hr style='border-color:#2a2a3a;margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["📊 Dashboard", "🔍 Eksplorasi", "💡 Insight", "🤖 Machine Learning", "👤 About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2a2a3a;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#888899;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;'>Filter Global</div>", unsafe_allow_html=True)

    opt_sentiment = safe_unique(df['SentimentCategory'])
    sentiment_filter = st.multiselect("Sentimen", options=opt_sentiment, default=opt_sentiment)

    opt_rating = sorted(safe_unique(df['Rating']))
    rating_filter = st.multiselect("Rating", options=opt_rating, default=opt_rating)

    st.markdown("<hr style='border-color:#2a2a3a;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#555566;line-height:1.6;'>
        Built with Streamlit + Plotly<br>
        NLP · Data Visualization<br>
        <span style='color:#ff4b4b;'>Portfolio Project 2024</span>
    </div>
    """, unsafe_allow_html=True)

# ======================
# APPLY FILTER
# ======================
filtered_df = df[
    df['SentimentCategory'].isin(sentiment_filter) &
    df['Rating'].isin(rating_filter)
].copy()

total      = len(filtered_df)
positive_n = (filtered_df['SentimentCategory'] == 'positive').sum()
negative_n = (filtered_df['SentimentCategory'] == 'negative').sum()
# neutral tidak dipakai (4 kelas)
avg_rating = filtered_df['Rating'].mean() if total > 0 else 0
avg_score  = filtered_df['SentimentScore'].mean() if ('SentimentScore' in filtered_df.columns and total > 0) else 0

# ======================
# PAGE: DASHBOARD
# ======================
if page == "📊 Dashboard":

    if data_warning:
        st.warning(data_warning)

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Portfolio · NLP Project</div>
        <h1 class="hero-title">Analisis Sentimen<br>Ulasan Aplikasi <span>RedBus</span></h1>
        <p class="hero-subtitle">Memahami persepsi pengguna melalui machine learning & natural language processing</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Ulasan", f"{total:,}")
    with c2:
        pct = round(positive_n/total*100, 1) if total > 0 else 0
        st.metric("Positif", f"{pct}%", delta=f"{positive_n} ulasan")
    with c3:
        nct = round(negative_n/total*100, 1) if total > 0 else 0
        st.metric("Negatif", f"{nct}%", delta=f"-{negative_n} ulasan", delta_color="inverse")
    with c4: st.metric("Avg Rating", f"⭐ {avg_rating:.2f}")
    with c5:
        score_label = "Positif" if avg_score > 0 else "Negatif"
        st.metric("Avg Sentiment Score", f"{avg_score:.3f}", delta=score_label)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Distribusi Utama</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sc = filtered_df['SentimentCategory'].value_counts().reset_index()
        sc.columns = ['Sentimen', 'Jumlah']
        fig1 = px.pie(sc, names='Sentimen', values='Jumlah', title='Distribusi Sentimen',
                      hole=0.45, color='Sentimen', color_discrete_map=COLORS)
        fig1.update_traces(textfont=dict(color='white'))
        apply_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        rc = filtered_df['Rating'].value_counts().sort_index().reset_index()
        rc.columns = ['Rating', 'Jumlah']
        fig2 = px.bar(rc, x='Rating', y='Jumlah', title='Distribusi Rating',
                      color='Rating', color_continuous_scale=['#ff4b4b', '#ffb84b', '#4bff9f'])
        apply_theme(fig2)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        if 'SentimentBucket' in filtered_df.columns:
            bucket_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            bc = filtered_df['SentimentBucket'].value_counts().reindex(bucket_order, fill_value=0).reset_index()
            bc.columns = ['Bucket', 'Jumlah']
            fig3 = px.bar(bc, x='Bucket', y='Jumlah', title='Distribusi Skor Sentimen',
                          color='Jumlah', color_continuous_scale=['#ff4b4b', '#ffb84b', '#4bff9f'])
            apply_theme(fig3)
            fig3.update_coloraxes(showscale=False)
            fig3.update_xaxes(tickangle=-20)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Analisis Mendalam</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        fig4 = px.histogram(filtered_df, x='Rating', color='SentimentCategory',
                            barmode='group', title='Korelasi Rating & Sentimen',
                            color_discrete_map=COLORS, nbins=5)
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        if 'date' in filtered_df.columns and filtered_df['date'].notna().sum() > 0:
            trend = (
                filtered_df
                .assign(Bulan=filtered_df['date'].dt.to_period('M').astype(str))
                .groupby(['Bulan', 'SentimentCategory'])
                .size().reset_index(name='Jumlah')
            )
            fig5 = px.line(trend, x='Bulan', y='Jumlah', color='SentimentCategory',
                           title='Tren Sentimen per Bulan', color_discrete_map=COLORS, markers=True)
            apply_theme(fig5)
            fig5.update_traces(line=dict(width=2.5))
            fig5.update_xaxes(tickangle=-45)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Kolom tanggal tidak tersedia untuk analisis tren")

    if 'SentimentScore' in filtered_df.columns:
        st.markdown('<div class="section-header">Distribusi Skor Sentimen</div>', unsafe_allow_html=True)
        fig6 = px.histogram(filtered_df, x='SentimentScore', color='SentimentCategory',
                            title='Distribusi Skor (-1 hingga +1)', nbins=40,
                            opacity=0.8, color_discrete_map=COLORS, barmode='overlay')
        apply_theme(fig6)
        st.plotly_chart(fig6, use_container_width=True)

# ======================
# PAGE: EKSPLORASI
# ======================
elif page == "🔍 Eksplorasi":

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Eksplorasi Data</div>
        <h1 class="hero-title">Jelajahi <span>Dataset</span></h1>
        <p class="hero-subtitle">Filter, cari, dan ekspor data ulasan sesuai kebutuhan analisis</p>
    </div>
    """, unsafe_allow_html=True)

    if 'Review' in filtered_df.columns:
        st.markdown('<div class="section-header">Kata Paling Sering Muncul</div>', unsafe_allow_html=True)
        col_w1, col_w2 = st.columns(2)

        def get_word_freq(df_sub, sentiment, top=15):
            sub = df_sub[df_sub['SentimentCategory'] == sentiment]['Review'].dropna()
            words = ' '.join(sub.astype(str).str.lower()).split()
            stop = {'the','a','an','is','in','it','of','and','to','for','bus','redbus',
                    'that','this','was','are','very','i','my','with','so','not','be','have','but','on','at'}
            words = [w for w in words if w not in stop and len(w) > 2]
            return Counter(words).most_common(top)

        with col_w1:
            if 'positive' in filtered_df['SentimentCategory'].values:
                data = get_word_freq(filtered_df, 'positive')
                if data:
                    wdf = pd.DataFrame(data, columns=['Kata', 'Frekuensi'])
                    fig_w1 = px.bar(wdf, x='Frekuensi', y='Kata', orientation='h',
                                    title='Top Kata — Ulasan Positif',
                                    color='Frekuensi', color_continuous_scale=['#1a3d2b', '#4bff9f'])
                    apply_theme(fig_w1)
                    fig_w1.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_w1, use_container_width=True)

        with col_w2:
            if 'negative' in filtered_df['SentimentCategory'].values:
                data = get_word_freq(filtered_df, 'negative')
                if data:
                    wdf2 = pd.DataFrame(data, columns=['Kata', 'Frekuensi'])
                    fig_w2 = px.bar(wdf2, x='Frekuensi', y='Kata', orientation='h',
                                    title='Top Kata — Ulasan Negatif',
                                    color='Frekuensi', color_continuous_scale=['#3d1a1a', '#ff4b4b'])
                    apply_theme(fig_w2)
                    fig_w2.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_w2, use_container_width=True)

    st.markdown('<div class="section-header">Heatmap Rating vs Sentimen</div>', unsafe_allow_html=True)
    if total > 0:
        hm = filtered_df.groupby(['Rating', 'SentimentCategory']).size().unstack(fill_value=0)
        fig_hm = px.imshow(hm, title='Rating × Sentimen (jumlah ulasan)',
                           color_continuous_scale=['#0f0f13', '#ff4b4b'], text_auto=True)
        apply_theme(fig_hm)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<div class="section-header">Tabel Data Ulasan</div>', unsafe_allow_html=True)
    search = st.text_input("🔍 Cari kata dalam ulasan...", placeholder="Ketik keyword...")
    display_df = filtered_df.copy()
    if search and 'Review' in display_df.columns:
        display_df = display_df[display_df['Review'].astype(str).str.contains(search, case=False, na=False)]

    st.caption(f"Menampilkan {min(100, len(display_df))} dari {len(display_df):,} ulasan")
    st.dataframe(display_df.head(100), use_container_width=True, height=400)

    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        buf = io.StringIO()
        filtered_df.to_csv(buf, index=False)
        st.download_button("⬇️ Download CSV (Filtered)", buf.getvalue(),
                           file_name="redbus_filtered.csv", mime="text/csv",
                           use_container_width=True)
    with col_e2:
        st.info(f"📦 {len(filtered_df):,} baris × {len(filtered_df.columns)} kolom siap diexport")

# ======================
# PAGE: INSIGHT
# ======================
elif page == "💡 Insight":

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Business Insight</div>
        <h1 class="hero-title">Temuan & <span>Rekomendasi</span></h1>
        <p class="hero-subtitle">Insight yang dapat dieksekusi berdasarkan analisis data ulasan pengguna</p>
    </div>
    """, unsafe_allow_html=True)

    if total == 0:
        st.warning("Filter terlalu ketat, tidak ada data untuk dianalisis.")
    else:
        pos_pct = round(positive_n / total * 100, 1)
        neg_pct = round(negative_n / total * 100, 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            health = "Baik 🟢" if pos_pct > 60 else "Perlu Perhatian 🟡" if pos_pct > 40 else "Kritis 🔴"
            st.metric("Sentiment Health", health)
        with col2:
            nps = pos_pct - neg_pct
            st.metric("NPS Proxy Score", f"{nps:.1f}", delta="positif" if nps > 0 else "negatif")
        with col3:
            st.metric("Avg Rating", f"{avg_rating:.2f} / 5.0")

        st.markdown('<div class="section-header">Temuan Kunci</div>', unsafe_allow_html=True)
        for ins in [
            f"<strong>Sentimen dominan positif ({pos_pct}%)</strong> — Mayoritas pengguna puas dengan layanan RedBus.",
            f"<strong>Ulasan negatif sebesar {neg_pct}%</strong> — Perlu investigasi mendalam terhadap keluhan spesifik.",
            f"<strong>Rata-rata rating {avg_rating:.2f}/5.0</strong> — {'Di atas rata-rata, pertahankan.' if avg_rating >= 3.5 else 'Di bawah ekspektasi, perlu perbaikan.'}",
            "<strong>Korelasi rating-sentimen kuat</strong> — Rating 1-2 bintang hampir selalu bersentimen negatif.",
        ]:
            st.markdown(f'<div class="insight-card"><p>{ins}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Rekomendasi Aksi</div>', unsafe_allow_html=True)
        for priority, actions in {
            "🔴 Prioritas Tinggi": [
                "Lakukan root cause analysis pada ulasan negatif rating 1-2 bintang",
                "Buat sistem alert jika sentimen negatif meningkat >5% dalam seminggu",
            ],
            "🟡 Prioritas Sedang": [
                "Kumpulkan feedback terstruktur untuk mengidentifikasi pain points",
                "Analisis kata kunci negatif untuk panduan pelatihan customer service",
            ],
            "🟢 Pertahankan": [
                "Amplifikasi aspek positif dalam materi marketing",
                "Bagikan insight sentimen ke tim produk setiap bulan",
            ]
        }.items():
            st.markdown(f"**{priority}**")
            for action in actions:
                st.markdown(f'<div class="insight-card" style="border-left-color:#2a2a3a;"><p>→ {action}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Sentiment Health Gauge</div>', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pos_pct,
            title={'text': "Tingkat Sentimen Positif (%)", 'font': {'color': '#e8e8f0', 'family': 'Plus Jakarta Sans'}},
            delta={'reference': 60, 'decreasing': {'color': '#ff4b4b'}, 'increasing': {'color': '#4bff9f'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#888899'},
                'bar': {'color': '#ff4b4b'},
                'steps': [
                    {'range': [0, 40],  'color': '#2a1a1a'},
                    {'range': [40, 65], 'color': '#2a2a1a'},
                    {'range': [65, 100],'color': '#1a2a1a'},
                ],
                'threshold': {'line': {'color': '#4bff9f', 'width': 3}, 'thickness': 0.75, 'value': 60}
            }
        ))
        fig_g.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig_g, use_container_width=True)

# ======================
# PAGE: ABOUT
# ======================
elif page == "👤 About":

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Portfolio</div>
        <h1 class="hero-title">Tentang <span>Proyek Ini</span></h1>
        <p class="hero-subtitle">Dokumentasi teknis dan latar belakang analisis</p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([3, 2])
    with col_a1:
        st.markdown("""
        <div class="about-card">
        <h3>🎯 Tujuan Proyek</h3>
        <p>Dashboard ini dikembangkan sebagai proyek portofolio untuk mendemonstrasikan kemampuan dalam
        <strong>Natural Language Processing (NLP)</strong> dan <strong>data visualization</strong>.
        Dataset berisi ulasan pengguna aplikasi RedBus yang dianalisis menggunakan teknik analisis sentimen.</p>
        <h3>🔬 Metodologi</h3>
        <ul>
            <li>Preprocessing teks: regex cleaning & normalisasi teks ulasan</li>
            <li>Analisis sentimen berbasis VADER Sentiment (compound score -1.0 hingga +1.0)</li>
            <li>Klasifikasi ke 4 kategori: Positive, Mixed Positive, Mixed Negative, Negative</li>
            <li>Bucketing skor sentimen untuk granularitas lebih tinggi</li>
            <li>Visualisasi interaktif menggunakan Plotly & Streamlit</li>
        </ul>
        <h3>📊 Sumber Data</h3>
        <p>Ulasan pengguna aplikasi RedBus dari Google Play Store, mencakup teks ulasan, rating bintang, dan metadata lainnya.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        skills = ["Python 3.11","Streamlit","Pandas","NumPy","Plotly","NLTK","VADER Sentiment","Scikit-learn","TF-IDF"]
        badges = "".join([f'<span class="skill-badge">{s}</span>' for s in skills])
        st.markdown(f'<div class="about-card"><h3>🛠 Tech Stack</h3>{badges}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="about-card">
        <h3>📈 Fitur Dashboard</h3>
        <ul>
            <li>KPI metrics real-time</li>
            <li>Filter interaktif multi-dimensi</li>
            <li>Analisis tren temporal</li>
            <li>Word frequency analysis</li>
            <li>Heatmap korelasi</li>
            <li>Business insight otomatis</li>
            <li>Export data CSV</li>
            <li>Gauge sentiment health</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;color:#555566;font-size:0.8rem;padding:1rem 0;'>
        Built with ❤️ using Streamlit · <span style='color:#ff4b4b;'>Portfolio Project</span> · 2024
    </div>
    """, unsafe_allow_html=True)

# ======================
# PAGE: MACHINE LEARNING
# ======================
elif page == "🤖 Machine Learning":

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Machine Learning · Logistic Regression</div>
        <h1 class="hero-title">Klasifikasi Sentimen<br>dengan <span>Logistic Regression</span></h1>
        <p class="hero-subtitle">Membandingkan pendekatan rule-based (VADER) dengan supervised machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Cek kolom Review tersedia
    if 'Review' not in df.columns:
        st.error("Kolom 'Review' tidak ditemukan di dataset.")
    else:
        # ======================
        # TRAIN MODEL
        # ======================
        @st.cache_data
        def train_model(data):
            ml_df = data[['Review', 'SentimentCategory']].dropna().copy()
            ml_df = ml_df[ml_df['Review'].astype(str).str.strip() != '']

            X = ml_df['Review'].astype(str)
            y = ml_df['SentimentCategory']

            # TF-IDF vectorizer
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                strip_accents='unicode',
                lowercase=True
            )

            X_vec = tfidf.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train Logistic Regression
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm     = confusion_matrix(y_test, y_pred, labels=model.classes_)

            return model, tfidf, acc, report, cm, model.classes_, y_test, y_pred, len(ml_df)

        with st.spinner("⏳ Melatih model Logistic Regression..."):
            model, tfidf, acc, report, cm, classes, y_test, y_pred, n_data = train_model(df)

        # ======================
        # KPI
        # ======================
        st.markdown('<div class="section-header">Performa Model</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Akurasi Model", f"{acc*100:.1f}%")
        with c2: st.metric("Total Data Training", f"{int(n_data*0.8):,}")
        with c3: st.metric("Total Data Testing", f"{int(n_data*0.2):,}")
        with c4: st.metric("Jumlah Kelas", "4")

        # ======================
        # CLASSIFICATION REPORT
        # ======================
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)

        report_rows = []
        for label in classes:
            if label in report:
                r = report[label]
                report_rows.append({
                    'Kelas': label,
                    'Precision': round(r['precision'], 3),
                    'Recall': round(r['recall'], 3),
                    'F1-Score': round(r['f1-score'], 3),
                    'Support': int(r['support'])
                })
        report_df = pd.DataFrame(report_rows)

        # Tampilkan sebagai chart
        fig_report = px.bar(
            report_df.melt(id_vars='Kelas', value_vars=['Precision', 'Recall', 'F1-Score']),
            x='Kelas', y='value', color='variable', barmode='group',
            title='Precision / Recall / F1-Score per Kelas',
            color_discrete_map={'Precision': '#4b9eff', 'Recall': '#4bff9f', 'F1-Score': '#ff4b4b'},
            labels={'value': 'Skor', 'variable': 'Metrik'}
        )
        apply_theme(fig_report)
        fig_report.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_report, use_container_width=True)

        # Tabel detail
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        # ======================
        # CONFUSION MATRIX
        # ======================
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

        col_cm1, col_cm2 = st.columns([2, 1])
        with col_cm1:
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            fig_cm = px.imshow(
                cm_df,
                title='Confusion Matrix — Logistic Regression',
                color_continuous_scale=['#0f0f13', '#ff4b4b'],
                text_auto=True,
                labels=dict(x='Prediksi', y='Aktual')
            )
            apply_theme(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm2:
            st.markdown("""
            <div class="insight-card">
            <p><strong>Cara baca Confusion Matrix:</strong><br><br>
            Diagonal = prediksi <strong style='color:#4bff9f;'>benar</strong><br>
            Di luar diagonal = prediksi <strong style='color:#ff4b4b;'>salah</strong><br><br>
            Semakin terang warna diagonal, semakin baik model.</p>
            </div>
            """, unsafe_allow_html=True)

            # Macro avg
            macro = report.get('macro avg', {})
            st.markdown(f"""
            <div class="insight-card" style="margin-top:1rem;">
            <p><strong>Macro Average:</strong><br><br>
            Precision: <strong style='color:#4b9eff;'>{macro.get('precision',0):.3f}</strong><br>
            Recall: <strong style='color:#4bff9f;'>{macro.get('recall',0):.3f}</strong><br>
            F1-Score: <strong style='color:#ff4b4b;'>{macro.get('f1-score',0):.3f}</strong>
            </p>
            </div>
            """, unsafe_allow_html=True)

        # ======================
        # PERBANDINGAN VADER vs LR
        # ======================
        st.markdown('<div class="section-header">Perbandingan VADER vs Logistic Regression</div>', unsafe_allow_html=True)

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("""
            <div class="insight-card" style="border-left-color:#4b9eff;">
            <p><strong style='color:#4b9eff; font-size:1rem;'>⚙️ VADER (Rule-Based)</strong><br><br>
            • Tidak butuh data training<br>
            • Cepat dan ringan<br>
            • Cocok untuk bahasa Inggris<br>
            • Kurang akurat untuk slang & bahasa campuran<br>
            • Skor: -1.0 hingga +1.0 (kontinu)<br>
            • Digunakan: Labeling awal dataset ini
            </p>
            </div>
            """, unsafe_allow_html=True)

        with col_v2:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#4bff9f;">
            <p><strong style='color:#4bff9f; font-size:1rem;'>🤖 Logistic Regression (ML)</strong><br><br>
            • Belajar dari data (supervised)<br>
            • Akurasi: <strong style='color:#ff4b4b;'>{acc*100:.1f}%</strong> pada data test<br>
            • TF-IDF + Logistic Regression<br>
            • Lebih adaptif terhadap pola lokal<br>
            • Output: probabilitas per kelas<br>
            • Bisa diimprove dengan lebih banyak data
            </p>
            </div>
            """, unsafe_allow_html=True)

        # Distribusi prediksi LR vs VADER
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            vader_dist = df['SentimentCategory'].value_counts().reset_index()
            vader_dist.columns = ['Kelas', 'Jumlah']
            fig_v = px.pie(vader_dist, names='Kelas', values='Jumlah',
                           title='Distribusi Label VADER (Full Data)',
                           hole=0.4, color='Kelas', color_discrete_map=COLORS)
            fig_v.update_traces(textfont=dict(color='white'))
            apply_theme(fig_v)
            st.plotly_chart(fig_v, use_container_width=True)

        with col_d2:
            lr_dist = pd.Series(y_pred).value_counts().reset_index()
            lr_dist.columns = ['Kelas', 'Jumlah']
            fig_lr = px.pie(lr_dist, names='Kelas', values='Jumlah',
                            title='Distribusi Prediksi LR (Data Test)',
                            hole=0.4, color='Kelas', color_discrete_map=COLORS)
            fig_lr.update_traces(textfont=dict(color='white'))
            apply_theme(fig_lr)
            st.plotly_chart(fig_lr, use_container_width=True)

        # ======================
        # COBA PREDIKSI
        # ======================
        st.markdown('<div class="section-header">🧪 Coba Prediksi Teks Sendiri</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card" style="border-left-color:#ffb84b; margin-bottom:1rem;">
        <p>Masukkan teks ulasan di bawah, model akan memprediksi sentimennya menggunakan Logistic Regression.</p>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Ketik ulasan di sini:",
            placeholder="Contoh: aplikasi ini sangat membantu, mudah digunakan dan cepat...",
            height=100
        )

        if st.button("🔍 Prediksi Sentimen", use_container_width=False):
            if user_input.strip() == '':
                st.warning("Masukkan teks dulu sebelum prediksi.")
            else:
                vec_input  = tfidf.transform([user_input])
                prediction = model.predict(vec_input)[0]
                proba      = model.predict_proba(vec_input)[0]

                color_map = {
                    'positive':       '#4bff9f',
                    'mixed positive': '#a8ffcc',
                    'neutral':        '#4b9eff',
                    'mixed negative': '#ffb84b',
                    'negative':       '#ff4b4b',
                }
                pred_color = color_map.get(prediction, '#ffffff')

                st.markdown(f"""
                <div class="insight-card" style="border-left-color:{pred_color}; margin-top:1rem;">
                <p>Hasil Prediksi: <strong style='color:{pred_color}; font-size:1.3rem;'>{prediction.upper()}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                # Probability chart
                proba_df = pd.DataFrame({
                    'Kelas': model.classes_,
                    'Probabilitas': proba
                }).sort_values('Probabilitas', ascending=True)

                fig_proba = px.bar(
                    proba_df, x='Probabilitas', y='Kelas', orientation='h',
                    title='Probabilitas per Kelas',
                    color='Probabilitas',
                    color_continuous_scale=['#1a1a24', '#ff4b4b']
                )
                apply_theme(fig_proba)
                fig_proba.update_coloraxes(showscale=False)
                fig_proba.update_xaxes(range=[0, 1])
                st.plotly_chart(fig_proba, use_container_width=True)