import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
import cv2
import keras
from keras import layers
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
except ImportError:
    st.error("Eksik kütüphane: Lütfen terminale 'pip install av streamlit-webrtc' yazın.")
    st.stop()


# =========================================================
# 1. TEMEL AYARLAR
# =========================================================
st.set_page_config(
    page_title="Duyguliz",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_NAME = "duyguliz_arsiv.db"
MODEL_PATH = "duyguliz_improved_cnn_targeted_balanced.keras"
KAYIT_ARALIGI_SANIYE = 3

DUYGULAR_TR = ["Kızgın", "İğrenme", "Korku", "Mutlu", "Doğal", "Üzgün", "Şaşırmış"]
DUYGULAR_VIDEO = ["Kizgin", "Igrenme", "Korku", "Mutlu", "Dogal", "Uzgun", "Sasirmis"]

EMOJI_MAP = {
    "Kızgın": "😠",
    "İğrenme": "🤢",
    "Korku": "😨",
    "Mutlu": "😊",
    "Doğal": "😐",
    "Üzgün": "😔",
    "Şaşırmış": "😮"
}

COLOR_MAP = {
    "Kızgın": "#ff4b4b",
    "İğrenme": "#84cc16",
    "Korku": "#a855f7",
    "Mutlu": "#facc15",
    "Doğal": "#38bdf8",
    "Üzgün": "#60a5fa",
    "Şaşırmış": "#fb923c"
}


# =========================================================
# 2. MODERN ARAYÜZ TASARIMI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #0f172a 48%, #111827 100%);
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

.hero-card {
    padding: 36px 38px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.80));
    border: 1px solid rgba(148,163,184,0.22);
    box-shadow: 0 24px 70px rgba(0,0,0,0.34);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 54px;
    font-weight: 850;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #f8fafc, #38bdf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    max-width: 950px;
    line-height: 1.6;
}

.badge-row {
    margin-top: 22px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    margin-right: 8px;
    margin-bottom: 8px;
    border-radius: 999px;
    background: rgba(56,189,248,0.10);
    color: #bae6fd;
    border: 1px solid rgba(56,189,248,0.22);
    font-size: 13px;
    font-weight: 700;
}

.glass-card {
    padding: 23px;
    border-radius: 24px;
    background: rgba(15, 23, 42, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 20px 45px rgba(0,0,0,0.25);
    margin-bottom: 18px;
}

.metric-card {
    padding: 21px;
    border-radius: 22px;
    background: rgba(30, 41, 59, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.16);
    min-height: 132px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.metric-value {
    font-size: 30px;
    font-weight: 850;
    color: #f8fafc;
    margin-top: 8px;
}

.metric-note {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 18px;
    line-height: 1.6;
}

.status-pill {
    display: inline-block;
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.24);
    color: #bbf7d0;
    font-weight: 700;
    font-size: 13px;
}

.warning-pill {
    display: inline-block;
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(250,204,21,0.12);
    border: 1px solid rgba(250,204,21,0.26);
    color: #fef08a;
    font-weight: 700;
    font-size: 13px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30,41,59,0.65);
    border-radius: 999px;
    padding: 12px 20px;
    color: #cbd5e1;
    border: 1px solid rgba(148,163,184,0.14);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(56,189,248,0.24), rgba(168,85,247,0.24));
    color: #ffffff;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

hr {
    border-color: rgba(148,163,184,0.15);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. VERİTABANI VE MODEL AYARLARI
# =========================================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kayitlar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kullanici TEXT,
                duygu TEXT,
                guven REAL,
                zaman_damgasi TIMESTAMP
            )
        """)


def patch_keras():
    if hasattr(layers, "Dense"):
        orig = layers.Dense.from_config
        layers.Dense.from_config = lambda config: orig(
            {k: v for k, v in config.items() if k != "quantization_config"}
        )


patch_keras()
init_db()


@st.cache_resource
def model_yukle():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH, compile=False)
    return None


model = model_yukle()


def verileri_getir():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query("SELECT * FROM kayitlar", conn)

        if not df.empty:
            df["zaman_damgasi"] = pd.to_datetime(df["zaman_damgasi"])

        return df
    except Exception:
        return pd.DataFrame()


def kullanici_kayitlarini_temizle(kullanici_adi):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM kayitlar WHERE kullanici = ?", (kullanici_adi,))


# =========================================================
# 4. CANLI ANALİZ MOTORU
# =========================================================
class DuyguAnalizMotoru(VideoProcessorBase):
    def __init__(self, kullanici_adi="Ziyaretçi"):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.son_kayit_zamani = datetime.now()
        self.kullanici_adi = kullanici_adi

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if model is None:
                continue

            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            tahmin = model.predict(roi.reshape(1, 48, 48, 1), verbose=0)
            idx = np.argmax(tahmin)

            duygu_tablo = DUYGULAR_TR[idx]
            duygu_video = DUYGULAR_VIDEO[idx]
            guven = float(np.max(tahmin))

            cv2.putText(
                img,
                f"{duygu_video} %{int(guven * 100)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            su_an = datetime.now()

            if (su_an - self.son_kayit_zamani).total_seconds() >= KAYIT_ARALIGI_SANIYE:
                try:
                    with sqlite3.connect(DB_NAME) as conn:
                        conn.execute(
                            """
                            INSERT INTO kayitlar 
                            (kullanici, duygu, guven, zaman_damgasi) 
                            VALUES (?, ?, ?, ?)
                            """,
                            (self.kullanici_adi, duygu_tablo, guven, su_an)
                        )
                except:
                    pass

                self.son_kayit_zamani = su_an

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================================================
# 5. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎭 Duyguliz")
    st.caption("Kişisel duygu farkındalığı paneli")

    st.divider()

    kullanici = st.text_input("👤 Kullanıcı adı", value="Ziyaretçi")
    kullanici_adi = kullanici.strip() if kullanici.strip() else "Ziyaretçi"

    st.caption(f"Analiz kayıtları otomatik olarak her {KAYIT_ARALIGI_SANIYE} saniyede bir alınır.")

    st.divider()

    if model is not None:
        st.markdown('<span class="status-pill">● Sistem hazır</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warning-pill">● Sistem başlatılamadı</span>', unsafe_allow_html=True)
        st.caption("Model dosyası bulunamadı.")

    st.divider()

    st.markdown("### Algılanabilen Duygular")
    for d in DUYGULAR_TR:
        st.markdown(f"{EMOJI_MAP[d]} {d}")

    st.divider()

    if st.button("🗑️ Bu Kullanıcının Kayıtlarını Temizle"):
        kullanici_kayitlarini_temizle(kullanici_adi)
        st.success(f"{kullanici_adi} kullanıcısına ait kayıtlar temizlendi.")
        st.rerun()


# =========================================================
# 6. AKTİF KULLANICI VERİLERİ
# =========================================================
df_tum = verileri_getir()

if not df_tum.empty:
    df = df_tum[df_tum["kullanici"] == kullanici_adi].copy()
else:
    df = pd.DataFrame()


# =========================================================
# 7. ANA KARŞILAMA
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">Duyguliz</div>
    <div class="hero-subtitle">
        Merhaba <b>{kullanici_adi}</b>. Duyguliz, kamera görüntüsünden yüz ifadelerini analiz ederek
        duygu tahmini yapan ve bu sonuçları zaman içinde izlenebilir hale getiren yapay zekâ tabanlı
        bir duygu analizi sistemidir. Sistem; kişisel farkındalık amacıyla kullanılabileceği gibi
        eğitim teknolojileri, kullanıcı deneyimi analizi, sunum provası ve insan-bilgisayar etkileşimi
        gibi alanlara da entegre edilebilir.
    </div>
    <div class="badge-row">
        <span class="badge">Canlı Kamera Analizi</span>
        <span class="badge">Kullanıcıya Özel Kayıtlar</span>
        <span class="badge">Zaman Serisi Takibi</span>
        <span class="badge">Entegre Edilebilir Analiz Modülü</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. GENEL ÖZET KARTLARI
# =========================================================
if not df.empty:
    df_son = df.sort_values("zaman_damgasi", ascending=False)
    son_duygu = df_son.iloc[0]["duygu"]
    son_guven = df_son.iloc[0]["guven"]
    baskin_duygu = df["duygu"].value_counts().idxmax()
    toplam_kayit = len(df)
else:
    son_duygu = "-"
    son_guven = 0
    baskin_duygu = "-"
    toplam_kayit = 0

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Şu Anki Duygu</div>
        <div class="metric-value">{EMOJI_MAP.get(son_duygu, "🎭")} {son_duygu}</div>
        <div class="metric-note">{kullanici_adi} için son algılanan duygu</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tahmin Güveni</div>
        <div class="metric-value">%{son_guven * 100:.1f}</div>
        <div class="metric-note">Modelin son tahminden emin olma düzeyi</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baskın Ruh Hâli</div>
        <div class="metric-value">{EMOJI_MAP.get(baskin_duygu, "📊")} {baskin_duygu}</div>
        <div class="metric-note">{kullanici_adi} kayıtlarında en sık görülen duygu</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Toplam Analiz</div>
        <div class="metric-value">{toplam_kayit}</div>
        <div class="metric-note">{kullanici_adi} için kaydedilen analiz sayısı</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 9. SEKMELER
# =========================================================
sekme1, sekme2, sekme3, sekme4 = st.tabs([
    "🎥 Canlı Analiz",
    "📊 Ruh Hâlim",
    "🧾 Geçmiş Kayıtlar",
    "💜 Hakkında"
])


# =========================================================
# 9.1 CANLI ANALİZ
# =========================================================
with sekme1:
    sol, sag = st.columns([1.55, 1])

    with sol:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🎥 Canlı Duygu Analizi</div>
            <div class="section-subtitle">
                Kamerayı başlat ve yüz ifaden üzerinden anlık duygu analizini görüntüle.
                Analiz sonuçların seçili kullanıcı adına göre otomatik olarak kaydedilir.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if model is not None:
            st.info(f"💡 Kamera açıkken analizler her {KAYIT_ARALIGI_SANIYE} saniyede bir {kullanici_adi} kullanıcısı için kaydedilir.")
           webrtc_streamer(
               key="duyguliz-stream",
               mode=WebRtcMode.SENDRECV,
               video_processor_factory=lambda: DuyguAnalizMotoru(kullanici_adi),
               rtc_configuration={
                  "iceServers": [
                      {"urls": ["stun:stun.l.google.com:19302"]}
                  ]
               },
               media_stream_constraints={
                  "video": True,
                  "audio": False
               },
               async_processing=True
           )

        else:
            st.error("Sistem başlatılamadı. Model dosyası app.py ile aynı klasörde olmalı.")

    with sag:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">✨ Kullanım Önerileri</div>
            <div class="section-subtitle">
                Daha doğru analiz için yüzünü kameraya net şekilde göster, ortam ışığının yeterli olmasına dikkat et
                ve yüzünün büyük kısmının görünür olduğundan emin ol.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("Kamera açıkken duygu analizlerin kişisel arşivine otomatik olarak kaydedilir.")
        st.success("Sonuçlarını Ruh Hâlim sekmesinden kullanıcıya özel grafik olarak takip edebilirsin.")

        if st.button("🔄 Sonuçları Yenile"):
            st.rerun()


# =========================================================
# 9.2 RUH HÂLİ PANELİ
# =========================================================
with sekme2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">📊 {kullanici_adi} için Ruh Hâli Özeti</div>
        <div class="section-subtitle">
            Bu bölümde yalnızca <b>{kullanici_adi}</b> kullanıcısına ait duygu analizleri gösterilir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        df_sorted = df.sort_values("zaman_damgasi").copy()
        df_sorted["tahmin_guveni"] = df_sorted["guven"] * 100

        c1, c2 = st.columns([1.45, 1])

        with c1:
            fig_line = px.line(
                df_sorted,
                x="zaman_damgasi",
                y="tahmin_guveni",
                color="duygu",
                markers=True,
                color_discrete_map=COLOR_MAP,
                title="Tahmin Güveni Değişimi"
            )

            fig_line.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.25)",
                font=dict(color="#e5e7eb"),
                title_font=dict(size=20),
                legend_title_text="Duygu",
                yaxis_title="Tahmin Güveni (%)",
                xaxis_title="Zaman"
            )

            st.plotly_chart(fig_line, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                df_sorted,
                names="duygu",
                hole=0.58,
                color="duygu",
                color_discrete_map=COLOR_MAP,
                title="Genel Duygu Dağılımı"
            )

            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                title_font=dict(size=20),
                legend_title_text="Duygu"
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            duygu_sayim = df_sorted["duygu"].value_counts().reset_index()
            duygu_sayim.columns = ["duygu", "adet"]

            fig_bar = px.bar(
                duygu_sayim,
                x="duygu",
                y="adet",
                color="duygu",
                color_discrete_map=COLOR_MAP,
                title="En Çok Görülen Duygular"
            )

            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.25)",
                font=dict(color="#e5e7eb"),
                showlegend=False,
                xaxis_title="Duygu",
                yaxis_title="Kayıt Sayısı"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with c4:
            son_20 = df_sorted.tail(20)

            fig_scatter = px.scatter(
                son_20,
                x="zaman_damgasi",
                y="duygu",
                size="tahmin_guveni",
                color="duygu",
                color_discrete_map=COLOR_MAP,
                title="Son Duygu Kayıtları"
            )

            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.25)",
                font=dict(color="#e5e7eb"),
                xaxis_title="Zaman",
                yaxis_title="Duygu",
                showlegend=False
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.warning(f"{kullanici_adi} kullanıcısına ait kayıt bulunmuyor. Önce Canlı Analiz sekmesinden kamerayı başlat.")


# =========================================================
# 9.3 GEÇMİŞ KAYITLAR
# =========================================================
with sekme3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">🧾 {kullanici_adi} için Geçmiş Duygu Kayıtları</div>
        <div class="section-subtitle">
            Burada yalnızca <b>{kullanici_adi}</b> kullanıcısına ait duygu analizleri listelenir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        df_view = df.sort_values("zaman_damgasi", ascending=False).copy()
        df_view["guven"] = (df_view["guven"] * 100).round(2).astype(str) + " %"

        df_view = df_view.rename(columns={
            "id": "Kayıt No",
            "kullanici": "Kullanıcı",
            "duygu": "Duygu",
            "guven": "Tahmin Güveni",
            "zaman_damgasi": "Tarih / Saat"
        })

        if "Kayıt No" in df_view.columns:
            df_view = df_view.drop(columns=["Kayıt No"])

        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True
        )

        csv = df.sort_values("zaman_damgasi", ascending=False).to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 Kendi Kayıtlarımı CSV Olarak İndir",
            data=csv,
            file_name=f"duyguliz_{kullanici_adi}_duygu_kayitlari.csv",
            mime="text/csv"
        )

    else:
        st.warning(f"{kullanici_adi} kullanıcısına ait kayıt bulunmuyor.")


# =========================================================
# 9.4 HAKKINDA
# =========================================================
with sekme4:
    st.markdown("""
<div class="glass-card">
    <div class="section-title">💜 Duyguliz Nedir?</div>
    <div class="section-subtitle">
        Duyguliz, kamera görüntülerinden yüz ifadelerini analiz ederek duygu tahmini yapan ve bu tahminleri
        zaman içinde takip edilebilir hale getiren yapay zekâ tabanlı bir analiz sistemidir.
        Mevcut prototip, kullanıcı bazlı duygu kayıtları ve grafiksel ruh hâli takibi sunar.
        Bu yapı; online eğitim, sunum provası, kullanıcı deneyimi analizi ve insan-bilgisayar etkileşimi
        gibi sistemlere entegre edilebilecek bir duygu analizi modülü olarak genişletilebilir.
    </div>
</div>
""", unsafe_allow_html=True)
    
    h1, h2, h3 = st.columns(3)

with h1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Duygu Analizi</div>
        <div class="metric-value">🎥</div>
        <div class="metric-note">
            Kamera görüntüsünden yüz ifadesi analizi yaparak duygu tahmini üretir.
        </div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Zamansal Takip</div>
        <div class="metric-value">📊</div>
        <div class="metric-note">
            Tahminleri zaman damgasıyla kaydederek duygu değişimini grafiklerle gösterir.
        </div>
    </div>
    """, unsafe_allow_html=True)

with h3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Entegrasyon Potansiyeli</div>
        <div class="metric-value">🔗</div>
        <div class="metric-note">
            Eğitim, kullanıcı deneyimi ve insan-bilgisayar etkileşimi sistemlerine uyarlanabilir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### Kullanım Notu

    Duyguliz, yüz ifadelerinden duygu tahmini yapan destekleyici bir analiz sistemidir. 
    Sonuçlar psikolojik tanı amacı taşımaz; kullanıcı farkındalığı, davranış analizi ve farklı dijital sistemlere entegrasyon amacıyla değerlendirilmelidir.

    Daha doğru sonuçlar için kameranın net olması, yüzün görünür olması ve ortam ışığının yeterli olması önerilir.
    """)
