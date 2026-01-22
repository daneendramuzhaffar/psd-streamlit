import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Analisis HIV Jawa Barat", layout="wide")

st.title("📊 Pemetaan dan Clustering Penyebaran HIV/AIDS Jawa Barat")
st.markdown("Aplikasi ini memetakan zonasi risiko menggunakan algoritma K-Means berdasarkan kelompok umur.")

# ------------------------------------------------------------------------------
# BAGIAN 1: PRE-PROCESSING DATA
# ------------------------------------------------------------------------------

# Nama file (Pastikan file ini sudah di-upload ke GitHub di folder yang sama)
csv_filename = 'jumlah_kasus_hiv_berdasarkan_kelompok_umur_v1_data.csv'
geojson_filename = 'jawa_barat_32_batas_kabkota.geojson'

@st.cache_data
def load_data():
    if not os.path.exists(csv_filename) or not os.path.exists(geojson_filename):
        return None, None
    
    df_load = pd.read_csv(csv_filename)
    with open(geojson_filename, 'r') as f:
        geojson_load = json.load(f)
    return df_load, geojson_load

# Memuat data
df, jabar_geojson = load_data()

if df is None:
    st.error(f"❌ File '{csv_filename}' atau '{geojson_filename}' tidak ditemukan!")
    st.info("Silakan pastikan kedua file tersebut sudah berada di repository GitHub Anda.")
    st.stop()

# Transformasi Data
df_agg = df.groupby(['nama_kabupaten_kota', 'kelompok_umur'])['jumlah_kasus'].sum().reset_index()
df_pivot = df_agg.pivot(index='nama_kabupaten_kota', columns='kelompok_umur', values='jumlah_kasus').fillna(0).reset_index()
df_total = df.groupby('nama_kabupaten_kota')['jumlah_kasus'].sum().reset_index()
df_total.columns = ['nama_kabupaten_kota', 'Total_Kasus']

df_final = pd.merge(df_total, df_pivot, on='nama_kabupaten_kota')

def normalisasi_nama(nama):
    return nama.title() # Ubah "KOTA BANDUNG" jadi "Kota Bandung"
df_final['nama_peta'] = df_final['nama_kabupaten_kota'].apply(normalisasi_nama)

# ------------------------------------------------------------------------------
# BAGIAN 2: METODOLOGI CLUSTERING
# ------------------------------------------------------------------------------
st.header("⚙️ Metodologi & Validasi Cluster")

# Fitur untuk clustering (Kolom Umur)
features = df_pivot.columns[1:] 
X = df_final[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Grafik Elbow
inertia = []
range_k = range(1, 10)
for k in range_k:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertia.append(kmeans_temp.inertia_)

fig_elbow = go.Figure(data=go.Scatter(x=list(range_k), y=inertia, mode='lines+markers'))
fig_elbow.update_layout(
    title='Uji Validasi Elbow Method',
    xaxis_title='Jumlah Cluster (k)',
    yaxis_title='Inertia (SSE)',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig_elbow, use_container_width=True)

# ------------------------------------------------------------------------------
# BAGIAN 3: PROSES CLUSTERING & LABELING
# ------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['Cluster'] = kmeans.fit_predict(X_scaled)

# Mapping Label Risiko
cluster_mean = df_final.groupby('Cluster')['Total_Kasus'].mean().sort_values()
risk_map = {
    cluster_mean.index[0]: 'Resiko Rendah',
    cluster_mean.index[1]: 'Resiko Sedang',
    cluster_mean.index[2]: 'Resiko Tinggi (Hotspot)'
}
df_final['Kategori_Resiko'] = df_final['Cluster'].map(risk_map)

# ------------------------------------------------------------------------------
# BAGIAN 4: VISUALISASI PETA & STATISTIK
# ------------------------------------------------------------------------------
st.header("🗺️ Hasil Visualisasi Geospasial")

col1, col2 = st.columns([2, 1])

with col1:
    color_map = {'Resiko Tinggi (Hotspot)': 'red', 'Resiko Sedang': 'orange', 'Resiko Rendah': 'green'}
    fig_map = px.choropleth_mapbox(
        df_final,
        geojson=jabar_geojson,
        locations='nama_peta',
        featureidkey="properties.name",
        color='Kategori_Resiko',
        color_discrete_map=color_map,
        hover_name='nama_kabupaten_kota',
        hover_data={'Total_Kasus': True},
        center={"lat": -6.9175, "lon": 107.6191},
        zoom=7,
        mapbox_style="carto-positron",
        title="Peta Zonasi Risiko HIV Jawa Barat"
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.write("**Tabel Klasifikasi Wilayah:**")
    st.dataframe(df_final[['nama_kabupaten_kota', 'Total_Kasus', 'Kategori_Resiko']].sort_values(by='Total_Kasus', ascending=False), height=400)

# Analisis Karakteristik Umur
st.subheader("👥 Karakteristik Berdasarkan Kelompok Umur")
cluster_profile = df_final.groupby('Kategori_Resiko')[features].mean().reset_index()
cluster_melt = cluster_profile.melt(id_vars='Kategori_Resiko', var_name='Kelompok Umur', value_name='Rata-rata Kasus')

fig_profile = px.bar(
    cluster_melt, x='Kelompok Umur', y='Rata-rata Kasus', color='Kategori_Resiko',
    barmode='group',
    color_discrete_map=color_map,
    title='Rata-rata Kasus per Kelompok Umur di Setiap Cluster'
)
st.plotly_chart(fig_profile, use_container_width=True)

# ------------------------------------------------------------------------------
# BAGIAN 5: ANALISIS TREN TAHUNAN
# ------------------------------------------------------------------------------
st.header("📈 Analisis Tren & Timeline")

tab1, tab2 = st.tabs(["Tren Total", "Peta Animasi"])

with tab1:
    df_trend_jabar = df.groupby('tahun')['jumlah_kasus'].sum().reset_index()
    fig_line = px.line(df_trend_jabar, x='tahun', y='jumlah_kasus', markers=True, title='Kenaikan Total Kasus Tahunan')
    fig_line.update_traces(line_color='red', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    df_map_yearly = df.groupby(['nama_kabupaten_kota', 'tahun'])['jumlah_kasus'].sum().reset_index()
    df_map_yearly['nama_peta'] = df_map_yearly['nama_kabupaten_kota'].apply(normalisasi_nama)
    df_map_yearly = df_map_yearly.sort_values('tahun')

    fig_anim = px.choropleth_mapbox(
        df_map_yearly, geojson=jabar_geojson, locations='nama_peta', featureidkey="properties.name",
        color='jumlah_kasus', animation_frame='tahun',
        range_color=[0, df_map_yearly['jumlah_kasus'].max()],
        color_continuous_scale="Reds",
        center={"lat": -6.9175, "lon": 107.6191}, zoom=6.5,
        mapbox_style="carto-positron", title="Timeline Penyebaran Kasus"
    )
    fig_anim.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_anim, use_container_width=True)

# ------------------------------------------------------------------------------
# BAGIAN 6: KESIMPULAN & REKOMENDASI
# ------------------------------------------------------------------------------
st.divider()
st.header("📋 Kesimpulan & Rekomendasi")

hotspots = df_final[df_final['Kategori_Resiko'] == 'Resiko Tinggi (Hotspot)']
avg_hotspot = cluster_profile[cluster_profile['Kategori_Resiko']=='Resiko Tinggi (Hotspot)'].iloc[:, 1:].values.max()

st.success(f"**Kesimpulan Wilayah:** Teridentifikasi {len(hotspots)} wilayah 'HOTSPOT': {', '.join(hotspots['nama_kabupaten_kota'].tolist())}.")

if '15-19' in df_final.columns:
    top_remaja = df_final.sort_values(by='15-19', ascending=False).iloc[0]
    st.warning(f"**Temuan Khusus:** {top_remaja['nama_kabupaten_kota']} memiliki angka kerentanan tertinggi pada usia remaja (15-19 tahun).")

st.info("**Saran Kebijakan:** Diperlukan penguatan program Mobile VCT dan distribusi ARV yang lebih masif pada wilayah Zona Merah, terutama yang memiliki kawasan industri padat.")
