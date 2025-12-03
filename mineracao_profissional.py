import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import geopandas as gpd
import requests
import json
warnings.filterwarnings('ignore')
pio.templates.default = "plotly_white"

print("=" * 80)
print("SISTEMA DE MINERAÇÃO DE DADOS MUNICIPAIS - K-MEDOIDS")
print("=" * 80)

# ==============================================================================
# 1. FUNÇÕES DE CARREGAMENTO E LIMPEZA (ETL)
# ==============================================================================

def limpar_string_ibge(val):
    if isinstance(val, str):
        return pd.to_numeric(''.join(filter(str.isdigit, val)), errors='coerce')
    return val


def carregar_dados_ibge():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    files = {
        'pib': os.path.join(base_dir, 'pib.xlsx'),
        'alfa': os.path.join(base_dir, 'alfabetizacao.xlsx'),
        'dens': os.path.join(base_dir, 'densidade.xlsx')
    }

    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {path}")

    print("[1/6] Carregando e tratando bases de dados")

    # --- 1. PIB (2021) ---
    try:
        df_pib = pd.read_excel(files['pib'], header=None)
        start_idx = df_pib[df_pib[0].astype(str).str.contains('^1$|^11', regex=True)].index[0]
        df_pib = df_pib.iloc[start_idx:].copy()
        df_pib = df_pib.iloc[:, [0, 1, 2]]
        df_pib.columns = ['codigo_ibge', 'municipio_sujo', 'pib']
        df_pib['codigo_ibge'] = pd.to_numeric(df_pib['codigo_ibge'], errors='coerce')
        df_pib['pib'] = pd.to_numeric(df_pib['pib'], errors='coerce')
        df_pib = df_pib.dropna(subset=['codigo_ibge', 'pib'])
        df_pib = df_pib[df_pib['municipio_sujo'] != 'Brasil']
    except Exception as e:
        print(f"Erro ao ler PIB: {e}")
        exit()

    # --- 2. Alfabetização (2010) ---
    try:
        df_alfa = pd.read_excel(files['alfa'], header=None)
        start_idx = df_alfa[df_alfa[0].astype(str).str.contains('^1$|^11', regex=True)].index[0]
        df_alfa = df_alfa.iloc[start_idx:].copy()
        df_alfa = df_alfa.iloc[:, [0, 2]]
        df_alfa.columns = ['codigo_ibge', 'taxa_alfabetizacao']
        df_alfa['codigo_ibge'] = pd.to_numeric(df_alfa['codigo_ibge'], errors='coerce')
        df_alfa['taxa_alfabetizacao'] = pd.to_numeric(df_alfa['taxa_alfabetizacao'], errors='coerce')
        df_alfa = df_alfa.dropna()
    except Exception as e:
        print(f"Erro ao ler Alfabetização: {e}")
        exit()

    # --- 3. Densidade (2010) ---
    try:
        df_dens = pd.read_excel(files['dens'], header=None)
        start_idx = df_dens[df_dens[0].astype(str).str.contains('^1$|^11', regex=True)].index[0]
        df_dens = df_dens.iloc[start_idx:].copy()
        df_dens = df_dens.iloc[:, [0, 2]]
        df_dens.columns = ['codigo_ibge', 'densidade']
        df_dens['codigo_ibge'] = pd.to_numeric(df_dens['codigo_ibge'], errors='coerce')
        df_dens['densidade'] = pd.to_numeric(df_dens['densidade'], errors='coerce')
        df_dens = df_dens.dropna()
    except Exception as e:
        print(f"Erro ao ler Densidade: {e}")
        exit()

    # --- MERGE ---
    print("[2/6] Unificando datasets")
    df_final = pd.merge(df_pib, df_alfa, on='codigo_ibge', how='inner')
    df_final = pd.merge(df_final, df_dens, on='codigo_ibge', how='inner')
    df_final['municipio'] = df_final['municipio_sujo'].astype(str).str.strip()
    df_final['codigo_ibge'] = df_final['codigo_ibge'].astype('int64')
    df_final['log_pib'] = np.log1p(df_final['pib'])
    df_final['log_densidade'] = np.log1p(df_final['densidade'])
    return df_final[
        [
            'codigo_ibge', 'municipio', 'pib',
            'taxa_alfabetizacao', 'densidade',
            'log_pib', 'log_densidade'
        ]
    ]


# ==============================================================================
# 2. EXECUÇÃO DA ANÁLISE (K-MEDOIDS)
# ==============================================================================

df = carregar_dados_ibge()
print(f"Total de municípios carregados e limpos: {len(df)}")

features_originais = ['pib', 'taxa_alfabetizacao', 'densidade']
features_modelo = ['log_pib', 'taxa_alfabetizacao', 'log_densidade']

print("\n[3/6] Normalizando dados (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_modelo])

N_CLUSTERS = 4
print(f"\n[4/6] Executando K-Medoids (k={N_CLUSTERS})")
kmedoids = KMedoids(
    n_clusters=N_CLUSTERS,
    random_state=42,
    metric='euclidean',
    method='pam',
    init='k-medoids++'
) 

labels = kmedoids.fit_predict(X_scaled)
medoid_indices = kmedoids.medoid_indices_

df['cluster_id'] = labels
df['e_medoide'] = df.index.isin(medoid_indices)

silhueta = silhouette_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)

print("Métricas do Modelo:")
print(f" - Silhouette Score: {silhueta:.4f}")
print(f" - Calinski-Harabasz: {calinski:.2f}")

# ==============================================================================
# 3. INTERPRETAÇÃO E NOMEAÇÃO DOS CLUSTERS
# ==============================================================================

print("\n[5/6] Gerando perfis e visualizações")

resumo_clusters = df.groupby('cluster_id')[features_originais].mean()
ordem_riqueza = resumo_clusters.sort_values('pib').index

mapa_nomes = {}
nomes_sugeridos = [
    "Baixo Desenv. Econômico",
    "Médio Desenv. / Rural",
    "Alto Desenv. / Urbano",
    "Polos Econômicos / Metrópoles"
]

if N_CLUSTERS == 4:
    for i, cluster_idx in enumerate(ordem_riqueza):
        mapa_nomes[cluster_idx] = nomes_sugeridos[i]
else:
    for i, cluster_idx in enumerate(ordem_riqueza):
        mapa_nomes[cluster_idx] = f"Nível {i+1} (Riqueza)"

df['perfil_municipio'] = df['cluster_id'].map(mapa_nomes)

print("\nPERFIL DOS GRUPOS ENCONTRADOS (Médias):")
print(df.groupby('perfil_municipio')[features_originais].mean().round(2).sort_values('pib'))

# ==============================================================================
# 3.1. DISCRETIZAÇÃO EM FAIXAS PARA REGRAS DE ASSOCIAÇÃO
# ==============================================================================
df['faixa_pib'] = pd.qcut(
    df['pib'],
    q=4,
    labels=['PIB_Baixo', 'PIB_Medio', 'PIB_Alto', 'PIB_MuitoAlto']
)
def categorizar_alfabetizacao(x):
    if x < 80:
        return 'Alfa_Baixa'
    elif x < 90:
        return 'Alfa_Media'
    else:
        return 'Alfa_Alta'

df['faixa_alfabetizacao'] = df['taxa_alfabetizacao'].apply(categorizar_alfabetizacao)
def categorizar_densidade(x):
    if x < 10:
        return 'Dens_MuitoBaixa'
    elif x < 50:
        return 'Dens_Baixa'
    elif x < 150:
        return 'Dens_Media'
    else:
        return 'Dens_Alta'

df['faixa_densidade'] = df['densidade'].apply(categorizar_densidade)

# ==============================================================================
# 4. VISUALIZAÇÕES CLÁSSICAS (PCA, BOXPLOT, 3D)
# ==============================================================================

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

fig_pca = px.scatter(
    x=components[:, 0],
    y=components[:, 1],
    color=df['perfil_municipio'],
    hover_data=[df['municipio'], df['pib'], df['taxa_alfabetizacao']],
    title=f"Segmentação de Municípios (K-Medoids, Silhueta: {silhueta:.2f})",
    labels={'x': 'Componente Principal 1', 'y': 'Componente Principal 2'},
    height=600
)

medoides_pca = components[medoid_indices]
fig_pca.add_trace(
    go.Scatter(
        x=medoides_pca[:, 0],
        y=medoides_pca[:, 1],
        mode='markers+text',
        marker=dict(symbol='star', size=15, color='black'),
        text=[df.iloc[i]['municipio'] for i in medoid_indices],
        textposition="top center",
        name='Municípios Representativos (Medóides)'
    )
)

fig_box = px.box(
    df,
    x='perfil_municipio',
    y='pib',
    points=False,
    log_y=True,
    title="Distribuição do PIB por Perfil (Escala Log)",
    color='perfil_municipio'
)

fig_3d = px.scatter_3d(
    df,
    x='log_pib',
    y='taxa_alfabetizacao',
    z='log_densidade',
    color='perfil_municipio',
    hover_name='municipio',
    title="Espaço 3D: Riqueza vs Educação vs Densidade",
    opacity=0.7
)

# ==============================================================================
# 5. MAPA DINÂMICO DO BRASIL (GELOCALIZAÇÃO)
# ==============================================================================

print("\nCriando mapa dinâmico do Brasil por município...")

try:
    url_geojson = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json"
    geojson_data = requests.get(url_geojson).json()
    df['codigo_ibge'] = df['codigo_ibge'].astype('int64')
    df['uf'] = df['municipio'].str.extract(r'\((..)\)').iloc[:, 0]
    df['uf'] = df['uf'].fillna('SEM_UF')
    df['tipo_municipio'] = np.where(df['e_medoide'], 'Medóide', 'Município')
    fig_mapa = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations='codigo_ibge',
        featureidkey='properties.id',
        color='perfil_municipio',
        hover_name='municipio',
        hover_data={
            'codigo_ibge': True,
            'pib': ':.0f',
            'taxa_alfabetizacao': ':.2f',
            'densidade': ':.2f',
            'uf': True,
            'tipo_municipio': True
        },
        mapbox_style='carto-positron',
        center={'lat': -14.235, 'lon': -51.925},
        zoom=3.0,
        opacity=0.75,
        title="Mapa Interativo dos Municípios Brasileiros por Perfil (K-Medoids)"
    )

    fig_mapa.update_layout(
        margin={'r': 0, 't': 60, 'l': 0, 'b': 0},
        legend_title_text="Perfil Socioeconômico"
    )

    fig_mapa.write_html("0_mapa_brasil_municipios.html")
    print("Mapa interativo salvo em: 0_mapa_brasil_municipios.html")
except Exception as e:
    print(f"Falha ao gerar mapa dinâmico: {e}")
    print("   Verifique conexão com a internet e instalação de 'requests' e 'geopandas' se necessário.")

# ==============================================================================
# 6. SALVAR GRÁFICOS E EXPORTAR RESULTADOS
# ==============================================================================

print("\nSalvando gráficos HTML padrões (PCA, Boxplot, 3D)...")
fig_pca.write_html("1_mapa_clusters_pca.html")
fig_box.write_html("2_distribuicao_pib.html")
fig_3d.write_html("3_analise_3d.html")
colunas_finais = [
    'codigo_ibge', 'municipio', 'uf', 'perfil_municipio',
    'pib', 'taxa_alfabetizacao', 'densidade', 'e_medoide',
    'faixa_pib', 'faixa_alfabetizacao', 'faixa_densidade'
]

df_export = df[colunas_finais].sort_values(
    ['perfil_municipio', 'pib'],
    ascending=[True, False]
)

nome_arquivo_final = "resultado_mineracao_municipios.xlsx"
writer = pd.ExcelWriter(nome_arquivo_final, engine='xlsxwriter')
df_export.to_excel(writer, sheet_name='Segmentacao', index=False)

workbook = writer.book
worksheet = writer.sheets['Segmentacao']
format_money = workbook.add_format({'num_format': '#,##0'})
format_float = workbook.add_format({'num_format': '0.00'})

worksheet.set_column('B:B', 30)
worksheet.set_column('C:C', 8)
worksheet.set_column('D:D', 30)
worksheet.set_column('E:E', 15, format_money)
worksheet.set_column('F:G', 12, format_float)

writer.close()

print(f"\nArquivo final gerado: {nome_arquivo_final}")
print("Medóides (Municípios Típicos de cada grupo):")
for idx in medoid_indices:
    row = df.iloc[idx]
    print(
        f" > {row['perfil_municipio']}: {row['municipio']} "
        f"(PIB: {row['pib']:.0f}, Alfa: {row['taxa_alfabetizacao']}%)"
    )

print("\nProcesso concluído com sucesso.")
print("=" * 80)
