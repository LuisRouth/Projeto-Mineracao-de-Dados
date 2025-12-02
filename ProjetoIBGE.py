import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from pyclustering.cluster.kmedoids import kmedoids

warnings.filterwarnings('ignore')
px.defaults.template = "plotly_white"

print("="*80)
print("SEGMENTAÇÃO DE MUNICÍPIOS BRASILEIROS (VERSÃO APRIMORADA) - K-MEDOIDS")
print("="*80)

# ==============================================================================
# PASSO 1: CARREGAMENTO E LIMPEZA DOS DADOS IBGE
# ==============================================================================
print("\n[FASE 1] Carregando e limpando os arquivos Excel do IBGE")

PASTA_DADOS = 'dados_brutos'

print(f"Lendo arquivos...")
df_pib = pd.read_excel(os.path.join(PASTA_DADOS, 'pib.xlsx'), skiprows=4, header=None, engine='openpyxl').iloc[1:, [0, 1, -1]]
df_pib.columns = ['codigo_ibge', 'municipio', 'pib_per_capita']

df_alfabetizacao = pd.read_excel(os.path.join(PASTA_DADOS, 'alfabetizacao.xlsx'), skiprows=4, header=None, engine='openpyxl').iloc[1:, [0, -1]]
df_alfabetizacao.columns = ['codigo_ibge', 'taxa_alfabetizacao']

df_densidade = pd.read_excel(os.path.join(PASTA_DADOS, 'densidade.xlsx'), skiprows=4, header=None, engine='openpyxl').iloc[1:, [0, -1]]
df_densidade.columns = ['codigo_ibge', 'densidade_demografica']

for df_temp in [df_pib, df_alfabetizacao, df_densidade]:
    for col in df_temp.columns:
        if col != 'municipio':
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp.dropna(inplace=True)
    df_temp['codigo_ibge'] = df_temp['codigo_ibge'].astype('int64')

print("\nUnificando os 3 datasets por código IBGE...")
df_final = pd.merge(df_pib, df_alfabetizacao, on='codigo_ibge', how='inner')
df_final = pd.merge(df_final, df_densidade, on='codigo_ibge', how='inner')
print(f"SUCESSO! Dataset consolidado com {len(df_final)} municípios")

# ==============================================================================
# PASSO 1.5: ANÁLISE E TRATAMENTO DE OUTLIERS (MANTIDO 100% ORIGINAL)
# ==============================================================================
print("\n[FASE 1.5] Análise e tratamento de outliers (Método IQR)...")
df_antes = df_final.copy()
indicadores_analise = ['pib_per_capita', 'densidade_demografica']

for col in indicadores_analise:
    Q1 = df_final[col].quantile(0.25)
    Q3 = df_final[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    outliers = df_final[df_final[col] > limite_superior]
    if not outliers.empty:
        print(f"Encontrados {len(outliers)} outliers para '{col}' (valores acima de {limite_superior:.2f})")
        df_final = df_final[df_final[col] <= limite_superior]

print(f"\nTotal de municípios após a remoção de outliers: {len(df_final)}")
print(f"Foram removidos {len(df_antes) - len(df_final)} municípios considerados extremos.")
df_final.reset_index(drop=True, inplace=True)
print("Índice do DataFrame resetado para garantir consistência.")

# ==============================================================================
# PASSO 2: NORMALIZAÇÃO DOS INDICADORES
# ==============================================================================
print("\n[FASE 2] Normalizando os dados com StandardScaler...")
indicadores = ['pib_per_capita', 'taxa_alfabetizacao', 'densidade_demografica']
scaler = StandardScaler()
df_normalizado = pd.DataFrame(scaler.fit_transform(df_final[indicadores]), columns=indicadores)
print("Dados normalizados (média=0, desvio padrão=1)")

# ==============================================================================
# PASSO 3: DETERMINAÇÃO ÓTIMA DO NÚMERO DE CLUSTERS (K)
# ==============================================================================
print("\n[FASE 3] Determinando K ótimo (Análise de Silhueta)...")
print("Aviso: Esta fase pode demorar.")
silhuetas = []
k_range = range(2, 8)
for k in k_range:
    np.random.seed(42)
    initial_medoids = np.random.choice(len(df_normalizado), k, replace=False).tolist()
    kmedoids_instance = kmedoids(df_normalizado.values, initial_medoids)
    kmedoids_instance.process()
    cluster_indices = kmedoids_instance.get_clusters()
    labels_temp = np.zeros(len(df_normalizado))
    for cluster_id, indices in enumerate(cluster_indices):
        labels_temp[indices] = cluster_id
    silhuetas.append(silhouette_score(df_normalizado, labels_temp))
fig_silhueta = px.bar(x=list(k_range), y=silhuetas, title='<b>Coeficiente de Silhueta por K</b>')
fig_silhueta.show()
K_ESCOLHIDO = 4
print(f"K escolhido = {K_ESCOLHIDO} clusters.")

# ==============================================================================
# PASSO 4: APLICAÇÃO FINAL DO K-MEDOIDS E ANÁLISE
# ==============================================================================
print(f"\n[FASE 4] Executando K-Medoids final com K={K_ESCOLHIDO}...")
np.random.seed(42)
initial_medoids = np.random.choice(len(df_normalizado), K_ESCOLHIDO, replace=False).tolist()
modelo_final = kmedoids(df_normalizado.values, initial_medoids)
modelo_final.process()
clusters = np.zeros(len(df_normalizado))
for cluster_id, indices in enumerate(modelo_final.get_clusters()):
    clusters[indices] = cluster_id
df_final['cluster_num'] = clusters.astype(int)
indices_medoides = modelo_final.get_medoids()
df_final['e_medoide'] = df_final.index.isin(indices_medoides)

silhueta = silhouette_score(df_normalizado, clusters)
calinski = calinski_harabasz_score(df_normalizado, clusters)

perfil_ordenado = df_final.groupby('cluster_num')[indicadores].mean().sort_values('pib_per_capita')
nomes_perfis = ["Vulneráveis", "Emergentes", "Consolidados", "Dinâmicos"]
mapa_nomes = {cluster_idx: nome for i, (cluster_idx, nome) in enumerate(zip(perfil_ordenado.index, nomes_perfis))}
df_final['perfil'] = df_final['cluster_num'].map(mapa_nomes)

print("\nPERFIS IDENTIFICADOS (Médias):")
print(df_final.groupby('perfil')[indicadores].mean().round(2).sort_values('pib_per_capita'))

# ==============================================================================
# PASSO 5: VISUALIZAÇÕES
# ==============================================================================
print("\n[FASE 5] Gerando e exibindo visualizações...")

df_radar = df_final.groupby('perfil')[indicadores].mean().reset_index()
df_radar_melted = pd.melt(df_radar, id_vars=['perfil'], var_name='Indicador', value_name='Valor')
fig_radar = px.line_polar(df_radar_melted, r='Valor', theta='Indicador', color='perfil', line_close=True,
                          title='Personalidade Média de Cada Perfil', category_orders={'perfil': nomes_perfis})
fig_radar.show()

for indicador in indicadores:
    is_log = (indicador != 'taxa_alfabetizacao')
    titulo = f'Distribuição de {indicador.replace("_", " ").title()} por Perfil'
    if is_log:
        titulo += " (Escala Logarítmica)"
    fig_box = px.box(df_final, x='perfil', y=indicador, color='perfil', title=titulo,
                     category_orders={'perfil': nomes_perfis}, log_y=is_log)
    fig_box.show()

pca = PCA(n_components=2)
components = pca.fit_transform(df_normalizado)
df_final['pca1'] = components[:, 0]
df_final['pca2'] = components[:, 1]
fig_pca = px.scatter(df_final, x='pca1', y='pca2', color='perfil', hover_data=['municipio', 'pib_per_capita'],
                     title=f"Segmentação de Municípios via PCA (Silhueta: {silhueta:.2f})")
fig_pca.add_trace(go.Scatter(x=df_final.iloc[indices_medoides]['pca1'], y=df_final.iloc[indices_medoides]['pca2'], mode='markers+text',
                             marker=dict(symbol='star', size=15, color='black'),
                             text=df_final.iloc[indices_medoides]['municipio'], textposition="top center", name='Medóides'))
fig_pca.show()

try:
    url_geojson = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json"
    gdf = gpd.read_file(url_geojson)
    gdf['id'] = gdf['id'].astype('int64')
    mapa_final = gdf.merge(df_final, left_on='id', right_on='codigo_ibge', how='inner')
    fig_mapa = px.choropleth_mapbox(
        mapa_final, 
        geojson=mapa_final.geometry, 
        locations=mapa_final.index, 
        color='perfil',
        hover_name='municipio',
        hover_data={
            'pib_per_capita': ':.2f',
            'taxa_alfabetizacao': ':.2f',
            'densidade_demografica': ':.2f',
            'perfil': True
        },
        mapbox_style="carto-positron", 
        zoom=3.2, 
        center={"lat": -14.235, "lon": -51.925},
        title="Distribuição Geográfica dos Perfis", 
        category_orders={'perfil': nomes_perfis}
    )
    fig_mapa.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig_mapa.show()
    print("Gráficos exibidos com sucesso.")
except Exception as e:
    print(f"Mapa não pôde ser gerado: {e}.")
# ==============================================================================
# PASSO 6: GERAÇÃO DO RELATÓRIO PDF E EXPORTAÇÃO
# ==============================================================================
print("\n[FASE 6] Exportando resultados finais...")

PASTA_RESULTADOS = 'resultados'
os.makedirs(PASTA_RESULTADOS, exist_ok=True)
print(f"Arquivos serão salvos na pasta '{PASTA_RESULTADOS}/'")

colunas_finais = ['codigo_ibge', 'municipio', 'perfil'] + indicadores + ['e_medoide']
df_export = df_final[colunas_finais].sort_values(['perfil', 'pib_per_capita'], ascending=[True, False])

nome_arquivo_csv = os.path.join(PASTA_RESULTADOS, "01_dados_completos_segmentacao.csv")
df_export.to_csv(nome_arquivo_csv, index=False)
print(f"Dados completos exportados para: {nome_arquivo_csv}")

perfis_medios_export = df_final.groupby('perfil')[indicadores].mean().round(2).sort_values('pib_per_capita')
nome_arquivo_perfis = os.path.join(PASTA_RESULTADOS, "02_resumo_perfis_medios.csv")
perfis_medios_export.to_csv(nome_arquivo_perfis)
print(f"Resumo dos perfis médios exportado para: {nome_arquivo_perfis}")

medoides_export = df_final.iloc[indices_medoides][['perfil', 'municipio', 'pib_per_capita']].sort_values('pib_per_capita')
nome_arquivo_medoides = os.path.join(PASTA_RESULTADOS, "03_municipios_representantes.csv")
medoides_export.to_csv(nome_arquivo_medoides, index=False)
print(f"Municípios representantes exportados para: {nome_arquivo_medoides}")

print("\n" + "="*80)
print("RELATÓRIO EXECUTIVO DA SEGMENTAÇÃO DE MUNICÍPIOS")
print("="*80)
print(f"\n1. DADOS E METODOLOGIA:")
print(f"  - Foram analisados {len(df_antes)} municípios, e {len(df_final)} após remoção de outliers.")
print(f"  - Modelo: K-Medoids com {K_ESCOLHIDO} clusters.")
print(f"\n2. MÉTRICAS DE QUALIDADE DO MODELO:")
print(f"  - Silhouette Score: {silhueta:.3f}")
print(f"  - Calinski-Harabasz Score: {calinski:.0f}")
print(f"\n3. MUNICÍPIOS REPRESENTATIVOS (MEDÓIDES):")
for index, row in medoides_export.iterrows():
    print(f"  - {row['perfil']}: {row['municipio']} (PIB per capita: R$ {row['pib_per_capita']:,.0f})")

print("\n" + "="*80)
print("PROCESSO CONCLUÍDO COM SUCESSO!")
print("="*80)
