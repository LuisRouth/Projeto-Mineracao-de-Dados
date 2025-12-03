# Projeto de Mineracao de Dados - Municipios Brasileiros

Este projeto implementa um pipeline completo de Data Mining (KDD) aplicado a dados socioeconomicos de municipios brasileiros. O sistema utiliza algoritmos de aprendizado nao supervisionado para segmentacao (clusterizacao) e mineracao de regras de associacao para descoberta de padroes.

## Descricao Geral

O projeto opera em duas fases distintas:
1. **Clusterizacao (K-Medoids):** Agrupa municipios com caracteristicas similares baseando-se em PIB, Taxa de Alfabetizacao e Densidade Demografica.
2. **Regras de Associacao (Apriori):** Analisa os clusters gerados para encontrar regras condicionais (ex: Se X entao Y) relacionando UF, perfil economico e indicadores sociais.

## Estrutura de Arquivos

Para a execucao correta, o diretorio do projeto deve conter os seguintes arquivos na raiz:

* `mineracao_profissional.py`: Script principal de ETL, pre-processamento e clusterizacao.
* `associacao_municipios.py`: Script secundario para geracao de regras de associacao.
* `pib.xlsx`: Dados brutos do PIB (IBGE).
* `alfabetizacao.xlsx`: Dados brutos de alfabetizacao (IBGE).
* `densidade.xlsx`: Dados brutos de densidade demografica (IBGE).
* `requirements.txt`: Lista de dependencias do projeto.

## Pre-requisitos e Instalacao

O projeto foi desenvolvido em Python. Recomenda-se o uso de ambiente virtual (venv).

1. Instale as dependencias listadas no arquivo requirements.txt:

```
pip install -r requirements.txt
```

Como Executar
O pipeline deve ser executado sequencialmente.
Passo 1: Segmentacao e Analise Exploratoria
Execute o script de mineracao. Este processo carrega os dados brutos, realiza a limpeza, normalizacao (StandardScaler) e aplica o algoritmo K-Medoids.
```
python mineracao_profissional.py
```
* `resultado_mineracao_municipios.xlsx`: Arquivo Excel contendo a base tratada e a classificacao dos clusters.
* `0_mapa_brasil_municipios.html`: Mapa interativo do Brasil com a distribuicao dos clusters.
* `1_mapa_clusters_pca.html`: Visualizacao dos clusters reduzida a 2 dimensoes (PCA).
* `2_distribuicao_pib.html`: Boxplot da distribuicao de renda por grupo.
* `3_analise_3d.html`: Grafico tridimensional das variaveis analisadas.

Passo 2: Mineracao de Regras de Associacao
Apos a geracao do arquivo Excel no passo anterior, execute o script de associacao. Ele le a segmentacao, aplica One-Hot Encoding e executa o algoritmo Apriori.
```
python associacao_municipios.py
```
Saidas geradas:

* `regras_associacao_municipios.xlsx`: Relatorio contendo as regras encontradas, ordenadas por confianca e lift.
Metodologia e Parametros
K-Medoids (Clusterizacao)
Utilizado em substituicao ao K-Means por ser mais robusto a outliers e utilizar pontos reais (medoides) como centroides.
* `Metrica de Distancia`: Euclidiana.
* `Clusters (k)`: 4 grupos definidos apos analise de silhueta.
* `Features`: Log do PIB, Taxa de Alfabetizacao e Log da Densidade.
Apriori (Associacao)
Utilizado para encontrar padroes frequentes entre os atributos categoricos e os clusters definidos.
* `Suporte Minimo`: 0.05 (5%)
* `Confianca Minima`: 0.7 (70%)
* `Lift Minimo`: 1.0
Dependencias Principais
* `pandas`
* `numpy`
* `scikit-learn`
* `scikit-learn-extra (Implementacao do K-Medoids)`
* `mlxtend (Implementacao do Apriori)`
* `plotly (Visualizacao de dados)`
* `geopandas (Manipulacao geoespacial)`
* `XlsxWriter (Exportacao de relatorios)`