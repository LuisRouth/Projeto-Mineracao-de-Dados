Este projeto consiste em uma aplicação de Ciência de Dados voltada para a análise de clusters de municípios brasileiros. Utilizando algoritmos de Aprendizado de Máquina Não-Supervisionado (K-Medoids), o sistema agrupa cidades com características similares baseando-se em indicadores oficiais do IBGE.

O objetivo é identificar padrões regionais e perfis socioeconômicos (ex: Vulneráveis, Emergentes, Consolidados e Dinâmicos) para auxiliar em estudos demográficos ou definição de políticas públicas.

## Fonte dos Dados

Os dados brutos foram extraídos do SIDRA (Sistema IBGE de Recuperação Automática). Foram utilizadas as seguintes tabelas oficiais:

*   **Tabela 5938:** Produto Interno Bruto (PIB) a preços correntes (Utilizado para o cálculo do PIB per Capita).
*   **Tabela 1383:** Taxa de alfabetização das pessoas de 15 anos ou mais de idade (Dados censitários/amostrais).
*   **Tabela 1301:** Densidade demográfica (Habitantes por km²).

## Tecnologias Utilizadas

O projeto foi desenvolvido em **Python 3.8+** utilizando as seguintes bibliotecas para manipulação, modelagem e visualização:

*   **Pandas & NumPy:** Processamento de dados e álgebra linear.
*   **Scikit-Learn:** Pré-processamento (StandardScaler), redução de dimensionalidade (PCA) e métricas de validação (Silhouette Score, Calinski-Harabasz).
*   **PyClustering:** Implementação do algoritmo K-Medoids (Partitioning Around Medoids).
*   **Plotly (Express & Graph Objects):** Visualizações interativas para web e análise exploratória.
*   **GeoPandas:** Manipulação de dados geoespaciais para plotagem do mapa do Brasil.

## Metodologia

O pipeline de dados segue as etapas descritas abaixo:

1.  **Carga e Limpeza:** Importação dos arquivos Excel oriundos do SIDRA e unificação (merge) através do código do município (Código IBGE).
2.  **Tratamento de Outliers:** Remoção de dados discrepantes utilizando o método do Intervalo Interquartil (IQR) para evitar distorções na clusterização.
3.  **Normalização:** Padronização dos dados (Média 0 e Desvio Padrão 1) para garantir que variáveis com grandezas diferentes (ex: PIB vs Taxa de Alfabetização) tenham o mesmo peso no modelo.
4.  **Definição de K:** Análise de Silhueta para determinar o número ideal de grupos.
5.  **Modelagem K-Medoids:** Execução do algoritmo utilizando medóides (pontos reais do dataset) como centros, o que oferece maior robustez contra ruídos se comparado ao K-Means.
6.  **Visualização:** Geração de mapas, gráficos de dispersão (PCA) e radares de perfil.

## Estrutura do Projeto e Execução

Para executar o script corretamente, é necessário manter a estrutura de diretórios abaixo, onde a pasta `dados_brutos` contém os arquivos extraídos do SIDRA.

### Estrutura de Pastas

```
/
├── ProjetoIBGE.py
├── ProjetoIBGE.ipynb
├── README.md
└── dados_brutos/
    ├── pib.xlsx            (Dados da Tabela 5938)
    ├── alfabetizacao.xlsx  (Dados da Tabela 1383)
    └── densidade.xlsx      (Dados da Tabela 1301)
```

### Pré-requisitos
*   **Python 3.8+** instalado.
*   Recomenda-se o uso de um ambiente virtual (`venv`) para isolar as dependências.

### 1. Instalação das Dependências
O projeto conta agora com um arquivo de requisitos otimizado. No terminal, execute:

```
pip install -r requirements.txt
```
### 2. Como Rodar
Você pode executar o projeto de duas formas, dependendo da sua preferência:
Opção A: Via Script Python (Terminal)
Ideal para gerar os resultados e visualizações de uma só vez. O navegador abrirá automaticamente com os gráficos.
```
python ProjetoIBGE.py
```
Opção B: Via Jupyter Notebook (Interativo)
Ideal para análises exploratórias passo a passo. Se você tiver o Jupyter ou VS Code instalado:
```
jupyter notebook ProjetoIBGE.ipynb
```

Resultados Gerados
Após a execução, o script exibe visualizações interativas no navegador e cria uma pasta chamada resultados contendo os seguintes arquivos CSV para análise posterior:
01_dados_completos_segmentacao.csv: Base de dados completa contendo as métricas originais, o cluster atribuído e a identificação se o município é um medóide.
02_resumo_perfis_medios.csv: Tabela com as médias dos indicadores (PIB, Alfabetização, Densidade) por perfil (cluster).
03_municipios_representantes.csv: Lista dos municípios medóides, que são os exemplos mais representativos (o centro matemático) de cada perfil.