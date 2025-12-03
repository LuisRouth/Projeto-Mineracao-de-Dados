import pandas as pd
import os

from mlxtend.frequent_patterns import apriori, association_rules 

print("=" * 80)
print("MINERAÇÃO DE REGRAS DE ASSOCIAÇÃO - MUNICÍPIOS")
print("=" * 80)

# ==============================================================================
# 1. CARREGAR A SEGMENTAÇÃO PRODUZIDA PELO SCRIPT DE K-MEDOIDS
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
arquivo_segmentacao = os.path.join(base_dir, "resultado_mineracao_municipios.xlsx")
if not os.path.exists(arquivo_segmentacao):
    raise FileNotFoundError(f"Arquivo de segmentação não encontrado: {arquivo_segmentacao}")

print("[1/4] Lendo base de municípios segmentados")
df = pd.read_excel(arquivo_segmentacao, sheet_name="Segmentacao")
colunas_esperadas = [
    "codigo_ibge", "municipio", "uf", "perfil_municipio",
    "pib", "taxa_alfabetizacao", "densidade", "e_medoide",
    "faixa_pib", "faixa_alfabetizacao", "faixa_densidade"
]
faltantes = [c for c in colunas_esperadas if c not in df.columns]
if faltantes:
    raise ValueError(f"Colunas faltando na segmentação: {faltantes}")

print(f"Total de municípios na base: {len(df)}")

# ==============================================================================
# 2. MONTAR A BASE TRANSACIONAL (ONE-HOT) PARA APRIORI
# ==============================================================================

print("[2/4] Construindo itens (one-hot encoding)")
df_categorico = df[[
    "codigo_ibge",
    "uf",
    "perfil_municipio",
    "faixa_pib",
    "faixa_alfabetizacao",
    "faixa_densidade"
]].copy()
itens = pd.get_dummies(
    df_categorico[["uf", "perfil_municipio", "faixa_pib", "faixa_alfabetizacao", "faixa_densidade"]],
    prefix=["UF", "PERFIL", "PIB", "ALFA", "DENS"],
    prefix_sep="_"
)
print(f"Total de itens criados: {itens.shape[1]}")

# ==============================================================================
# 3. MINERAR CONJUNTOS FREQUENTES E REGRAS (APRIORI)
# ==============================================================================

print(">>> [3/4] Rodando Apriori para encontrar conjuntos frequentes...")
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.7
MIN_LIFT = 1.0

frequent_itemsets = apriori(
    itens,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

print(f"Conjuntos frequentes encontrados: {len(frequent_itemsets)}")

print(">>> Gerando regras de associação...")
regras = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

print(f"Regras geradas (antes de filtrar): {len(regras)}")

# ==============================================================================
# 4. FILTRAR REGRAS ÚTEIS PARA O PROJETO E EXPORTAR
# ==============================================================================

print(">>> [4/4] Filtrando regras com foco em perfis de desenvolvimento...")
mascara_consequente_perfil = regras['consequents'].apply(
    lambda s: any(str(item).startswith("PERFIL_") for item in s)
)

regras_filtradas = regras[mascara_consequente_perfil].copy()
regras_filtradas['tam_antecedente'] = regras_filtradas['antecedents'].apply(len)
regras_filtradas = regras_filtradas[regras_filtradas['tam_antecedente'] <= 3]
regras_filtradas = regras_filtradas[regras_filtradas['lift'] >= MIN_LIFT]
regras_filtradas = regras_filtradas.sort_values(
    by=["confidence", "support"],
    ascending=[False, False]
)

print(f"Regras após filtragem: {len(regras_filtradas)}")
def formatar_conjunto(conjunto):
    return ", ".join(sorted(list(conjunto)))

regras_filtradas['antecedents_str'] = regras_filtradas['antecedents'].apply(formatar_conjunto)
regras_filtradas['consequents_str'] = regras_filtradas['consequents'].apply(formatar_conjunto)

colunas_export = [
    "antecedents_str",
    "consequents_str",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "tam_antecedente"
]
arquivo_regras = os.path.join(base_dir, "regras_associacao_municipios.xlsx")
regras_filtradas[colunas_export].to_excel(arquivo_regras, index=False)

print(f"\nRegras de associação salvas em: {arquivo_regras}")
print("=" * 80)
