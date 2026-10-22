import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Previsão de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("---")

# ============================================================
# 1. Geração do Dataset Sintético
# ============================================================
@st.cache_data
def gerar_dataset(n_registros=30):
    np.random.seed(42)
    
    velocidade = np.random.uniform(0.5, 5.0, n_registros)          # m/s
    maquinas = np.random.randint(1, 15, n_registros)                # quantidade
    temp = np.random.uniform(18.0, 35.0, n_registros)               # °C
    
    # Fórmula realista de consumo (com ruído)
    consumo = (
        15 * velocidade +
        8 * maquinas +
        2.5 * temp +
        np.random.normal(0, 8, n_registros)
    )
    consumo = np.clip(consumo, 30, None)  # evita valores negativos irreais
    
    df = pd.DataFrame({
        "Velocidade_Esteira": np.round(velocidade, 2),
        "Maquinas_Ativas": maquinas,
        "Temp_Ambiente": np.round(temp, 1),
        "Consumo_kWh": np.round(consumo, 1)
    })
    return df

df = gerar_dataset(30)

# ============================================================
# 2. Treinamento do Modelo
# ============================================================
@st.cache_resource
def treinar_modelo(dataframe):
    X = dataframe[["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]]
    y = dataframe["Consumo_kWh"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    score = modelo.score(X_test, y_test)
    return modelo, score, X.columns.tolist()

modelo, r2_score, features = treinar_modelo(df)

# ============================================================
# 3. Sidebar - Controles de Entrada
# ============================================================
st.sidebar.header("🔧 Parâmetros de Entrada")
st.sidebar.markdown("Ajuste os valores abaixo para simular o consumo:")

vel_esteira = st.sidebar.number_input(
    "Velocidade da Esteira (m/s)",
    min_value=0.5,
    max_value=5.0,
    value=2.5,
    step=0.1,
    format="%.1f"
)

maquinas_ativas = st.sidebar.number_input(
    "Máquinas Ativas",
    min_value=1,
    max_value=20,
    value=6,
    step=1
)

temp_ambiente = st.sidebar.number_input(
    "Temperatura Ambiente (°C)",
    min_value=15.0,
    max_value=40.0,
    value=25.0,
    step=0.5,
    format="%.1f"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**R² do modelo:** {r2_score:.3f}")

# ============================================================
# 4. Previsão
# ============================================================
entrada = np.array([[vel_esteira, maquinas_ativas, temp_ambiente]])
consumo_previsto = modelo.predict(entrada)[0]

# ============================================================
# 5. Tela Principal - Métrica em Destaque
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.metric(
        label="Consumo Previsto de Energia",
        value=f"{consumo_previsto:.1f} kWh",
        delta=None
    )

st.markdown("---")

# ============================================================
# 6. Gráfico de Contribuição das Variáveis (Coeficientes)
# ============================================================
st.subheader("📊 Contribuição de cada variável no modelo")

# Coeficientes do modelo LinearRegression
coeficientes = modelo.coef_
nomes_features = ["Velocidade Esteira", "Máquinas Ativas", "Temp. Ambiente"]

# Cria DataFrame para facilitar a plotagem
df_contrib = pd.DataFrame({
    "Variável": nomes_features,
    "Coeficiente": coeficientes
})

# Ordena por valor absoluto para melhor visualização
df_contrib["Abs_Coef"] = np.abs(df_contrib["Coeficiente"])
df_contrib = df_contrib.sort_values("Abs_Coef", ascending=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
cores = ["#2ecc71" if c > 0 else "#e74c3c" for c in df_contrib["Coeficiente"]]

bars = ax.barh(df_contrib["Variável"], df_contrib["Coeficiente"], color=cores, edgecolor="black", height=0.6)

ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coeficiente (impacto no Consumo_kWh)", fontsize=11)
ax.set_title("Importância / Contribuição das Variáveis (Regressão Linear)", fontsize=13, pad=15)
ax.grid(axis="x", alpha=0.3)

# Adiciona valor do coeficiente nas barras
for bar, valor in zip(bars, df_contrib["Coeficiente"]):
    ax.text(
        valor + (0.3 if valor >= 0 else -0.3),
        bar.get_y() + bar.get_height()/2,
        f"{valor:.2f}",
        va="center",
        ha="left" if valor >= 0 else "right",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
st.pyplot(fig)

# ============================================================
# 7. Informações extras (opcional e útil)
# ============================================================
with st.expander("📋 Ver dataset utilizado no treinamento"):
    st.dataframe(df, use_container_width=True)

with st.expander("ℹ️ Sobre o modelo"):
    st.markdown(f"""
    - **Algoritmo:** Regressão Linear (`sklearn.linear_model.LinearRegression`)
    - **Variáveis de entrada:** Velocidade_Esteira, Maquinas_Ativas, Temp_Ambiente
    - **Alvo:** Consumo_kWh
    - **R² no conjunto de teste:** `{r2_score:.3f}`
    - **Interpretação dos coeficientes:**  
      Quanto maior o valor absoluto do coeficiente, maior o impacto da variável no consumo previsto.
    """)
