import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# Configuração da página
# --------------------------------------------------
st.set_page_config(
    page_title="Previsão de Consumo Industrial",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# Geração do dataset sintético
# --------------------------------------------------
dados = {
    "Velocidade_Esteira": [
        20, 25, 30, 35, 40, 45, 50, 55,
        60, 65, 70, 75, 80, 85, 90, 95,
        100, 105, 110, 115, 120, 125, 130, 135
    ],
    "Maquinas_Ativas": [
        1, 2, 2, 3, 3, 4, 4, 5,
        5, 6, 6, 7, 7, 8, 8, 9,
        9, 10, 10, 11, 11, 12, 12, 13
    ],
    "Temp_Ambiente": [
        18, 19, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42
    ],
    "Consumo_kWh": [
        72, 82, 91, 108, 118, 135, 143, 160,
        169, 187, 195, 212, 222, 240, 249, 267,
        276, 294, 303, 321, 330, 348, 357, 375
    ]
}

df = pd.DataFrame(dados)


# --------------------------------------------------
# Treinamento do modelo de regressão
# --------------------------------------------------
variaveis_entrada = [
    "Velocidade_Esteira",
    "Maquinas_Ativas",
    "Temp_Ambiente"
]

variavel_alvo = "Consumo_kWh"

X = df[variaveis_entrada]
y = df[variavel_alvo]

modelo = LinearRegression()
modelo.fit(X, y)


# --------------------------------------------------
# Sidebar com entradas do usuário
# --------------------------------------------------
st.sidebar.header("Parâmetros da fábrica")

velocidade = st.sidebar.number_input(
    "Velocidade da esteira",
    min_value=0.0,
    max_value=300.0,
    value=80.0,
    step=1.0
)

maquinas = st.sidebar.number_input(
    "Máquinas ativas",
    min_value=0,
    max_value=50,
    value=8,
    step=1
)

temperatura = st.sidebar.number_input(
    "Temperatura ambiente (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0,
    step=0.5
)


# --------------------------------------------------
# Previsão
# --------------------------------------------------
entrada_usuario = pd.DataFrame({
    "Velocidade_Esteira": [velocidade],
    "Maquinas_Ativas": [maquinas],
    "Temp_Ambiente": [temperatura]
})

consumo_previsto = modelo.predict(entrada_usuario)[0]


# --------------------------------------------------
# Interface principal
# --------------------------------------------------
st.title("Painel de Previsão de Consumo de Energia Industrial")
st.write(
    "Informe os parâmetros operacionais na barra lateral para estimar "
    "o consumo de energia da fábrica."
)

st.subheader("Consumo estimado")

st.metric(
    label="Consumo previsto",
    value=f"{consumo_previsto:.2f} kWh"
)


# --------------------------------------------------
# Exibição dos dados utilizados
# --------------------------------------------------
with st.expander("Visualizar dataset sintético"):
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------
# Gráfico de contribuição das variáveis
# --------------------------------------------------
st.subheader("Contribuição das variáveis para a previsão")

# A contribuição é calculada como:
# coeficiente da variável x valor informado pelo usuário
coeficientes = modelo.coef_

contribuicoes = entrada_usuario.iloc[0].values * coeficientes

df_contribuicoes = pd.DataFrame({
    "Variável": variaveis_entrada,
    "Contribuição": contribuicoes
})

fig, ax = plt.subplots(figsize=(10, 5))

cores = [
    "#1f77b4" if valor >= 0 else "#d62728"
    for valor in df_contribuicoes["Contribuição"]
]

ax.bar(
    df_contribuicoes["Variável"],
    df_contribuicoes["Contribuição"],
    color=cores
)

ax.set_title("Contribuição de cada variável para o consumo previsto")
ax.set_xlabel("Variáveis de entrada")
ax.set_ylabel("Contribuição estimada em kWh")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=15)
plt.tight_layout()

st.pyplot(fig)


# --------------------------------------------------
# Informações do modelo
# --------------------------------------------------
st.subheader("Informações do modelo")

coluna1, coluna2 = st.columns(2)

with coluna1:
    st.write("**Intercepto:**")
    st.write(f"{modelo.intercept_:.4f}")

with coluna2:
    st.write("**Coeficientes:**")
    coeficientes_formatados = pd.DataFrame({
        "Variável": variaveis_entrada,
        "Coeficiente": modelo.coef_
    })
    st.dataframe(coeficientes_formatados, hide_index=True)
