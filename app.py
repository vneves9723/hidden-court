from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import METRICS_PATH, PROSPECTS_PATH


st.set_page_config(page_title="Hidden Court", page_icon="🏀", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #f4f4f5; }
    [data-testid="stMetric"] { background: #151c31; border: 1px solid #28324f;
        border-radius: 14px; padding: 14px; }
    h1, h2, h3 { color: #f8fafc; }
    .accent { color: #ff8a1f; }
    .small-note { color: #aab3c5; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    prospects = pd.read_csv(PROSPECTS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return prospects, metrics


prospects, metrics = load_data()

st.title("🏀 Hidden Court")
st.markdown(
    "### Um modelo de scouting para encontrar jovens com potencial de *breakout*"
)
st.markdown(
    '<p class="small-note">O ranking combina probabilidade de breakout, crescimento previsto, '
    "eficiência e oportunidade atual. Use como ponto de partida para vídeo e scouting humano.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filtros de scouting")
    max_age = st.slider("Idade máxima", 20, 25, 25)
    min_games = st.slider("Mínimo de jogos", 20, int(prospects["gp"].max()), 25)
    max_minutes = st.slider("Máximo de minutos/jogo", 10, 28, 28)
    teams = sorted(prospects["team"].dropna().unique())
    selected_teams = st.multiselect("Times", teams, default=teams)

filtered = prospects[
    (prospects["age"] <= max_age)
    & (prospects["gp"] >= min_games)
    & (prospects["mpg"] <= max_minutes)
    & (prospects["team"].isin(selected_teams))
].copy()

top = filtered.iloc[0] if not filtered.empty else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidatos", len(filtered))
c2.metric("Maior Hidden Gem Score", f"{top['hidden_gem_score']:.1f}" if top is not None else "—")
c3.metric("Melhor prob. de breakout", f"{filtered['breakout_probability'].max():.0%}" if len(filtered) else "—")
c4.metric("MAE da projeção de PPG", f"{metrics['next_ppg_mae']:.2f}")

if filtered.empty:
    st.info("Nenhum jogador atende aos filtros. Amplie os critérios ou selecione um time.")
    st.stop()

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Ranking de talentos subvalorizados")
    table = filtered[
        [
            "player_name",
            "team",
            "age",
            "mpg",
            "ppg",
            "projected_ppg",
            "breakout_probability",
            "hidden_gem_score",
        ]
    ].head(25)
    table = table.copy()
    table["breakout_probability"] *= 100
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "player_name": "Jogador",
            "team": "Time",
            "age": "Idade",
            "mpg": st.column_config.NumberColumn("MIN", format="%.1f"),
            "ppg": st.column_config.NumberColumn("PTS", format="%.1f"),
            "projected_ppg": st.column_config.NumberColumn("PTS projetados", format="%.1f"),
            "breakout_probability": st.column_config.ProgressColumn(
                "Sinal breakout (%)", min_value=0, max_value=100, format="%.0f%%"
            ),
            "hidden_gem_score": st.column_config.ProgressColumn(
                "Hidden Gem", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

with right:
    st.subheader("Produção atual × potencial")
    chart = px.scatter(
        filtered,
        x="mpg",
        y="projected_ppg_growth",
        size="hidden_gem_score",
        color="breakout_probability",
        hover_name="player_name",
        hover_data={"team": True, "age": True, "hidden_gem_score": ":.1f"},
        color_continuous_scale=["#334155", "#ff8a1f", "#ffd166"],
        labels={
            "mpg": "Minutos por jogo",
            "projected_ppg_growth": "Crescimento projetado de PPG",
            "breakout_probability": "Probabilidade",
        },
    )
    chart.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#11182a",
        font_color="#e5e7eb",
        coloraxis_colorbar_tickformat=".0%",
    )
    st.plotly_chart(chart, width="stretch")

st.divider()
st.subheader("Ficha do jogador")
choices = filtered["player_name"].tolist() or prospects["player_name"].tolist()
selected_name = st.selectbox("Escolha um candidato", choices)
player = prospects.loc[prospects["player_name"] == selected_name].iloc[0]

p1, p2 = st.columns([1, 1.2])
with p1:
    st.markdown(f"## {player['player_name']} · {player['team']}")
    st.write(f"**Por que apareceu:** {player['model_reason']}")
    a, b, c = st.columns(3)
    a.metric("Hidden Gem", f"{player['hidden_gem_score']:.1f}")
    b.metric("Breakout", f"{player['breakout_probability']:.0%}")
    c.metric("PTS projetados", f"{player['projected_ppg']:.1f}", f"{player['projected_ppg_growth']:+.1f}")
    st.caption(
        "A projeção considera somente estatísticas agregadas. Lesões, trocas, elenco, função tática "
        "e evolução técnica precisam ser avaliados por scouting humano."
    )

with p2:
    labels = ["PTS/36", "AST/36", "REB/36", "TS%", "AST/TOV"]
    values = [
        min(player["pts_per36"] / 25, 1) * 100,
        min(player["ast_per36"] / 9, 1) * 100,
        min(player["reb_per36"] / 12, 1) * 100,
        min(player["ts_pct"] / 0.68, 1) * 100,
        min(player["ast_tov"] / 4, 1) * 100,
    ]
    radar = go.Figure(
        go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", line_color="#ff8a1f")
    )
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor="#11182a"),
        paper_bgcolor="#0b1020",
        font_color="#e5e7eb",
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    st.plotly_chart(radar, width="stretch")
    st.caption("Escalas ilustrativas: 25 PTS/36, 9 AST/36, 12 REB/36, 68% TS e 4 AST/TOV equivalem a 100. Não são percentis; valores acima são limitados a 100.")

st.caption("As saídas de breakout não foram calibradas como probabilidades reais. O projeto prioriza candidatos; não mede valor de contrato nem comprova que um jogador é subvalorizado.")

with st.expander("Como o modelo foi avaliado"):
    st.write(
        f"O teste final usou **{metrics['holdout_season']}** como temporada de entrada e a temporada "
        "seguinte como resultado. Nenhum dado futuro entrou nas variáveis do modelo."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("Lift no top 10%", f"{metrics['lift_top_decile']:.2f}×")
    m3.metric("MAE PPG", f"{metrics['next_ppg_mae']:.2f}")
