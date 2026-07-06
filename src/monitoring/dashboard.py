import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from logger import log

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ALERT_THRESHOLD = 10  # seuil d'alerte : 10% d'anomalies
PREDICTIONS_PATH = "data/predictions.csv"

app = Dash(__name__)
app.title = "Safran — Dashboard Monitoring Anomalies"

# ---------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ---------------------------------------------------------

def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Safran Data Systems", style={"color": "white", "margin": "0", "fontSize": "28px"}),
        html.P("Dashboard Monitoring — Détection d'Anomalies Moteurs", style={"color": "#1ABC9C", "margin": "5px 0 0 0"}),
    ], style={"backgroundColor": "#0D1B2A", "padding": "20px 30px"}),

    # Zone d'alerte
    html.Div(id="alert-zone"),

    # KPIs
    html.Div(id="kpis", style={"display": "flex", "gap": "20px", "padding": "20px 30px", "backgroundColor": "#F2F3F4"}),

    # Filtres
    html.Div([
        html.Label("Filtrer par unité moteur :", style={"fontWeight": "bold", "color": "#1B4F72"}),
        dcc.Dropdown(id="unit-filter", placeholder="Toutes les unités", multi=True,
            style={"width": "400px", "marginTop": "8px"}),
        html.Button("Rafraîchir", id="refresh-btn", n_clicks=0,
            style={"marginLeft": "20px", "backgroundColor": "#1ABC9C", "color": "white",
                   "border": "none", "padding": "10px 20px", "borderRadius": "5px", "cursor": "pointer"}),
    ], style={"padding": "15px 30px", "backgroundColor": "white", "borderBottom": "1px solid #D5D8DC"}),

    # Graphiques
    html.Div([
        html.Div([dcc.Graph(id="anomaly-timeline")], style={"width": "65%"}),
        html.Div([dcc.Graph(id="anomaly-pie")], style={"width": "33%"}),
    ], style={"display": "flex", "gap": "20px", "padding": "20px 30px"}),

    html.Div([
        html.Div([dcc.Graph(id="score-distribution")], style={"width": "50%"}),
        html.Div([dcc.Graph(id="anomaly-by-unit")], style={"width": "48%"}),
    ], style={"display": "flex", "gap": "20px", "padding": "0 30px 20px 30px"}),

    # Intervalle de refresh automatique
    dcc.Interval(id="interval", interval=30000, n_intervals=0),

], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#F8F9FA"})

# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------

@app.callback(
    Output("unit-filter", "options"),
    Input("interval", "n_intervals")
)
def update_unit_options(n):
    df = load_predictions()
    if df.empty or "unit" not in df.columns:
        return []
    units = sorted(df["unit"].unique())
    return [{"label": f"Unité {u}", "value": u} for u in units]

@app.callback(
    [Output("alert-zone", "children"),
     Output("kpis", "children"),
     Output("anomaly-timeline", "figure"),
     Output("anomaly-pie", "figure"),
     Output("score-distribution", "figure"),
     Output("anomaly-by-unit", "figure")],
    [Input("interval", "n_intervals"),
     Input("refresh-btn", "n_clicks"),
     Input("unit-filter", "value")]
)
def update_dashboard(n_intervals, n_clicks, selected_units):
    df = load_predictions()

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Aucune donnée disponible")
        return html.Div(), [], empty_fig, empty_fig, empty_fig, empty_fig

    # Filtrage par unité
    if selected_units:
        df = df[df["unit"].isin(selected_units)]

    total = len(df)
    anomalies = int(df["anomalie"].sum())
    normaux = total - anomalies
    taux = round(anomalies / total * 100, 1) if total > 0 else 0
    alert_banner = html.Div()
    if taux > ALERT_THRESHOLD:
        log(f"⚠️ ALERTE : taux d'anomalies à {taux}%, au-dessus du seuil de {ALERT_THRESHOLD}%")
        alert_banner = html.Div(
            f"⚠️ ALERTE — Taux d'anomalies élevé : {taux}% (seuil : {ALERT_THRESHOLD}%)",
            style={"backgroundColor": "#C0392B", "color": "white", "padding": "15px",
                "textAlign": "center", "fontWeight": "bold", "borderRadius": "5px"}
        )

    # ── KPIs
    kpi_style = {"backgroundColor": "white", "padding": "20px", "borderRadius": "8px",
                 "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "textAlign": "center", "flex": "1"}
    kpis = [
        html.Div([html.H2(str(total), style={"color": "#1B4F72", "margin": "0", "fontSize": "32px"}),
                  html.P("Total échantillons", style={"color": "#5D6D7E", "margin": "5px 0 0 0"})], style=kpi_style),
        html.Div([html.H2(str(normaux), style={"color": "#1E8449", "margin": "0", "fontSize": "32px"}),
                  html.P("Normaux", style={"color": "#5D6D7E", "margin": "5px 0 0 0"})], style=kpi_style),
        html.Div([html.H2(str(anomalies), style={"color": "#C0392B", "margin": "0", "fontSize": "32px"}),
                  html.P("Anomalies", style={"color": "#5D6D7E", "margin": "5px 0 0 0"})], style=kpi_style),
        html.Div([html.H2(f"{taux}%", style={"color": "#E67E22", "margin": "0", "fontSize": "32px"}),
                  html.P("Taux d'anomalies", style={"color": "#5D6D7E", "margin": "5px 0 0 0"})], style=kpi_style),
        html.Div([html.H2(f"{round(df['score_anomalie'].mean(), 3)}", style={"color": "#6C3483", "margin": "0", "fontSize": "32px"}),
                  html.P("Score moyen", style={"color": "#5D6D7E", "margin": "5px 0 0 0"})], style=kpi_style),
    ]

    # ── Timeline anomalies par cycle
    if "time" in df.columns and "unit" in df.columns:
        df_grouped = df.groupby(["unit", "time"])["anomalie"].sum().reset_index()
        df_sample = df_grouped[df_grouped["unit"].isin(df_grouped["unit"].unique()[:5])]
        fig_timeline = px.line(
            df_sample, x="time", y="anomalie", color="unit",
            title="Anomalies par cycle — Top 5 unités moteur",
            labels={"time": "Cycle", "anomalie": "Anomalies", "unit": "Unité"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_timeline.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            title_font=dict(size=14, color="#1B4F72"),
            legend_title="Unité moteur"
        )
    else:
        fig_timeline = go.Figure()

    # ── Pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Normal", "Anomalie"],
        values=[normaux, anomalies],
        hole=0.4,
        marker_colors=["#1E8449", "#C0392B"]
    )])
    fig_pie.update_layout(
        title="Répartition Normal / Anomalie",
        title_font=dict(size=14, color="#1B4F72"),
        paper_bgcolor="white"
    )

    # ── Distribution des scores
    fig_score = px.histogram(
        df, x="score_anomalie", color="statut",
        title="Distribution des scores d'anomalie",
        labels={"score_anomalie": "Score", "statut": "Statut"},
        color_discrete_map={"NORMAL": "#1E8449", "ANOMALIE": "#C0392B"},
        nbins=50
    )
    fig_score.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        title_font=dict(size=14, color="#1B4F72")
    )

    # ── Anomalies par unité
    if "unit" in df.columns:
        df_unit = df.groupby("unit")["anomalie"].sum().reset_index()
        df_unit = df_unit.sort_values("anomalie", ascending=False).head(20)
        fig_unit = px.bar(
            df_unit, x="unit", y="anomalie",
            title="Top 20 unités — Nombre d'anomalies",
            labels={"unit": "Unité moteur", "anomalie": "Nombre d'anomalies"},
            color="anomalie",
            color_continuous_scale=["#1E8449", "#E67E22", "#C0392B"]
        )
        fig_unit.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            title_font=dict(size=14, color="#1B4F72")
        )
    else:
        fig_unit = go.Figure()

    return alert_banner, kpis, fig_timeline, fig_pie, fig_score, fig_unit

# ---------------------------------------------------------
# LANCEMENT
# ---------------------------------------------------------

if __name__ == "__main__":
    log("🚀 Démarrage du dashboard de monitoring...")
    app.run(debug=True, port=8050)