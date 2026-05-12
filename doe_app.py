"""
Application d'Analyse de Plans d'Expérience (DOE)
Cours : Plans d'expérience — Projet Python / Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DOE Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  STYLE
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.block-container { padding-top: 2rem; }

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
}

div[data-testid="metric-container"] {
    background: #f7f7f2;
    border: 1px solid #e0e0d8;
    border-radius: 6px;
    padding: 0.6rem 1rem;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DOE Analyser")
    st.markdown("---")

    uploaded = st.file_uploader(
        "📂 Charger un fichier CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Le fichier doit contenir les facteurs (colonnes d'entrée) et les réponses (colonnes de sortie)."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres du modèle")
    poly_degree = st.selectbox("Degré de régression", [1, 2], index=1,
                               help="1 = linéaire, 2 = quadratique (RSM)")
    alpha_anova = st.slider("Seuil α (ANOVA)", 0.01, 0.10, 0.05, 0.01)

    st.markdown("---")
    st.caption("Projet DOE • Python + Streamlit")

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if uploaded is None:
    # Default demo dataset
    st.info("💡 Aucun fichier chargé — utilisation du **jeu de données démo** (procédé chimique : 4 facteurs, 2 réponses).")
    demo_data = {
        "Temperature": [160,200,160,200,160,200,160,200,160,200,160,200,160,200,160,200,180,180,180],
        "Pression":    [2,2,4,4,2,2,4,4,2,2,4,4,2,2,4,4,3,3,3],
        "Vitesse":     [100,100,100,100,200,200,200,200,100,100,100,100,200,200,200,200,150,150,150],
        "Concentration":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.75,0.75,0.75],
        "Rendement":   [72.3,78.6,69.8,81.2,74.5,83.1,71.2,85.7,68.4,75.9,66.1,79.3,70.8,80.4,67.9,83.6,77.2,76.8,77.5],
        "Purete":      [91.2,88.4,93.1,87.6,90.8,86.9,92.4,86.1,94.3,89.7,95.8,88.2,93.5,87.3,94.9,86.5,90.1,90.4,89.9],
    }
    df = pd.DataFrame(demo_data)
else:
    df = load_data(uploaded)

# ─────────────────────────────────────────
#  COLUMN SELECTION
# ─────────────────────────────────────────
st.title("🔬 Analyse de Plan d'Expérience")
st.markdown("<p class='section-label'>Design of Experiments — Analyse complète</p>", unsafe_allow_html=True)

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    factors = st.multiselect("📥 Variables d'entrée (facteurs)", numeric_cols,
                             default=numeric_cols[:-2] if len(numeric_cols) > 2 else numeric_cols[:1])
with col2:
    responses = st.multiselect("📤 Variables de sortie (réponses)", numeric_cols,
                               default=numeric_cols[-2:] if len(numeric_cols) >= 2 else numeric_cols[-1:])

if not factors or not responses:
    st.warning("⚠️ Veuillez sélectionner au moins un facteur et une réponse.")
    st.stop()

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tabs = st.tabs([
    "📊 Données",
    "📐 Modélisation",
    "📈 Visualisation",
    "🎯 Influence des facteurs",
    "📋 ANOVA",
    "🏆 Optimisation"
])

# ══════════════════════════════════════════
#  TAB 1 — DONNÉES
# ══════════════════════════════════════════
with tabs[0]:
    st.subheader("Aperçu des données")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expériences", len(df))
    m2.metric("Facteurs", len(factors))
    m3.metric("Réponses", len(responses))
    m4.metric("Valeurs manquantes", df.isnull().sum().sum())

    st.dataframe(df.style.background_gradient(subset=responses, cmap="YlGn"), use_container_width=True)

    st.markdown("### 📌 Statistiques descriptives")
    st.dataframe(df[factors + responses].describe().round(3), use_container_width=True)

    st.markdown("### 🔗 Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[factors + responses].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════
#  TAB 2 — MODÉLISATION
# ══════════════════════════════════════════
with tabs[1]:
    st.subheader("Modélisation par régression polynomiale")

    X = df[factors].values
    models = {}

    for resp in responses:
        y = df[resp].values
        pipeline = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False),
                                 LinearRegression())
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        models[resp] = {"pipeline": pipeline, "r2": r2, "rmse": rmse, "y_pred": y_pred}

        # Feature names
        pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
        pf.fit(X)
        feat_names = pf.get_feature_names_out(factors)
        coefs = pipeline.named_steps["linearregression"].coef_
        intercept = pipeline.named_steps["linearregression"].intercept_

        st.markdown(f"#### 🎯 Réponse : **{resp}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("Dégré", poly_degree)

        # Equation display
        eq_parts = [f"{intercept:.3f}"]
        for name, coef in zip(feat_names, coefs):
            sign = "+" if coef >= 0 else "-"
            eq_parts.append(f"{sign} {abs(coef):.3f}·{name}")
        st.code(f"{resp} = " + " ".join(eq_parts), language="")

        # Observed vs Predicted
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y, y_pred, color="#2563EB", edgecolors="white", s=70, zorder=3)
        lims = [min(y.min(), y_pred.min())-1, max(y.max(), y_pred.max())+1]
        ax.plot(lims, lims, "r--", lw=1.5, label="Idéal")
        ax.set_xlabel(f"{resp} observé"); ax.set_ylabel(f"{resp} prédit")
        ax.set_title(f"Observé vs Prédit — {resp}")
        ax.legend(); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("---")

# ══════════════════════════════════════════
#  TAB 3 — VISUALISATION
# ══════════════════════════════════════════
with tabs[2]:
    st.subheader("Visualisation des effets")

    resp_viz = st.selectbox("Réponse à visualiser", responses, key="viz_resp")
    y_viz = df[resp_viz].values

    # Effect plots for each factor
    st.markdown("#### Effet individuel de chaque facteur")
    ncols = min(len(factors), 3)
    nrows = (len(factors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    for i, fac in enumerate(factors):
        x_vals = df[fac].values
        axes[i].scatter(x_vals, y_viz, color="#10B981", edgecolors="white", s=60, zorder=3)
        m, b = np.polyfit(x_vals, y_viz, 1)
        xline = np.linspace(x_vals.min(), x_vals.max(), 100)
        axes[i].plot(xline, m*xline+b, color="#EF4444", lw=2)
        axes[i].set_xlabel(fac); axes[i].set_ylabel(resp_viz)
        axes[i].set_title(f"{resp_viz} vs {fac}")
        axes[i].grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Interaction plots (pairs)
    if len(factors) >= 2:
        st.markdown("#### Interactions entre facteurs (2 à 2)")
        pair = st.selectbox("Paire de facteurs", [f"{a} × {b}" for a, b in combinations(factors, 2)], key="pair")
        fac_a, fac_b = pair.split(" × ")

        levels_b = sorted(df[fac_b].unique())
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.tab10.colors
        for idx, lev in enumerate(levels_b):
            mask = df[fac_b] == lev
            sub = df[mask].sort_values(fac_a)
            ax.plot(sub[fac_a], sub[resp_viz], marker="o", color=colors[idx % 10],
                    label=f"{fac_b}={lev}", lw=2)
        ax.set_xlabel(fac_a); ax.set_ylabel(resp_viz)
        ax.set_title(f"Interaction : {fac_a} × {fac_b} → {resp_viz}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Surface de réponse (si 2 facteurs sélectionnés)
    if len(factors) >= 2:
        st.markdown("#### Surface de réponse 3D")
        col_x = st.selectbox("Axe X", factors, key="surf_x")
        col_y_axis = st.selectbox("Axe Y", [f for f in factors if f != col_x], key="surf_y")

        model_surf = models[resp_viz]["pipeline"]
        x_range = np.linspace(df[col_x].min(), df[col_x].max(), 40)
        y_range = np.linspace(df[col_y_axis].min(), df[col_y_axis].max(), 40)
        XX, YY = np.meshgrid(x_range, y_range)

        mean_vals = df[factors].mean()
        grid_data = pd.DataFrame({f: mean_vals[f] * np.ones(XX.size) for f in factors})
        grid_data[col_x] = XX.ravel()
        grid_data[col_y_axis] = YY.ravel()
        ZZ = model_surf.predict(grid_data[factors].values).reshape(XX.shape)

        fig = plt.figure(figsize=(8, 5))
        ax3d = fig.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.85, edgecolor="none")
        ax3d.scatter(df[col_x], df[col_y_axis], df[resp_viz], color="red", s=40, zorder=5)
        ax3d.set_xlabel(col_x); ax3d.set_ylabel(col_y_axis); ax3d.set_zlabel(resp_viz)
        ax3d.set_title(f"Surface de réponse : {resp_viz}")
        fig.colorbar(surf, shrink=0.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
#  TAB 4 — INFLUENCE DES FACTEURS
# ══════════════════════════════════════════
with tabs[3]:
    st.subheader("Analyse de l'influence des facteurs")

    resp_inf = st.selectbox("Réponse", responses, key="inf_resp")

    # Standardized coefficients
    from sklearn.preprocessing import StandardScaler
    X_raw = df[factors].values
    y_raw = df[resp_inf].values
    scaler_X = StandardScaler(); scaler_y = StandardScaler()
    X_std = scaler_X.fit_transform(X_raw)
    y_std = scaler_y.fit_transform(y_raw.reshape(-1,1)).ravel()

    pipeline_std = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        LinearRegression()
    )
    pipeline_std.fit(X_std, y_std)
    std_coefs = pipeline_std.named_steps["linearregression"].coef_
    importance = pd.DataFrame({"Facteur": factors, "Coefficient standardisé": std_coefs})
    importance["Importance (%)"] = (np.abs(std_coefs) / np.abs(std_coefs).sum() * 100).round(1)
    importance = importance.sort_values("Importance (%)", ascending=False).reset_index(drop=True)
    importance["Rang"] = range(1, len(importance)+1)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Tableau d'importance")
        st.dataframe(importance[["Rang","Facteur","Coefficient standardisé","Importance (%)"]].style
                     .background_gradient(subset=["Importance (%)"], cmap="Blues"), use_container_width=True)

    with c2:
        st.markdown("#### Diagramme de Pareto")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors_bar = ["#2563EB" if c >= 0 else "#EF4444"
                      for c in importance["Coefficient standardisé"]]
        bars = ax.barh(importance["Facteur"], np.abs(importance["Coefficient standardisé"]),
                       color=colors_bar, edgecolor="white")
        ax.set_xlabel("| Coefficient standardisé |")
        ax.set_title(f"Importance des facteurs → {resp_inf}")
        ax.invert_yaxis()
        for bar, val in zip(bars, importance["Importance (%)"]):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{val}%", va="center", fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Interprétation automatique")
    top = importance.iloc[0]
    bot = importance.iloc[-1]
    sign_top = "positivement" if importance.iloc[0]["Coefficient standardisé"] > 0 else "négativement"
    st.info(
        f"🔑 **{top['Facteur']}** est le facteur le plus influent ({top['Importance (%)']:.1f}% de l'effet total) "
        f"et agit **{sign_top}** sur **{resp_inf}**.\n\n"
        f"📉 **{bot['Facteur']}** est le facteur le moins influent ({bot['Importance (%)']:.1f}%)."
    )

# ══════════════════════════════════════════
#  TAB 5 — ANOVA
# ══════════════════════════════════════════
with tabs[4]:
    st.subheader("Analyse ANOVA")
    st.markdown(f"Seuil de signification : **α = {alpha_anova}**")

    resp_anova = st.selectbox("Réponse à analyser", responses, key="anova_resp")

    # Build formula
    formula_terms = " + ".join([f"Q('{f}')" if " " in f else f for f in factors])
    formula = f"`{resp_anova}` ~ {formula_terms}"

    try:
        model_ols = ols(formula, data=df).fit()
        anova_table = anova_lm(model_ols, typ=2)
        anova_table = anova_table.reset_index().rename(columns={"index": "Source"})
        anova_table["Significatif"] = anova_table["PR(>F)"].apply(
            lambda p: "✅ Oui" if pd.notna(p) and p < alpha_anova else ("➖ —" if pd.isna(p) else "❌ Non")
        )

        st.markdown("#### Tableau ANOVA (Type II)")
        styled = anova_table.style.applymap(
            lambda v: "background-color: #d1fae5; font-weight: bold" if v == "✅ Oui" else "",
            subset=["Significatif"]
        ).format({"sum_sq": "{:.4f}", "df": "{:.0f}", "F": "{:.3f}", "PR(>F)": "{:.4f}"})
        st.dataframe(styled, use_container_width=True)

        # F-value bar chart
        anova_factors = anova_table.dropna(subset=["F"])
        anova_factors = anova_factors[anova_factors["Source"] != "Residual"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = ["#10B981" if row["PR(>F)"] < alpha_anova else "#94A3B8"
                      for _, row in anova_factors.iterrows()]
        ax.bar(anova_factors["Source"], anova_factors["F"], color=bar_colors, edgecolor="white")
        ax.axhline(y=anova_factors["F"].mean(), color="red", linestyle="--", lw=1.5, label="Moyenne F")
        ax.set_ylabel("Valeur F")
        ax.set_title(f"Valeurs F par facteur → {resp_anova}")
        ax.set_xticklabels(anova_factors["Source"], rotation=30, ha="right")
        ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close()

        # Summary
        sig_factors = anova_table[anova_table["Significatif"] == "✅ Oui"]["Source"].tolist()
        if sig_factors:
            st.success(f"✅ Facteurs significatifs (p < {alpha_anova}) : **{', '.join(sig_factors)}**")
        else:
            st.warning(f"⚠️ Aucun facteur significatif détecté au seuil α = {alpha_anova}.")

        st.markdown("#### Résumé du modèle OLS")
        st.text(model_ols.summary().as_text())

    except Exception as e:
        st.error(f"Erreur ANOVA : {e}")

# ══════════════════════════════════════════
#  TAB 6 — OPTIMISATION MULTI-OBJECTIF
# ══════════════════════════════════════════
with tabs[5]:
    st.subheader("Optimisation multi-objectif")

    if len(responses) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 réponses pour l'optimisation multi-objectif.")
        st.stop()

    st.markdown("""
    **Méthode utilisée : Désirabilité globale (D)**
    
    Chaque réponse est normalisée entre 0 (indésirable) et 1 (optimal), 
    puis la désirabilité globale est calculée comme la moyenne géométrique.
    """)

    st.markdown("#### Définir les objectifs")
    goals = {}
    cols_obj = st.columns(len(responses))
    for i, resp in enumerate(responses):
        with cols_obj[i]:
            st.markdown(f"**{resp}**")
            goal = st.selectbox("Objectif", ["Maximiser", "Minimiser", "Cible"],
                                key=f"goal_{resp}")
            target_val = None
            if goal == "Cible":
                target_val = st.number_input("Valeur cible", value=float(df[resp].mean()),
                                             key=f"target_{resp}")
            goals[resp] = {"goal": goal, "target": target_val}

    # Generate candidate solutions via grid search
    n_grid = 10
    grid_vals = {f: np.linspace(df[f].min(), df[f].max(), n_grid) for f in factors}
    import itertools
    grid_points = list(itertools.product(*[grid_vals[f] for f in factors]))
    X_cand = np.array(grid_points)

    # Predict all responses
    preds = {}
    for resp in responses:
        preds[resp] = models[resp]["pipeline"].predict(X_cand)

    # Compute desirability for each response
    desirabilities = {}
    for resp in responses:
        y_min = preds[resp].min(); y_max = preds[resp].max()
        p = preds[resp]
        g = goals[resp]["goal"]
        if g == "Maximiser":
            d = (p - y_min) / (y_max - y_min + 1e-9)
        elif g == "Minimiser":
            d = (y_max - p) / (y_max - y_min + 1e-9)
        else:
            t = goals[resp]["target"]
            d = 1 - np.abs(p - t) / (max(abs(y_max-t), abs(y_min-t)) + 1e-9)
        desirabilities[resp] = np.clip(d, 0, 1)

    # Global desirability (geometric mean)
    D_global = np.ones(len(X_cand))
    for resp in responses:
        D_global *= desirabilities[resp]
    D_global = D_global ** (1 / len(responses))

    best_idx = np.argmax(D_global)
    best_X = X_cand[best_idx]
    best_preds = {resp: preds[resp][best_idx] for resp in responses}
    best_D = D_global[best_idx]

    st.markdown("#### 🏆 Solution optimale trouvée")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Facteurs optimaux :**")
        opt_df = pd.DataFrame({"Facteur": factors, "Valeur optimale": best_X.round(3)})
        st.dataframe(opt_df, use_container_width=True)
    with c2:
        st.markdown("**Réponses prédites :**")
        resp_df = pd.DataFrame({
            "Réponse": list(best_preds.keys()),
            "Valeur prédite": [round(v, 3) for v in best_preds.values()]
        })
        st.dataframe(resp_df, use_container_width=True)

    st.metric("🎯 Désirabilité globale D", f"{best_D:.4f}", help="1.0 = parfait, 0 = inacceptable")

    # Pareto front (2 first responses)
    r1, r2 = responses[0], responses[1]
    p1, p2 = preds[r1], preds[r2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter Pareto
    sc = axes[0].scatter(p1, p2, c=D_global, cmap="RdYlGn", s=20, alpha=0.7)
    axes[0].scatter(best_preds[r1], best_preds[r2], color="blue", s=150,
                    marker="*", zorder=5, label="Optimum")
    plt.colorbar(sc, ax=axes[0], label="Désirabilité D")
    axes[0].set_xlabel(r1); axes[0].set_ylabel(r2)
    axes[0].set_title("Espace des solutions (front de Pareto)")
    axes[0].legend()

    # Désirabilité par facteur (premier facteur)
    fac_main = factors[0]
    fac_idx = factors.index(fac_main)
    axes[1].scatter(X_cand[:, fac_idx], D_global, c=D_global, cmap="RdYlGn", s=15, alpha=0.6)
    axes[1].axvline(best_X[fac_idx], color="blue", lw=2, linestyle="--", label=f"Optimal = {best_X[fac_idx]:.2f}")
    axes[1].set_xlabel(fac_main); axes[1].set_ylabel("Désirabilité D")
    axes[1].set_title(f"Désirabilité vs {fac_main}")
    axes[1].legend()

    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Interprétation")
    st.success(
        f"La solution optimale atteint une désirabilité globale de **{best_D:.3f}**. "
        f"Les conditions optimales sont : "
        + ", ".join([f"**{f}** = {v:.2f}" for f, v in zip(factors, best_X)])
        + f". Ces conditions permettent de prédire "
        + ", ".join([f"**{resp}** ≈ {val:.2f}" for resp, val in best_preds.items()])
        + "."
    )
