"""
Application d'Analyse de Plans d'Expérience (DOE)
Projet : Impression 3D composite — Éprouvettes avec fibres
Cours : Plans d'expérience — Python / Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="DOE — Composite 3D", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.block-container { padding-top: 2rem; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; letter-spacing: 0.05em; }
div[data-testid="metric-container"] { background:#f7f7f2; border:1px solid #e0e0d8; border-radius:6px; padding:0.6rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 DOE — Composite 3D")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Charger CSV / Excel", type=["csv","xlsx","xls"])
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    poly_degree = st.selectbox("Degré régression", [1, 2], index=1)
    alpha_anova = st.slider("Seuil α (ANOVA)", 0.01, 0.10, 0.05, 0.01)
    st.markdown("---")
    st.caption("Projet DOE • Python + Streamlit")

# ─────────────────────────────────────────────────────
#  LOAD DATA
@st.cache_data
def load_file(file):
    """Détecte le type de fichier par magic bytes et gère les doubles en-têtes Excel."""
    import io
    raw = file.read()

    # Magic bytes Excel : PK\x03\x04 (xlsx) ou \xd0\xcf (xls legacy)
    if raw[:4] in (b'PK\x03\x04', b'\xd0\xcf\x11\xe0'):
        # Lecture normale d'abord
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        # Si la 1ère ligne ressemble à des codes courts (AP1, EP2...) → double header
        # On relit en utilisant la 2ème ligne comme en-tête
        first_cols = [str(c).strip() for c in df.columns]
        looks_like_codes = all(len(c) <= 5 and c.replace("_","").isalnum() for c in first_cols)
        if looks_like_codes:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl", header=1)
        return df

    # CSV : essai avec plusieurs séparateurs
    for sep in [",", ";", "\t"]:
        try:
            df_try = pd.read_csv(io.BytesIO(raw), sep=sep)
            if df_try.shape[1] > 1:
                return df_try
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw))

# ─────────────────────────────────────────────────────

@st.cache_data
def load_default():
    return pd.DataFrame({
        "filling_rate":       [50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,50,50,50,100,100,100],
        "Nb_Fiber_Layers":    [0,0,0,2,2,2,4,4,4,0,0,0,2,2,2,4,4,4,0,2,4,0,2,4],
        "Fiber_Type":         ["Carbon","Kevlar","HSHT Fiberglass","Carbon","Kevlar","HSHT Fiberglass",
                               "Carbon","Kevlar","HSHT Fiberglass","Carbon","Kevlar","HSHT Fiberglass",
                               "Carbon","Kevlar","HSHT Fiberglass","Carbon","Kevlar","HSHT Fiberglass",
                               "Fiberglass","Fiberglass","Fiberglass","Fiberglass","Fiberglass","Fiberglass"],
        "Printing_time_min":  [17,17,17,17,19,20,17,20,21,19,17,17,17,20,21,18,21,22,18,19,20,21,20,21],
        "Weight_g":           [1.22,1.22,1.22,1.46,1.37,1.38,1.48,1.40,1.43,1.52,1.48,1.22,1.52,1.52,1.53,1.54,1.52,1.56,1.54,1.38,1.43,1.52,1.53,1.56],
        "Cost_USD":           [0.25,0.25,0.25,0.46,0.36,0.36,0.62,0.45,0.45,0.31,0.62,0.25,0.47,0.39,0.39,0.64,0.48,0.48,0.64,0.34,0.41,0.48,0.37,0.44],
        "Elongation_pct":     [6.0,3.9,3.8,0.75,0.94,1.5,0.75,0.99,1.6,8.0,6.4,8.0,0.72,1.0,1.6,0.73,0.99,1.5,7.3,1.5,1.4,12.0,1.4,1.6],
        "Youngs_modulus_GPa": [0.871,1.264,1.604,2.730,1.495,1.565,4.082,2.267,2.049,1.251,1.674,1.329,2.730,1.828,1.984,4.238,2.319,2.080,0.515,1.347,2.075,0.858,1.365,1.960],
        "Tensile_strength_MPa":[29.8,25.9,36.6,68.8,9.59,62.7,111,80.2,98.4,42.6,45.1,42.9,14.0,40.8,76.7,114,16.6,93.8,7.6,20.3,87.6,37.5,32.3,95.5],
        "Stress_at_Break_MPa": [35.3,40.2,41.2,68.8,48.6,62.7,111,80.2,98.4,47.6,54.1,49.3,69.9,63.3,76.7,114,83.2,93.8,38.3,54.9,87.6,43.4,55.0,95.5],
    })

if uploaded is None:
    st.info("💡 Dataset démo chargé — **impression 3D composite** (fibres Carbon / Kevlar / Fiberglass)")
    df_raw = load_default()
else:
    df_raw = load_file(uploaded)
    df_raw.columns = [c.strip().replace(" ","_").replace("[","").replace("]","")
                       .replace("%","pct").replace("/","_") for c in df_raw.columns]

# ─────────────────────────────────────────────────────
#  COLUMN SELECTION
# ─────────────────────────────────────────────────────
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
cat_cols     = df_raw.select_dtypes(include="object").columns.tolist()

def_num = [c for c in ["filling_rate","Nb_Fiber_Layers"] if c in numeric_cols] or numeric_cols[:2]
def_cat = [c for c in ["Fiber_Type"] if c in cat_cols] or (cat_cols[:1] if cat_cols else [])
def_resp = [c for c in ["Youngs_modulus_GPa","Tensile_strength_MPa","Stress_at_Break_MPa","Cost_USD"]
            if c in numeric_cols] or numeric_cols[-3:]

st.title("🧪 DOE — Impression 3D Composite")
st.markdown("Analyse de l'influence du **taux de remplissage**, du **nombre de couches de fibres** et du **type de fibre** sur les propriétés mécaniques.")

c1, c2, c3 = st.columns(3)
with c1: factors_num = st.multiselect("📥 Facteurs numériques", numeric_cols, default=def_num)
with c2: factors_cat = st.multiselect("🏷️ Facteurs catégoriels", cat_cols, default=def_cat)
with c3: responses   = st.multiselect("📤 Réponses", numeric_cols, default=def_resp)

if not (factors_num or factors_cat) or not responses:
    st.warning("⚠️ Sélectionnez au moins un facteur et une réponse.")
    st.stop()

# ─────────────────────────────────────────────────────
#  ENCODE + BUILD MODELS
# ─────────────────────────────────────────────────────
df_enc = df_raw[factors_num + factors_cat + responses].copy()
if factors_cat:
    df_enc = pd.get_dummies(df_enc, columns=factors_cat, drop_first=False, dtype=float)
encoded_factors = [c for c in df_enc.columns if c not in responses]
X_full = df_enc[encoded_factors].values

models = {}
for resp in responses:
    y = df_enc[resp].values
    pipe = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression())
    pipe.fit(X_full, y)
    y_pred = pipe.predict(X_full)
    pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
    pf.fit(X_full)
    models[resp] = {
        "pipeline": pipe,
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "y_pred": y_pred,
        "feat_names": pf.get_feature_names_out(encoded_factors),
        "coefs": pipe.named_steps["linearregression"].coef_,
        "intercept": pipe.named_steps["linearregression"].intercept_,
    }

# ─────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────
tabs = st.tabs(["📊 Données","📐 Modélisation","📈 Visualisation","🎯 Influence","📋 ANOVA","🏆 Optimisation"])

# ═══ TAB 1 — DONNÉES ═══════════════════════════════
with tabs[0]:
    st.subheader("Aperçu du dataset")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Expériences", len(df_raw))
    m2.metric("Facteurs num.", len(factors_num))
    m3.metric("Facteurs cat.", len(factors_cat))
    m4.metric("Réponses", len(responses))
    m5.metric("Valeurs manquantes", int(df_raw[factors_num+factors_cat+responses].isnull().sum().sum()))

    st.dataframe(df_raw[factors_num+factors_cat+responses]
                 .style.background_gradient(subset=responses, cmap="YlGn"),
                 use_container_width=True)

    if factors_cat:
        st.markdown(f"### Moyenne des réponses par **{factors_cat[0]}**")
        st.dataframe(df_raw.groupby(factors_cat[0])[responses].mean().round(3)
                     .style.background_gradient(cmap="Blues"), use_container_width=True)

    if len(factors_num + responses) >= 2:
        st.markdown("### Matrice de corrélation")
        corr = df_raw[factors_num + responses].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(corr)*0.9), max(5, len(corr)*0.8)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ═══ TAB 2 — MODÉLISATION ═══════════════════════════
with tabs[1]:
    st.subheader("Modélisation par régression polynomiale")
    if factors_cat:
        st.info(f"✅ Encodage One-Hot sur **{', '.join(factors_cat)}** → {len(encoded_factors)} variables encodées")

    for resp in responses:
        m = models[resp]
        st.markdown(f"#### 🎯 {resp}")
        c1,c2,c3 = st.columns(3)
        c1.metric("R²", f"{m['r2']:.4f}")
        c2.metric("RMSE", f"{m['rmse']:.4f}")
        c3.metric("Degré", poly_degree)

        coef_df = pd.DataFrame({"Variable": m["feat_names"], "Coefficient": m["coefs"]})
        coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index).head(10)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top 10 termes :**")
            st.dataframe(coef_df.round(4), use_container_width=True)
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors_b = ["#2563EB" if c >= 0 else "#EF4444" for c in coef_df["Coefficient"]]
            ax.barh(coef_df["Variable"], coef_df["Coefficient"], color=colors_b, edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"Coefficients — {resp}")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        y_obs = df_enc[resp].values
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.scatter(y_obs, m["y_pred"], color="#10B981", edgecolors="white", s=60, zorder=3)
        lims = [min(y_obs.min(), m["y_pred"].min())-2, max(y_obs.max(), m["y_pred"].max())+2]
        ax.plot(lims, lims, "r--", lw=1.5, label="Idéal")
        ax.set_xlabel("Observé"); ax.set_ylabel("Prédit")
        ax.set_title(f"Observé vs Prédit — {resp} (R²={m['r2']:.3f})")
        ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("---")

# ═══ TAB 3 — VISUALISATION ══════════════════════════
with tabs[2]:
    st.subheader("Visualisation des effets")
    resp_viz = st.selectbox("Réponse", responses, key="viz_r")
    y_viz = df_raw[resp_viz].values

    # Facteurs numériques
    if factors_num:
        st.markdown("#### Effets des facteurs numériques")
        nc = min(len(factors_num), 3)
        nr = (len(factors_num)+nc-1)//nc
        fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
        axes = np.array(axes).flatten() if len(factors_num) > 1 else [axes]
        for i, fac in enumerate(factors_num):
            xv = df_raw[fac].values
            axes[i].scatter(xv, y_viz, color="#10B981", edgecolors="white", s=60, zorder=3)
            mb, bb = np.polyfit(xv, y_viz, 1)
            xl = np.linspace(xv.min(), xv.max(), 100)
            axes[i].plot(xl, mb*xl+bb, color="#EF4444", lw=2)
            axes[i].set_xlabel(fac); axes[i].set_ylabel(resp_viz)
            axes[i].set_title(f"{resp_viz} vs {fac}"); axes[i].grid(True, alpha=0.3)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Boxplot par catégorie
    if factors_cat:
        st.markdown(f"#### Distribution par **{factors_cat[0]}** (Boxplot)")
        cats = sorted(df_raw[factors_cat[0]].unique())
        data_bp = [df_raw[df_raw[factors_cat[0]] == c][resp_viz].values for c in cats]
        fig, ax = plt.subplots(figsize=(9, 4))
        bp = ax.boxplot(data_bp, labels=cats, patch_artist=True)
        colors_bp = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6"]
        for patch, col in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_ylabel(resp_viz); ax.set_title(f"{resp_viz} par {factors_cat[0]}")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(); st.pyplot(fig); plt.close()

        # Interaction fibre × facteur num
        if factors_num:
            st.markdown("#### Graphique d'interaction")
            fac_inter = st.selectbox("Facteur numérique", factors_num, key="inter")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_l = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6"]
            for idx, cat in enumerate(cats):
                sub = df_raw[df_raw[factors_cat[0]] == cat].sort_values(fac_inter)
                ax.plot(sub[fac_inter], sub[resp_viz], marker="o",
                        color=colors_l[idx % 5], label=cat, lw=2)
            ax.set_xlabel(fac_inter); ax.set_ylabel(resp_viz)
            ax.set_title(f"Interaction : {fac_inter} × {factors_cat[0]} → {resp_viz}")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # Surface de réponse 3D
    if len(factors_num) >= 2:
        st.markdown("#### Surface de réponse 3D")
        fx = st.selectbox("Axe X", factors_num, key="sx")
        fy = st.selectbox("Axe Y", [f for f in factors_num if f != fx], key="sy")
        x_r = np.linspace(df_raw[fx].min(), df_raw[fx].max(), 25)
        y_r = np.linspace(df_raw[fy].min(), df_raw[fy].max(), 25)
        XX, YY = np.meshgrid(x_r, y_r)
        mean_enc = df_enc[encoded_factors].mean()
        grid_df = pd.DataFrame(np.tile(mean_enc.values, (XX.size,1)), columns=encoded_factors)
        if fx in encoded_factors: grid_df[fx] = XX.ravel()
        if fy in encoded_factors: grid_df[fy] = YY.ravel()
        ZZ = models[resp_viz]["pipeline"].predict(grid_df[encoded_factors].values).reshape(XX.shape)
        fig = plt.figure(figsize=(8, 5))
        ax3d = fig.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.8, edgecolor="none")
        ax3d.scatter(df_raw[fx], df_raw[fy], df_raw[resp_viz], color="red", s=40, zorder=5)
        ax3d.set_xlabel(fx); ax3d.set_ylabel(fy); ax3d.set_zlabel(resp_viz)
        ax3d.set_title(f"Surface de réponse : {resp_viz}")
        fig.colorbar(surf, shrink=0.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ═══ TAB 4 — INFLUENCE ══════════════════════════════
with tabs[3]:
    st.subheader("Analyse de l'influence des facteurs")
    resp_inf = st.selectbox("Réponse", responses, key="inf_r")

    X_std = StandardScaler().fit_transform(X_full)
    y_inf = df_enc[resp_inf].values
    y_std = StandardScaler().fit_transform(y_inf.reshape(-1,1)).ravel()
    pipe_s = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), LinearRegression())
    pipe_s.fit(X_std, y_std)
    std_coefs = pipe_s.named_steps["linearregression"].coef_

    imp = pd.DataFrame({"Variable": encoded_factors, "Coef. std": std_coefs})
    imp["Importance (%)"] = (np.abs(std_coefs) / (np.abs(std_coefs).sum()+1e-9) * 100).round(1)
    imp = imp.sort_values("Importance (%)", ascending=False).reset_index(drop=True)
    imp["Rang"] = range(1, len(imp)+1)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Tableau d'importance")
        st.dataframe(imp[["Rang","Variable","Coef. std","Importance (%)"]].style
                     .background_gradient(subset=["Importance (%)"], cmap="Blues"),
                     use_container_width=True)
    with cb:
        st.markdown("#### Diagramme de Pareto")
        top10 = imp.head(10)
        fig, ax = plt.subplots(figsize=(5, max(4, len(top10)*0.45)))
        bar_c = ["#2563EB" if c >= 0 else "#EF4444" for c in top10["Coef. std"]]
        bars = ax.barh(top10["Variable"], np.abs(top10["Coef. std"]), color=bar_c, edgecolor="white")
        ax.set_xlabel("| Coef. standardisé |"); ax.set_title(f"Importance → {resp_inf}")
        ax.invert_yaxis()
        for bar, val in zip(bars, top10["Importance (%)"]):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{val}%", va="center", fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    top = imp.iloc[0]
    sign = "positivement" if top["Coef. std"] > 0 else "négativement"
    st.success(f"🔑 **{top['Variable']}** est le facteur le plus influent ({top['Importance (%)']:.1f}%) "
               f"et agit **{sign}** sur **{resp_inf}**.")

    # Heatmap importance toutes réponses
    st.markdown("#### Importance comparative — toutes les réponses")
    imp_matrix = {}
    for resp in responses:
        y_r = df_enc[resp].values
        y_rs = StandardScaler().fit_transform(y_r.reshape(-1,1)).ravel()
        p2 = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), LinearRegression())
        p2.fit(X_std, y_rs)
        c2 = p2.named_steps["linearregression"].coef_
        imp_matrix[resp] = np.abs(c2) / (np.abs(c2).sum()+1e-9) * 100
    imp_df = pd.DataFrame(imp_matrix, index=encoded_factors).round(1)
    st.dataframe(imp_df.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

# ═══ TAB 5 — ANOVA ══════════════════════════════════
with tabs[4]:
    st.subheader("Analyse ANOVA")
    st.markdown(f"Seuil de signification : **α = {alpha_anova}**")
    resp_anova = st.selectbox("Réponse", responses, key="anova_r")

    def q(c): return f"Q('{c}')" if any(x in c for x in [" ","-","%","[","]"]) else c

    terms = [q(f) for f in factors_num]
    for fc in factors_cat: terms.append(f"C({q(fc)})")
    formula = f"{q(resp_anova)} ~ {' + '.join(terms)}"

    try:
        df_a = df_raw[factors_num + factors_cat + [resp_anova]].copy()
        model_ols = ols(formula, data=df_a).fit()
        at = anova_lm(model_ols, typ=2).reset_index()
        at.columns = ["Source","SS","df","F","p-value"]
        at["Significatif"] = at["p-value"].apply(
            lambda p: "✅ Oui" if pd.notna(p) and p < alpha_anova else ("➖ —" if pd.isna(p) else "❌ Non"))

        st.markdown("#### Tableau ANOVA (Type II)")
        st.dataframe(at.style
            .applymap(lambda v: "background-color:#d1fae5;font-weight:bold" if v=="✅ Oui" else "",
                      subset=["Significatif"])
            .format({"SS":"{:.4f}","df":"{:.0f}","F":"{:.3f}","p-value":"{:.4f}"}),
            use_container_width=True)

        anova_f = at.dropna(subset=["F"])
        anova_f = anova_f[anova_f["Source"] != "Residual"]
        if not anova_f.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bc = ["#10B981" if p < alpha_anova else "#94A3B8" for p in anova_f["p-value"]]
            ax.bar(anova_f["Source"], anova_f["F"], color=bc, edgecolor="white")
            ax.set_ylabel("Valeur F"); ax.set_title(f"Valeurs F → {resp_anova}")
            ax.set_xticklabels(anova_f["Source"], rotation=25, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        sig = at[at["Significatif"]=="✅ Oui"]["Source"].tolist()
        if sig: st.success(f"✅ Facteurs significatifs (p < {alpha_anova}) : **{', '.join(sig)}**")
        else:   st.warning(f"⚠️ Aucun facteur significatif au seuil α = {alpha_anova}.")

        with st.expander("📄 Résumé OLS complet"):
            st.text(model_ols.summary().as_text())
    except Exception as e:
        st.error(f"Erreur ANOVA : {e}")
        st.code(f"Formule : {formula}")

# ═══ TAB 6 — OPTIMISATION ═══════════════════════════
with tabs[5]:
    st.subheader("Optimisation multi-objectif — Désirabilité globale")

    if len(responses) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 réponses."); st.stop()

    st.markdown("""
    **Méthode : Fonction de Désirabilité (D)**  
    Chaque réponse normalisée entre 0 (inacceptable) → 1 (optimal).  
    D global = **moyenne géométrique** de toutes les désirabilités.
    """)

    st.markdown("#### Définir les objectifs")
    goals = {}
    cols_g = st.columns(len(responses))
    for i, resp in enumerate(responses):
        with cols_g[i]:
            st.markdown(f"**{resp}**")
            goal = st.selectbox("Objectif", ["Maximiser","Minimiser","Cible"], key=f"g_{resp}")
            tval = st.number_input("Cible", value=float(df_raw[resp].mean()), key=f"t_{resp}") if goal == "Cible" else None
            goals[resp] = {"goal": goal, "target": tval}

    # Build candidate grid
    n_g = 8
    grid_num = {f: np.linspace(df_raw[f].min(), df_raw[f].max(), n_g) for f in factors_num}
    cat_combos = list(iproduct(*[df_raw[fc].unique().tolist() for fc in factors_cat])) if factors_cat else [()]
    num_combos = list(iproduct(*[grid_num[f] for f in factors_num])) if factors_num else [()]

    rows = []
    for nc in num_combos:
        for cc in cat_combos:
            row = {}
            for i,f in enumerate(factors_num): row[f] = nc[i]
            for i,fc in enumerate(factors_cat): row[fc] = cc[i]
            rows.append(row)

    df_cand = pd.DataFrame(rows)
    df_cand_enc = pd.get_dummies(df_cand, columns=factors_cat, drop_first=False, dtype=float) if factors_cat else df_cand.copy()
    for col in encoded_factors:
        if col not in df_cand_enc.columns: df_cand_enc[col] = 0.0
    X_cand = df_cand_enc[encoded_factors].values

    preds = {resp: models[resp]["pipeline"].predict(X_cand) for resp in responses}

    D_global = np.ones(len(X_cand))
    for resp in responses:
        p = preds[resp]; ymin = p.min(); ymax = p.max()
        g = goals[resp]["goal"]
        if g == "Maximiser": d = (p-ymin)/(ymax-ymin+1e-9)
        elif g == "Minimiser": d = (ymax-p)/(ymax-ymin+1e-9)
        else:
            t = goals[resp]["target"]
            d = 1 - np.abs(p-t) / (max(abs(ymax-t), abs(ymin-t))+1e-9)
        D_global *= np.clip(d, 0, 1)
    D_global = D_global ** (1/len(responses))

    best_idx = int(np.argmax(D_global))
    best_row = df_cand.iloc[best_idx]
    best_preds = {resp: float(preds[resp][best_idx]) for resp in responses}
    best_D = float(D_global[best_idx])

    st.markdown("#### 🏆 Solution optimale")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Conditions optimales :**")
        opt = [{"Facteur": f, "Valeur": round(float(best_row[f]),3)} for f in factors_num]
        opt += [{"Facteur": fc, "Valeur": best_row[fc]} for fc in factors_cat]
        st.dataframe(pd.DataFrame(opt), use_container_width=True)
    with cc2:
        st.markdown("**Réponses prédites :**")
        st.dataframe(pd.DataFrame({
            "Réponse": list(best_preds.keys()),
            "Prédit": [round(v,3) for v in best_preds.values()],
            "Objectif": [goals[r]["goal"] for r in best_preds]
        }), use_container_width=True)

    st.metric("🎯 Désirabilité globale D", f"{best_D:.4f}")

    # Scatter Pareto + désirabilité vs facteur
    r1, r2 = responses[0], responses[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sc = axes[0].scatter(preds[r1], preds[r2], c=D_global, cmap="RdYlGn", s=15, alpha=0.6)
    axes[0].scatter(best_preds[r1], best_preds[r2], color="blue", s=200, marker="*",
                    zorder=5, label=f"Optimum (D={best_D:.3f})")
    plt.colorbar(sc, ax=axes[0], label="D global")
    axes[0].set_xlabel(r1); axes[0].set_ylabel(r2)
    axes[0].set_title("Espace des solutions (Pareto)"); axes[0].legend()

    if factors_num:
        fn = factors_num[0]
        xp = df_cand[fn].values
        axes[1].scatter(xp, D_global, c=D_global, cmap="RdYlGn", s=15, alpha=0.6)
        axes[1].axvline(float(best_row[fn]), color="blue", lw=2, linestyle="--",
                        label=f"Optimal = {float(best_row[fn]):.1f}")
        axes[1].set_xlabel(fn); axes[1].set_ylabel("D global")
        axes[1].set_title(f"Désirabilité vs {fn}"); axes[1].legend()

    fig.tight_layout(); st.pyplot(fig); plt.close()

    conditions_str = ", ".join([f"**{f}** = {float(best_row[f]):.1f}" for f in factors_num])
    if factors_cat:
        conditions_str += " | " + ", ".join([f"**{fc}** = {best_row[fc]}" for fc in factors_cat])
    st.success(
        f"✅ **Conditions optimales** : {conditions_str}\n\n"
        + "**Réponses prédites** : "
        + ", ".join([f"**{r}** ≈ {v:.2f}" for r,v in best_preds.items()])
        + f"\n\n**Désirabilité globale D = {best_D:.4f}**"
    )
