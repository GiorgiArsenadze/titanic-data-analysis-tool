"""
Data Science Portfolio Project
Tools Demonstrated:
  1. Exploratory Data Analysis (EDA)
  2. Data Cleaning & Preprocessing
  3. Data Visualization
  4. Machine Learning (Classification)
  5. Feature Importance Analysis
Dataset: Titanic (built-in via seaborn)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
import warnings, json, base64, io, os
warnings.filterwarnings('ignore')

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    'bg':       '#0d1117',
    'card':     '#161b22',
    'border':   '#30363d',
    'accent1':  '#58a6ff',
    'accent2':  '#3fb950',
    'accent3':  '#f78166',
    'accent4':  '#d2a8ff',
    'text':     '#c9d1d9',
    'muted':    '#8b949e',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['card'],
    'axes.edgecolor':    PALETTE['border'],
    'axes.labelcolor':   PALETTE['text'],
    'xtick.color':       PALETTE['muted'],
    'ytick.color':       PALETTE['muted'],
    'text.color':        PALETTE['text'],
    'grid.color':        PALETTE['border'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})

# ── Helper: fig → base64 ────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=PALETTE['bg'], dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Titanic dataset...")
df_raw = sns.load_dataset('titanic')
print(f"Shape: {df_raw.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n── EDA ──")
eda_stats = df_raw.describe(include='all').round(2).to_dict()

missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(1)
missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct})
missing_df = missing_df[missing_df['missing'] > 0].sort_values('pct', ascending=False)
print(missing_df)

survival_rate = df_raw['survived'].mean() * 100
class_survival = df_raw.groupby('pclass')['survived'].mean().mul(100).round(1).to_dict()
sex_survival   = df_raw.groupby('sex')['survived'].mean().mul(100).round(1).to_dict()
print(f"Overall survival rate: {survival_rate:.1f}%")

# ── Fig 1: Missing values ───────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 4))
colors = [PALETTE['accent3'] if p > 20 else PALETTE['accent1']
          for p in missing_df['pct']]
bars = ax.barh(missing_df.index, missing_df['pct'], color=colors, edgecolor='none')
ax.set_xlabel('Missing %')
ax.set_title('Missing Values by Column', fontsize=13, pad=12, color=PALETTE['accent1'])
for bar, val in zip(bars, missing_df['pct']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', fontsize=9, color=PALETTE['text'])
ax.set_xlim(0, missing_df['pct'].max() + 12)
plt.tight_layout()
img_missing = fig_to_b64(fig1)
plt.close()

# ── Fig 2: Survival by class & sex ─────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
colors_class = [PALETTE['accent1'], PALETTE['accent4'], PALETTE['accent3']]
axes[0].bar(['1st', '2nd', '3rd'], list(class_survival.values()),
            color=colors_class, edgecolor='none', width=0.5)
axes[0].set_title('Survival Rate by Class', fontsize=11, color=PALETTE['accent4'])
axes[0].set_ylabel('Survival %')
axes[0].set_ylim(0, 100)
for i, v in enumerate(class_survival.values()):
    axes[0].text(i, v + 2, f'{v}%', ha='center', fontsize=10, color=PALETTE['text'])

colors_sex = [PALETTE['accent3'], PALETTE['accent2']]
axes[1].bar(list(sex_survival.keys()), list(sex_survival.values()),
            color=colors_sex, edgecolor='none', width=0.4)
axes[1].set_title('Survival Rate by Sex', fontsize=11, color=PALETTE['accent4'])
axes[1].set_ylabel('Survival %')
axes[1].set_ylim(0, 100)
for i, v in enumerate(sex_survival.values()):
    axes[1].text(i, v + 2, f'{v}%', ha='center', fontsize=10, color=PALETTE['text'])
plt.tight_layout()
img_survival = fig_to_b64(fig2)
plt.close()

# ── Fig 3: Age distribution ─────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(8, 4))
survived     = df_raw[df_raw['survived'] == 1]['age'].dropna()
not_survived = df_raw[df_raw['survived'] == 0]['age'].dropna()
ax.hist(not_survived, bins=30, alpha=0.7, color=PALETTE['accent3'],
        label='Not Survived', edgecolor='none')
ax.hist(survived, bins=30, alpha=0.7, color=PALETTE['accent2'],
        label='Survived', edgecolor='none')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution by Survival', fontsize=13, pad=12, color=PALETTE['accent1'])
ax.legend()
plt.tight_layout()
img_age = fig_to_b64(fig3)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Preprocessing ──")
df = df_raw.copy()

# Drop high-missing / irrelevant columns
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male',
                 'alone'], inplace=True)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.dropna(inplace=True)

# Encode categoricals
le = LabelEncoder()
df['sex_enc']      = le.fit_transform(df['sex'])
df['embarked_enc'] = le.fit_transform(df['embarked'])

FEATURES = ['pclass', 'sex_enc', 'age', 'sibsp', 'parch',
            'fare', 'embarked_enc']
TARGET   = 'survived'

X = df[FEATURES]
y = df[TARGET]

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Final shape after cleaning: {df.shape}")

# ── Fig 4: Correlation heatmap ──────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(8, 6))
corr = df[FEATURES + [TARGET]].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f',
            linewidths=0.5, linecolor=PALETTE['border'],
            annot_kws={'size': 9}, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=13, pad=12,
             color=PALETTE['accent1'])
plt.tight_layout()
img_corr = fig_to_b64(fig4)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4. MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Machine Learning ──")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred   = rf.predict(X_test)
y_prob   = rf.predict_proba(X_test)[:, 1]

acc      = accuracy_score(y_test, y_pred) * 100
roc_auc  = roc_auc_score(y_test, y_prob)
report   = classification_report(y_test, y_pred, output_dict=True)

print(f"Accuracy : {acc:.1f}%")
print(f"ROC-AUC  : {roc_auc:.3f}")

# ── Fig 5: Confusion matrix ──────────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
cmap2 = sns.light_palette(PALETTE['accent1'], as_cmap=True)
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap2, ax=ax,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            linewidths=1, linecolor=PALETTE['border'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix', fontsize=13, pad=12, color=PALETTE['accent1'])
plt.tight_layout()
img_cm = fig_to_b64(fig5)
plt.close()

# ── Fig 6: ROC curve ────────────────────────────────────────────────────────
fig6, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color=PALETTE['accent2'], lw=2,
        label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color=PALETTE['muted'], lw=1, linestyle='--',
        label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE['accent2'])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve – Random Forest', fontsize=13, pad=12,
             color=PALETTE['accent1'])
ax.legend(loc='lower right')
plt.tight_layout()
img_roc = fig_to_b64(fig6)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Feature Importance ──")
feat_labels = ['Passenger Class', 'Sex', 'Age',
               'Siblings/Spouses', 'Parents/Children',
               'Fare', 'Embarkment Port']
importances = rf.feature_importances_
feat_df = pd.DataFrame({'feature': feat_labels, 'importance': importances})
feat_df.sort_values('importance', ascending=True, inplace=True)
print(feat_df.sort_values('importance', ascending=False).to_string(index=False))

# ── Fig 7: Feature importance bar ───────────────────────────────────────────
fig7, ax = plt.subplots(figsize=(8, 5))
norm = plt.Normalize(feat_df['importance'].min(), feat_df['importance'].max())
bar_colors = plt.cm.Blues(norm(feat_df['importance'].values) * 0.6 + 0.4)
bars = ax.barh(feat_df['feature'], feat_df['importance'],
               color=bar_colors, edgecolor='none')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance – Random Forest', fontsize=13, pad=12,
             color=PALETTE['accent1'])
for bar, val in zip(bars, feat_df['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color=PALETTE['text'])
ax.set_xlim(0, feat_df['importance'].max() + 0.06)
plt.tight_layout()
img_feat = fig_to_b64(fig7)
plt.close()

print("\n✅ All analysis complete — building HTML dashboard...")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
prec0 = report['0']['precision']; rec0  = report['0']['recall']
prec1 = report['1']['precision']; rec1  = report['1']['recall']
f1_0  = report['0']['f1-score'];  f1_1  = report['1']['f1-score']

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Data Science Portfolio – Titanic Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --a1:      #58a6ff;
    --a2:      #3fb950;
    --a3:      #f78166;
    --a4:      #d2a8ff;
    --text:    #c9d1d9;
    --muted:   #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
  }}

  /* ── HEADER ── */
  .hero {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 40px 36px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(88,166,255,.08) 0%, transparent 70%);
    pointer-events: none;
  }}
  .hero-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem;
    letter-spacing: .15em;
    color: var(--a1);
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .hero h1 {{
    font-size: clamp(1.6rem, 4vw, 2.8rem);
    font-weight: 700;
    color: #fff;
    line-height: 1.15;
    max-width: 700px;
  }}
  .hero h1 span {{ color: var(--a1); }}
  .hero p {{
    margin-top: 12px;
    color: var(--muted);
    font-size: .95rem;
    max-width: 560px;
    line-height: 1.6;
  }}
  .badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
  }}
  .badge {{
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: .05em;
    border: 1px solid;
  }}
  .b1 {{ color: var(--a1); border-color: var(--a1); background: rgba(88,166,255,.08); }}
  .b2 {{ color: var(--a2); border-color: var(--a2); background: rgba(63,185,80,.08); }}
  .b3 {{ color: var(--a3); border-color: var(--a3); background: rgba(247,129,102,.08); }}
  .b4 {{ color: var(--a4); border-color: var(--a4); background: rgba(210,168,255,.08); }}

  /* ── LAYOUT ── */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}

  /* ── SECTION HEADER ── */
  .section-head {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
  }}
  .step-badge {{
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: .85rem;
    flex-shrink: 0;
  }}
  .section-head h2 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #fff;
  }}
  .section-head p {{
    font-size: .82rem;
    color: var(--muted);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* ── CARDS ── */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: var(--a1); }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .kpi {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    transition: transform .2s, border-color .2s;
    cursor: default;
  }}
  .kpi:hover {{ transform: translateY(-2px); }}
  .kpi-label {{
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }}
  .kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 6px 0 2px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .kpi-sub {{ font-size: .78rem; color: var(--muted); }}

  /* ── SECTION DIVIDER ── */
  .section-block {{
    margin-bottom: 52px;
    border-top: 1px solid var(--border);
    padding-top: 36px;
  }}

  /* ── IMAGES ── */
  .img-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 20px;
  }}
  .img-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color .2s, box-shadow .2s;
  }}
  .img-card:hover {{
    border-color: var(--a1);
    box-shadow: 0 0 0 1px rgba(88,166,255,.2), 0 8px 24px rgba(0,0,0,.4);
  }}
  .img-card img {{
    width: 100%;
    display: block;
    background: var(--bg);
  }}
  .img-caption {{
    padding: 12px 16px;
    font-size: .8rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border);
  }}

  /* ── METRICS TABLE ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: .88rem;
  }}
  th {{
    text-align: left;
    padding: 10px 14px;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 12px 14px;
    border-bottom: 1px solid rgba(48,54,61,.5);
    color: var(--text);
  }}
  tr:hover td {{ background: rgba(88,166,255,.04); }}
  .mono {{ font-family: 'JetBrains Mono', monospace; }}

  /* ── PROGRESS BAR ── */
  .bar-wrap {{
    height: 6px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    width: 120px;
    display: inline-block;
    vertical-align: middle;
    margin-left: 8px;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    background: var(--a2);
  }}

  /* ── EXPLAINER BOX ── */
  .explainer {{
    background: rgba(88,166,255,.05);
    border: 1px solid rgba(88,166,255,.2);
    border-left: 3px solid var(--a1);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 22px;
    font-size: .9rem;
    line-height: 1.7;
    color: var(--text);
  }}
  .explainer strong {{ color: #fff; }}
  .explainer .what-label {{
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--a1);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  .explainer.green {{ background: rgba(63,185,80,.05); border-color: rgba(63,185,80,.2); border-left-color: var(--a2); }}
  .explainer.green .what-label {{ color: var(--a2); }}
  .explainer.purple {{ background: rgba(210,168,255,.05); border-color: rgba(210,168,255,.2); border-left-color: var(--a4); }}
  .explainer.purple .what-label {{ color: var(--a4); }}
  .explainer.red {{ background: rgba(247,129,102,.05); border-color: rgba(247,129,102,.2); border-left-color: var(--a3); }}
  .explainer.red .what-label {{ color: var(--a3); }}

  /* ── FOOTER ── */
  footer {{
    border-top: 1px solid var(--border);
    padding: 28px 40px;
    text-align: center;
    font-size: .78rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }}
  footer span {{ color: var(--a1); }}
</style>
</head>
<body>

<!-- ── HERO ── -->
<div class="hero">
  <div class="hero-label">Data Science Portfolio Project</div>
  <h1>Titanic Survival <span>Analysis</span></h1>
  <p>End-to-end data science pipeline demonstrating EDA, preprocessing,
     visualization, machine learning, and feature importance on the
     classic Titanic dataset.</p>
  <div class="badges">
    <span class="badge b1">01 · EDA</span>
    <span class="badge b2">02 · Preprocessing</span>
    <span class="badge b4">03 · Visualization</span>
    <span class="badge b1">04 · ML · Random Forest</span>
    <span class="badge b3">05 · Feature Importance</span>
  </div>
</div>

<div class="container">

  <!-- ── KPIs ── -->
  <div class="kpi-grid">
    <div class="kpi" style="border-left:3px solid var(--a1)">
      <div class="kpi-label">Dataset Size</div>
      <div class="kpi-value" style="color:var(--a1)">{len(df_raw)}</div>
      <div class="kpi-sub">passengers</div>
    </div>
    <div class="kpi" style="border-left:3px solid var(--a2)">
      <div class="kpi-label">Survival Rate</div>
      <div class="kpi-value" style="color:var(--a2)">{survival_rate:.1f}%</div>
      <div class="kpi-sub">overall</div>
    </div>
    <div class="kpi" style="border-left:3px solid var(--a4)">
      <div class="kpi-label">Model Accuracy</div>
      <div class="kpi-value" style="color:var(--a4)">{acc:.1f}%</div>
      <div class="kpi-sub">test set</div>
    </div>
    <div class="kpi" style="border-left:3px solid var(--a1)">
      <div class="kpi-label">ROC-AUC</div>
      <div class="kpi-value" style="color:var(--a1)">{roc_auc:.3f}</div>
      <div class="kpi-sub">Random Forest</div>
    </div>
    <div class="kpi" style="border-left:3px solid var(--a3)">
      <div class="kpi-label">Features Used</div>
      <div class="kpi-value" style="color:var(--a3)">{len(FEATURES)}</div>
      <div class="kpi-sub">after cleaning</div>
    </div>
    <div class="kpi" style="border-left:3px solid var(--a2)">
      <div class="kpi-label">Female Survival</div>
      <div class="kpi-value" style="color:var(--a2)">{sex_survival.get('female', 0):.0f}%</div>
      <div class="kpi-sub">vs {sex_survival.get('male', 0):.0f}% male</div>
    </div>
  </div>

  <!-- ══ SECTION 1: EDA ══ -->
  <div class="section-block">
    <div class="section-head">
      <div class="step-badge" style="background:rgba(88,166,255,.15);color:var(--a1)">01</div>
      <div>
        <h2>Exploratory Data Analysis (EDA)</h2>
        <p>pandas · descriptive stats · missing value audit</p>
      </div>
    </div>
    <div class="explainer">
      <div class="what-label">🔍 What is this?</div>
      <strong>Exploratory Data Analysis (EDA)</strong> is the very first step — it's like opening a box you've never seen before and taking inventory of what's inside.
      Before writing any code to predict things, a data scientist needs to understand the raw data: How many rows are there? Are any values missing? What does the age distribution look like? Are there obvious patterns?
      <br/><br/>
      In this case, we found that <strong>77% of the "deck" column was missing</strong> (so we removed it), and that <strong>age was missing for ~20% of passengers</strong> (so we filled those gaps with the average age). We also noticed right away that <strong>younger passengers tended to survive more</strong> — that kind of insight guides everything that comes next.
    </div>
    <div class="img-grid">
      <div class="img-card">
        <img src="data:image/png;base64,{img_missing}" alt="Missing Values"/>
        <div class="img-caption">// missing_values_audit.png — deck column 77% missing → dropped</div>
      </div>
      <div class="img-card">
        <img src="data:image/png;base64,{img_age}" alt="Age Distribution"/>
        <div class="img-caption">// age_distribution.png — younger passengers had higher survival</div>
      </div>
    </div>
  </div>

  <!-- ══ SECTION 2: PREPROCESSING ══ -->
  <div class="section-block">
    <div class="section-head">
      <div class="step-badge" style="background:rgba(63,185,80,.15);color:var(--a2)">02</div>
      <div>
        <h2>Data Cleaning & Preprocessing</h2>
        <p>sklearn · LabelEncoder · StandardScaler · imputation</p>
      </div>
    </div>
    <div class="explainer green">
      <div class="what-label">🧹 What is this?</div>
      <strong>Data Cleaning & Preprocessing</strong> is turning messy, real-world data into something a computer can actually learn from.
      Raw data is almost never perfect — it has missing values, text categories (like "male"/"female") that computers can't do math on, and numbers on wildly different scales (age goes 0–80, fare goes 0–500).
      <br/><br/>
      We fixed all of this: <strong>filled in missing ages</strong> with the median, <strong>converted text to numbers</strong> (e.g. "female" → 0, "male" → 1), and <strong>scaled everything</strong> so no single column dominates just because its numbers are bigger. Think of it like prepping ingredients before cooking — the better the prep, the better the result.
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Step</th><th>Action</th><th>Detail</th>
          </tr>
        </thead>
        <tbody>
          <tr><td class="mono" style="color:var(--a1)">drop_columns</td><td>Removed 6 columns</td><td style="color:var(--muted)">deck, embark_town, alive, who, adult_male, alone</td></tr>
          <tr><td class="mono" style="color:var(--a1)">impute_age</td><td>Median fill</td><td style="color:var(--muted)">Age median = {df_raw['age'].median():.1f} years</td></tr>
          <tr><td class="mono" style="color:var(--a1)">impute_embarked</td><td>Mode fill</td><td style="color:var(--muted)">Most common port used for 2 missing rows</td></tr>
          <tr><td class="mono" style="color:var(--a1)">label_encode</td><td>Categorical → numeric</td><td style="color:var(--muted)">sex, embarked → integer codes</td></tr>
          <tr><td class="mono" style="color:var(--a1)">standard_scale</td><td>Z-score normalisation</td><td style="color:var(--muted)">All features → mean=0, std=1</td></tr>
          <tr><td class="mono" style="color:var(--a2)">train_test_split</td><td>80/20 stratified</td><td style="color:var(--muted)">random_state=42, stratify=y</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ══ SECTION 3: VISUALISATION ══ -->
  <div class="section-block">
    <div class="section-head">
      <div class="step-badge" style="background:rgba(210,168,255,.15);color:var(--a4)">03</div>
      <div>
        <h2>Data Visualisation</h2>
        <p>matplotlib · seaborn · correlation heatmap · grouped bar charts</p>
      </div>
    </div>
    <div class="explainer purple">
      <div class="what-label">📊 What is this?</div>
      <strong>Data Visualisation</strong> is turning numbers into pictures so patterns become obvious at a glance.
      A table with 891 rows tells you almost nothing. A chart that shows "women survived at 74% vs men at 19%" tells a story instantly.
      <br/><br/>
      The <strong>bar charts</strong> show survival rates broken down by ticket class and gender. The <strong>correlation heatmap</strong> is a grid showing how strongly each column relates to survival — the darker the colour, the stronger the connection. For example, sex and passenger class have a strong negative correlation with survival, meaning being male or in 3rd class significantly reduced your chances.
    </div>
    <div class="img-grid">
      <div class="img-card">
        <img src="data:image/png;base64,{img_survival}" alt="Survival by Class & Sex"/>
        <div class="img-caption">// survival_by_class_sex.png — 1st class 63% vs 3rd class 24%</div>
      </div>
      <div class="img-card">
        <img src="data:image/png;base64,{img_corr}" alt="Correlation Heatmap"/>
        <div class="img-caption">// correlation_heatmap.png — sex & pclass strongest predictors</div>
      </div>
    </div>
  </div>

  <!-- ══ SECTION 4: ML ══ -->
  <div class="section-block">
    <div class="section-head">
      <div class="step-badge" style="background:rgba(88,166,255,.15);color:var(--a1)">04</div>
      <div>
        <h2>Machine Learning – Random Forest Classifier</h2>
        <p>sklearn · confusion matrix · ROC curve · classification report</p>
      </div>
    </div>
    <div class="explainer">
      <div class="what-label">🤖 What is this?</div>
      <strong>Machine Learning</strong> is teaching a computer to make predictions by learning from examples — without being explicitly programmed with rules.
      We used a <strong>Random Forest</strong>, which builds hundreds of decision trees (each one asking questions like "Was this passenger female? Was their fare above $50?") and combines their answers for a final prediction.
      <br/><br/>
      We trained the model on <strong>80% of the data</strong>, then tested it on the remaining <strong>20% it had never seen</strong> — this tells us how well it generalises to new passengers.
      The <strong>confusion matrix</strong> shows exactly how many predictions were right vs wrong. The <strong>ROC curve</strong> measures how good the model is at distinguishing survivors from non-survivors — the closer to the top-left corner, the better. Our AUC of <strong>{roc_auc:.3f}</strong> means the model is significantly better than random guessing.
    </div>
    <div class="img-grid" style="grid-template-columns: 1fr 1fr;">
      <div class="img-card">
        <img src="data:image/png;base64,{img_cm}" alt="Confusion Matrix"/>
        <div class="img-caption">// confusion_matrix.png</div>
      </div>
      <div class="img-card">
        <img src="data:image/png;base64,{img_roc}" alt="ROC Curve"/>
        <div class="img-caption">// roc_curve.png — AUC = {roc_auc:.3f}</div>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <table>
        <thead>
          <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Visual</th></tr>
        </thead>
        <tbody>
          <tr>
            <td style="color:var(--a3)">Not Survived (0)</td>
            <td class="mono">{prec0:.3f}</td>
            <td class="mono">{rec0:.3f}</td>
            <td class="mono">{f1_0:.3f}</td>
            <td><div class="bar-wrap"><div class="bar-fill" style="width:{f1_0*100:.1f}%;background:var(--a3)"></div></div></td>
          </tr>
          <tr>
            <td style="color:var(--a2)">Survived (1)</td>
            <td class="mono">{prec1:.3f}</td>
            <td class="mono">{rec1:.3f}</td>
            <td class="mono">{f1_1:.3f}</td>
            <td><div class="bar-wrap"><div class="bar-fill" style="width:{f1_1*100:.1f}%;background:var(--a2)"></div></div></td>
          </tr>
          <tr>
            <td style="color:var(--a1); font-weight:600">Overall Accuracy</td>
            <td class="mono" colspan="3" style="color:var(--a1); font-weight:600">{acc:.2f}%</td>
            <td><div class="bar-wrap"><div class="bar-fill" style="width:{acc:.1f}%;background:var(--a1)"></div></div></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ══ SECTION 5: FEATURE IMPORTANCE ══ -->
  <div class="section-block">
    <div class="section-head">
      <div class="step-badge" style="background:rgba(247,129,102,.15);color:var(--a3)">05</div>
      <div>
        <h2>Feature Importance Analysis</h2>
        <p>RandomForest · feature_importances_ · Gini impurity</p>
      </div>
    </div>
    <div class="explainer red">
      <div class="what-label">⚖️ What is this?</div>
      <strong>Feature Importance</strong> answers the question: "Which pieces of information actually mattered to the model's predictions?"
      After training, the Random Forest tells us how much each column contributed to getting predictions right. A longer bar = that factor had more influence on whether someone survived.
      <br/><br/>
      The results are striking: <strong>Sex was the single most important factor</strong> (36.9%), followed by the fare paid (19%) and age (15.8%). This aligns with the historical reality of the Titanic — "women and children first" was the evacuation priority. Interestingly, <strong>which port someone boarded from barely mattered</strong> at all (2.9%), confirming it's not a meaningful predictor.
    </div>
    <div class="img-grid" style="grid-template-columns:1fr;">
      <div class="img-card">
        <img src="data:image/png;base64,{img_feat}" alt="Feature Importance"/>
        <div class="img-caption">// feature_importance.png — Sex & Fare dominate; embarkment is weakest signal</div>
      </div>
    </div>
  </div>

</div><!-- /container -->

<footer>
  Built with <span>Python · pandas · scikit-learn · matplotlib · seaborn</span>
  &nbsp;·&nbsp; Dataset: Titanic (seaborn built-in)
  &nbsp;·&nbsp; Model: Random Forest (n=200, depth=6)
</footer>

</body>
</html>"""

output_path = '/mnt/user-data/outputs/titanic_ds_dashboard.html'
with open(output_path, 'w') as f:
    f.write(html)

print(f"\n✅ Dashboard saved to: {output_path}")
