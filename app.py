import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', page_icon='📡',
                   layout='wide')
alt.data_transformers.disable_max_rows()

# =====================================================================
# Palette and styling
# =====================================================================
TEAL, ORANGE, GREEN = '#00838f', '#e4572e', '#2e9e5b'
AMBER, BLUE, PURPLE, NAVY = '#f2a93b', '#2f6fb0', '#6c4bb8', '#1f2d3d'

BAND_ORDER = ['LOW', 'WATCH', 'HIGH']
BAND_COLORS = {'LOW': GREEN, 'WATCH': AMBER, 'HIGH': ORANGE}
BAND_ACTION = {'HIGH': 'Contact now', 'WATCH': 'Keep an eye on',
               'LOW': 'No action'}
BAND_SCALE = alt.Scale(domain=BAND_ORDER,
                       range=[BAND_COLORS[b] for b in BAND_ORDER])
CONTRACTS = ['Month-to-month', 'One year', 'Two year']
CONTRACT_SCALE = alt.Scale(domain=CONTRACTS, range=[ORANGE, BLUE, GREEN])

st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg, {TEAL} 0%, {BLUE} 55%, {PURPLE} 100%);
        padding: 1.5rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,.18);}}
.hero h1 {{color: white; margin: 0; font-size: 2.1rem;}}
.hero p {{color: #e6f2f5; margin: .3rem 0 0 0; font-size: 1rem;}}
.pill {{display: inline-block; background: rgba(255,255,255,.20);
        border-radius: 999px; padding: .15rem .8rem; margin: .6rem .4rem 0 0;
        font-size: .85rem; font-weight: 600; color: white;}}
.kpi {{background: #ffffff; border-radius: 14px; padding: .9rem 1.1rem;
       box-shadow: 0 2px 10px rgba(0,0,0,.12); margin-bottom: .6rem;}}
.kpi-label {{color: #5b6b7b; font-size: .78rem; font-weight: 700;
             text-transform: uppercase; letter-spacing: .04em;}}
.kpi-value {{font-size: 2.1rem; font-weight: 800; line-height: 1.2;}}
.kpi-sub {{color: #5b6b7b; font-size: .8rem;}}
.note {{background: #fff6e5; border-left: 5px solid {AMBER}; color: #5a4300;
        padding: .6rem .9rem; border-radius: 8px; font-size: .9rem;}}
.section {{border-left: 6px solid {TEAL}; padding-left: .7rem; margin: 1rem 0 .3rem 0;}}
button[data-baseweb="tab"] {{font-weight: 700;}}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, color, sub=''):
    st.markdown(
        f"<div class='kpi' style='border-top:6px solid {color}'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value' style='color:{color}'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)


def section(title):
    st.markdown(f"<h3 class='section'>{title}</h3>", unsafe_allow_html=True)


# =====================================================================
# Constants and values taken from the Week 3 notebook
# =====================================================================
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)']
SAMPLE_FILE = 'sample_customers.csv'

COST_FN, COST_FP = 6000, 1000          # PKR, from the Week 2 cost analysis

# Week 3 results (copied from the notebook outputs)
MODEL_TABLE = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (tuned)'],
    'CV AUC': [0.8464, 0.8464, 0.8502],
    'Std': [0.0129, 0.0114, 0.0117]})
MODEL_TABLE['Low'] = MODEL_TABLE['CV AUC'] - MODEL_TABLE['Std']
MODEL_TABLE['High'] = MODEL_TABLE['CV AUC'] + MODEL_TABLE['Std']

SEGMENTS = pd.DataFrame({
    'Segment': ['Mid-tenure high spenders', 'New low-spend customers',
                'Loyal premium bundlers', 'Loyal basic customers'],
    'Customers': [2157, 1918, 1938, 1030],
    'Tenure (months)': [18.4, 9.0, 59.8, 53.6],
    'Monthly ($)': [80.4, 37.7, 92.1, 31.0],
    'Services': [3.3, 1.2, 5.1, 1.5],
    'Churn rate': [0.43, 0.32, 0.14, 0.05]})
OVERALL_CHURN = 0.265

# Test-set confusion matrix at threshold 0.5, rebuilt by hand from the
# notebook's recall 0.521 and precision 0.659 (about 374 churners of 1409)
CONFUSION = pd.DataFrame({
    'Actual': ['Churned', 'Churned', 'Stayed', 'Stayed'],
    'Predicted': ['Churn', 'Stay', 'Churn', 'Stay'],
    'Count': [195, 179, 101, 934],
    'Kind': ['TP', 'FN', 'FP', 'TN']})


# =====================================================================
# Load artifacts and the serving encoder (parity-tested in Kaggle)
# =====================================================================
@st.cache_resource            # load once per server, not on every click
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])            # no drop_first at serving
    return X.reindex(columns=columns, fill_value=0).astype(float)


model, meta = load_artifacts()
COLS = meta['feature_columns']


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


def band_of(p, t):
    return np.select([p >= t, p >= t / 2], ['HIGH', 'WATCH'], 'LOW')


def explain(cust):
    """Per-feature contribution to the log-odds (None if unsupported)."""
    try:
        X = prepare_input(cust, COLS)
        final = model[-1] if hasattr(model, 'steps') else model
        if hasattr(model, 'steps') and hasattr(final, 'coef_'):
            z = model[:-1].transform(X)[0]
            return (pd.Series(final.coef_[0] * z, index=COLS),
                    'Logistic regression: coefficient x standardized value')
        if hasattr(final, 'get_booster'):
            import xgboost as xgb
            Xin = model[:-1].transform(X) if hasattr(model, 'steps') else X
            c = final.get_booster().predict(
                xgb.DMatrix(Xin), pred_contribs=True)[0][:-1]
            return (pd.Series(c, index=COLS),
                    'XGBoost: contribution of each feature to the log-odds')
    except Exception:
        pass
    return None, ''


@st.cache_resource
def load_population():
    """Score the 50-customer sample once, for the distribution chart."""
    if not os.path.exists(SAMPLE_FILE):
        return None
    d = pd.read_csv(SAMPLE_FILE)
    d['TotalCharges'] = pd.to_numeric(d['TotalCharges'],
                                      errors='coerce').fillna(0)
    return pd.DataFrame({'p': score(d), 'Contract': d['Contract']})


# =====================================================================
# Sidebar: one customer
# =====================================================================
with st.sidebar:
    st.header('🧑 Customer profile')
    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract', CONTRACTS)
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0,
                              step=0.5)
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES,
                                  default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior citizen')
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    st.divider()
    st.subheader('🎯 Decision rule')
    threshold = st.slider('Contact threshold', 0.05, 0.95,
                          float(meta['threshold']), 0.05)
    st.caption(f"Notebook choice: {float(meta['threshold']):.2f} "
               f"(from the cost analysis, FN = PKR {COST_FN:,}, "
               f"FP = PKR {COST_FP:,}).")

row = {'gender': gender, 'SeniorCitizen': int(senior),
       'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure,
       'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly,
       'TotalCharges': tenure * monthly}           # derived, not typed
for s in SERVICES:
    row[s] = ('No internet service' if internet == 'No'
              else 'Yes' if s in services else 'No')
customer = pd.DataFrame([row])

# =====================================================================
# Hero banner
# =====================================================================
test_auc = meta.get('test_auc')
pills = [f"🤖 {meta['model_name']}",
         f"CV AUC {meta['cv_auc']:.3f} ± {meta['cv_auc_std']:.3f}"]
if test_auc is not None:
    pills.append(f"Test AUC {test_auc:.3f}")
pills += [f"Threshold {float(meta['threshold']):.2f}",
          f"{len(COLS)} features", f"v{meta.get('version', '1.0')}"]
st.markdown(
    "<div class='hero'><h1>📡 Customer Churn Risk Advisor</h1>"
    "<p>Score a customer, see what could lower the risk, and score a whole "
    "customer file. Decision support only.</p>"
    + ''.join(f"<span class='pill'>{t}</span>" for t in pills) + "</div>",
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ['👤 One customer', '📂 Batch scoring', '📊 Model insights', 'ℹ️ About'])

# =====================================================================
# Tab 1: one customer
# =====================================================================
with tab1:
    p = float(score(customer)[0])
    band = str(band_of(np.array([p]), threshold)[0])
    color = BAND_COLORS[band]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi('Churn probability', f'{p:.0%}', color,
            f'Average customer: {OVERALL_CHURN:.1%}')
    with c2:
        kpi('Risk band', band, color,
            f'LOW < {threshold / 2:.2f} <= WATCH < {threshold:.2f} <= HIGH')
    with c3:
        kpi('Action', BAND_ACTION[band], color,
            f'Missed churner costs PKR {COST_FN:,}; a call costs '
            f'PKR {COST_FP:,}')

    # --- probability on the band scale -------------------------------
    zones = pd.DataFrame({'band': BAND_ORDER,
                          'start': [0.0, threshold / 2, threshold],
                          'end': [threshold / 2, threshold, 1.0]})
    base = alt.Chart(zones).mark_bar(size=34).encode(
        x=alt.X('start:Q', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format='%', title='Churn probability')),
        x2='end:Q',
        color=alt.Color('band:N', scale=BAND_SCALE,
                        legend=alt.Legend(title='Band', orient='bottom')))
    pos = pd.DataFrame({'p': [p], 'label': [f'{p:.0%}']})
    marker = alt.Chart(pos).mark_rule(color=NAVY, strokeWidth=4).encode(
        x='p:Q')
    text = alt.Chart(pos).mark_text(dy=-28, fontWeight='bold', fontSize=15,
                                    color=NAVY).encode(x='p:Q', text='label:N')
    st.altair_chart((base + marker + text).properties(height=110,
                                                      width='container'))

    # --- what-if ------------------------------------------------------
    section('🔀 What would change the risk?')
    changes = [('Contract', c) for c in ['One year', 'Two year']
               if c != contract]
    if payment == 'Electronic check':
        changes.append(('PaymentMethod', 'Credit card (automatic)'))
    if internet != 'No' and 'TechSupport' not in services:
        changes.append(('TechSupport', 'Yes'))
    rows = []
    for feature, value in changes:
        alt_row = customer.copy()
        alt_row[feature] = value
        q = float(score(alt_row)[0])
        rows.append({'Change': f'{feature} = {value}',
                     'New probability': round(q, 3),
                     'Difference': round(q - p, 3)})
    if rows:
        wi = pd.DataFrame(rows)
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(
                wi, hide_index=True,
                column_config={
                    'New probability': st.column_config.ProgressColumn(
                        'New probability', min_value=0.0, max_value=1.0,
                        format='%.3f')})
        with right:
            wchart = alt.Chart(wi).mark_bar(cornerRadiusEnd=6).encode(
                y=alt.Y('Change:N', sort='x', title=None),
                x=alt.X('Difference:Q', title='Change in churn probability'),
                color=alt.condition(alt.datum.Difference < 0,
                                    alt.value(GREEN), alt.value(ORANGE)),
                tooltip=['Change', 'New probability', 'Difference'])
            st.altair_chart(wchart.properties(height=40 + 45 * len(wi),
                                              width='container'))
    st.markdown("<div class='note'>⚠️ <b>Association, not cause.</b> These "
                "are patterns the model learned from data. The model cannot "
                "tell whether <i>offering</i> a contract would make this "
                "customer stay. Only an A/B test can.</div>",
                unsafe_allow_html=True)

    # --- risk curve ---------------------------------------------------
    section('📈 Risk by tenure and contract (same customer otherwise)')
    tenures = np.arange(0, 73)
    frames = []
    for c in CONTRACTS:
        d = pd.concat([customer] * len(tenures), ignore_index=True)
        d['tenure'] = tenures
        d['Contract'] = c
        d['TotalCharges'] = tenures * float(monthly)
        frames.append(pd.DataFrame({'Tenure (months)': tenures,
                                    'Contract': c, 'p': score(d)}))
    curves = pd.concat(frames, ignore_index=True)
    line = alt.Chart(curves).mark_line(strokeWidth=3).encode(
        x=alt.X('Tenure (months):Q'),
        y=alt.Y('p:Q', axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1]), title='Churn probability'),
        color=alt.Color('Contract:N', scale=CONTRACT_SCALE),
        tooltip=['Contract', 'Tenure (months)',
                 alt.Tooltip('p:Q', format='.1%')])
    t_rule = alt.Chart(pd.DataFrame({'t': [threshold]})).mark_rule(
        color=NAVY, strokeDash=[6, 4]).encode(y='t:Q')
    cur = alt.Chart(pd.DataFrame({'x': [tenure], 'p': [p]})).mark_point(
        size=220, filled=True, color=NAVY).encode(x='x:Q', y='p:Q')
    st.altair_chart((line + t_rule + cur).properties(height=320,
                                                     width='container'))
    st.caption('Dashed line = contact threshold. Dark dot = the customer in '
               'the sidebar.')

    # --- explanation --------------------------------------------------
    contrib, how = explain(customer)
    if contrib is not None:
        section('🔎 Why this score?')
        order = contrib.abs().sort_values(ascending=False).index[:8]
        top = pd.DataFrame({
            'Feature': [c.replace('_', ' = ', 1) for c in order],
            'Contribution': contrib.reindex(order).values})
        echart = alt.Chart(top).mark_bar(cornerRadiusEnd=6).encode(
            y=alt.Y('Feature:N', sort=list(top['Feature']), title=None),
            x=alt.X('Contribution:Q', title='Push on the log-odds'),
            color=alt.condition(alt.datum.Contribution > 0,
                                alt.value(ORANGE), alt.value(GREEN)),
            tooltip=['Feature', alt.Tooltip('Contribution:Q', format='.2f')])
        st.altair_chart(echart.properties(height=300, width='container'))
        st.caption(f'{how}. Orange pushes toward churn, green pushes away.')

    # --- where does this customer sit? -------------------------------
    pop = load_population()
    if pop is not None:
        section('👥 Where this customer sits among 50 sample customers')
        hist = alt.Chart(pop).mark_bar(color=TEAL, opacity=.85).encode(
            x=alt.X('p:Q', bin=alt.Bin(maxbins=15, extent=[0, 1]),
                    axis=alt.Axis(format='%'), title='Churn probability'),
            y=alt.Y('count()', title='Customers'))
        you = alt.Chart(pd.DataFrame({'p': [p]})).mark_rule(
            color=color, strokeWidth=4).encode(x='p:Q')
        thr = alt.Chart(pd.DataFrame({'t': [threshold]})).mark_rule(
            color=NAVY, strokeDash=[6, 4]).encode(x='t:Q')
        st.altair_chart((hist + you + thr).properties(height=240,
                                                      width='container'))

# =====================================================================
# Tab 2: batch scoring
# =====================================================================
with tab2:
    st.write('Upload a CSV with the original Telco columns, or try the '
             'built-in sample.')
    have_sample = os.path.exists(SAMPLE_FILE)
    use_sample = st.checkbox('Use the built-in sample (50 customers)',
                             disabled=not have_sample)
    file = None if use_sample else st.file_uploader('Customer file',
                                                    type='csv')
    data = None
    if use_sample:
        data = pd.read_csv(SAMPLE_FILE)
    elif file is not None:
        data = pd.read_csv(file)

    if data is not None:
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            data = data.copy()
            data['TotalCharges'] = pd.to_numeric(
                data['TotalCharges'], errors='coerce').fillna(0)
            data['p_churn'] = score(data).round(3)
            data['band'] = band_of(data['p_churn'].values, threshold)
            data['contact'] = np.where(data['p_churn'] >= threshold,
                                       'Yes', '')
            n = len(data)
            n_high = int((data['contact'] == 'Yes').sum())

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi('Customers scored', f'{n:,}', BLUE)
            with k2:
                kpi('Above threshold', f'{n_high:,}', ORANGE,
                    f'{n_high / n:.0%} of the file')
            with k3:
                kpi('Average risk', f"{data['p_churn'].mean():.0%}", TEAL,
                    f'Training average: {OVERALL_CHURN:.1%}')
            with k4:
                kpi('Expected churners', f"{data['p_churn'].sum():.0f}",
                    PURPLE, 'Sum of all probabilities')

            g1, g2 = st.columns([3, 2])
            with g1:
                section('Distribution of churn probability')
                h = alt.Chart(data).mark_bar().encode(
                    x=alt.X('p_churn:Q',
                            bin=alt.Bin(maxbins=20, extent=[0, 1]),
                            axis=alt.Axis(format='%'),
                            title='Churn probability'),
                    y=alt.Y('count()', title='Customers'),
                    color=alt.Color('band:N', scale=BAND_SCALE,
                                    legend=alt.Legend(title='Band')))
                tr = alt.Chart(pd.DataFrame({'t': [threshold]})).mark_rule(
                    color=NAVY, strokeDash=[6, 4], strokeWidth=2).encode(
                    x='t:Q')
                st.altair_chart((h + tr).properties(height=280,
                                                    width='container'))
            with g2:
                section('Customers by risk band')
                counts = (data['band'].value_counts()
                          .rename_axis('band').reset_index(name='n'))
                donut = alt.Chart(counts).mark_arc(
                    innerRadius=60, outerRadius=110).encode(
                    theta='n:Q',
                    color=alt.Color('band:N', scale=BAND_SCALE,
                                    legend=alt.Legend(title='Band')),
                    tooltip=['band', 'n'])
                st.altair_chart(donut.properties(height=280))

            section('Average risk by contract type')
            by_c = (data.groupby('Contract')['p_churn'].mean()
                    .reset_index())
            bc = alt.Chart(by_c).mark_bar(cornerRadiusEnd=6).encode(
                y=alt.Y('Contract:N', sort='-x', title=None),
                x=alt.X('p_churn:Q', axis=alt.Axis(format='%'),
                        title='Average churn probability'),
                color=alt.Color('Contract:N', scale=CONTRACT_SCALE,
                                legend=None),
                tooltip=['Contract',
                         alt.Tooltip('p_churn:Q', format='.1%')])
            st.altair_chart(bc.properties(height=40 + 50 * len(by_c),
                                          width='container'))

            section('Highest-risk customers (top 50)')
            shown = data.sort_values('p_churn', ascending=False).head(50)
            st.dataframe(
                shown, hide_index=True,
                column_config={
                    'p_churn': st.column_config.ProgressColumn(
                        'p_churn', min_value=0.0, max_value=1.0,
                        format='%.3f')})
            st.download_button(
                '⬇️ Download scored CSV',
                data.drop(columns=['band']).to_csv(index=False),
                'scored_customers.csv', 'text/csv')

# =====================================================================
# Tab 3: model insights (values from the Week 3 notebook)
# =====================================================================
with tab3:
    st.markdown("<div class='note'>These charts show results from the "
                "Week 3 notebook, measured once on the training data (CV) "
                "and on the held-out test set. They do not change with the "
                "sidebar.</div>", unsafe_allow_html=True)

    i1, i2 = st.columns(2)
    with i1:
        section('🏁 Model comparison (5-fold CV AUC ± std)')
        pts = alt.Chart(MODEL_TABLE).mark_point(
            size=260, filled=True).encode(
            x=alt.X('Model:N', sort=None, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('CV AUC:Q', scale=alt.Scale(domain=[0.82, 0.875])),
            color=alt.Color('Model:N', legend=None,
                            scale=alt.Scale(range=[BLUE, GREEN, ORANGE])),
            tooltip=['Model', 'CV AUC', 'Std'])
        err = alt.Chart(MODEL_TABLE).mark_rule(strokeWidth=4).encode(
            x=alt.X('Model:N', sort=None), y='Low:Q', y2='High:Q',
            color=alt.Color('Model:N', legend=None,
                            scale=alt.Scale(range=[BLUE, GREEN, ORANGE])))
        st.altair_chart((err + pts).properties(height=300,
                                               width='container'))
        st.caption('The three error bars overlap: the models are tied '
                   'within noise. XGBoost is nominally best (0.8502).')
    with i2:
        section('🎯 Churn rate by customer segment')
        seg = alt.Chart(SEGMENTS).mark_bar(cornerRadiusEnd=6).encode(
            y=alt.Y('Segment:N', sort='-x', title=None),
            x=alt.X('Churn rate:Q', axis=alt.Axis(format='%')),
            color=alt.Color('Churn rate:Q', legend=None,
                            scale=alt.Scale(range=[GREEN, AMBER, ORANGE])),
            tooltip=['Segment', 'Customers',
                     alt.Tooltip('Churn rate:Q', format='.0%')])
        avg = alt.Chart(pd.DataFrame({'x': [OVERALL_CHURN]})).mark_rule(
            color=NAVY, strokeDash=[6, 4]).encode(x='x:Q')
        st.altair_chart((seg + avg).properties(height=300,
                                               width='container'))
        st.caption('Dashed line = overall churn rate (26.5%). The riskiest '
                   'segment holds about half of all churners.')

    i3, i4 = st.columns(2)
    with i3:
        section('🧮 Test-set confusion matrix (threshold 0.5)')
        cm = alt.Chart(CONFUSION).mark_rect(cornerRadius=6).encode(
            x=alt.X('Predicted:N', sort=['Churn', 'Stay']),
            y=alt.Y('Actual:N', sort=['Churned', 'Stayed']),
            color=alt.Color('Kind:N', legend=None, scale=alt.Scale(
                domain=['TP', 'FN', 'FP', 'TN'],
                range=[GREEN, ORANGE, AMBER, TEAL])))
        cm_text = alt.Chart(CONFUSION).mark_text(
            fontSize=20, fontWeight='bold', color='white').encode(
            x=alt.X('Predicted:N', sort=['Churn', 'Stay']),
            y=alt.Y('Actual:N', sort=['Churned', 'Stayed']),
            text='Count:Q')
        st.altair_chart((cm + cm_text).properties(height=260,
                                                  width='container'))
        st.caption('Rebuilt by hand from recall 0.521 and precision 0.659. '
                   'At 0.5 the model misses 179 of 374 churners, which is '
                   'why a lower threshold is used.')
    with i4:
        section('💰 Cost of three strategies (test set, PKR)')
        costs = pd.DataFrame({
            'Strategy': ['Contact nobody', 'Model at 0.5',
                         'Contact everybody'],
            'Cost': [374 * COST_FN, 179 * COST_FN + 101 * COST_FP,
                     1035 * COST_FP]})
        cb = alt.Chart(costs).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X('Strategy:N', sort=None, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Cost:Q', axis=alt.Axis(format='~s'),
                    title='Total cost (PKR)'),
            color=alt.Color('Strategy:N', legend=None, scale=alt.Scale(
                range=[ORANGE, BLUE, AMBER])),
            tooltip=['Strategy', alt.Tooltip('Cost:Q', format=',')])
        st.altair_chart(cb.properties(height=260, width='container'))
        st.caption('FN = PKR 6,000 and FP = PKR 1,000. A threshold of 0.5 '
                   'is not cost-optimal; the app uses a lower, '
                   'cost-based threshold.')

    section('📋 Segment profiles')
    st.dataframe(
        SEGMENTS, hide_index=True,
        column_config={
            'Churn rate': st.column_config.ProgressColumn(
                'Churn rate', min_value=0.0, max_value=0.5,
                format='%.2f')})

# =====================================================================
# Tab 4: about / model card
# =====================================================================
with tab4:
    a1, a2 = st.columns([3, 2])
    with a1:
        section('Model card: Churn Risk Advisor')
        st.markdown(f"""
**Intended use:** rank existing telecom customers by churn risk so a
retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** {meta.get('training_data', 'IBM Telco Customer Churn')}, 26.5% churn.

**Model:** {meta['model_name']}, chosen by 5-fold CV. Features: {len(COLS)} encoded columns.

**Performance:** CV AUC {meta['cv_auc']:.3f} ± {meta['cv_auc_std']:.3f}; test AUC {meta.get('test_auc', float('nan')):.3f} (test set used once).

**Threshold:** {float(meta['threshold']):.2f}, from a cost analysis (FN = PKR {COST_FN:,}, FP = PKR {COST_FP:,}).

**Limitations:** one US dataset from one period; no Pakistan data;
associations, not causes; performance may drift as plans change.
""")
    with a2:
        section('Metadata')
        st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
        with st.expander(f'All {len(COLS)} model input columns'):
            st.write(COLS)

st.divider()
st.caption('Churn Risk Advisor | Decision support only | Trained on one US '
           'telecom dataset, not validated for other markets.')
