"""
Nassau Candy Distributor — Factory Reallocation & Shipping Optimization
Decision Intelligence Dashboard

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

from geo import STATE_COORDS, FACTORY_COORDS, distance_to_factory

st.set_page_config(page_title="Nassau Candy | Factory Reallocation", layout="wide", page_icon="🏭")

PRODUCT_FACTORY = {
    'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
    'Laffy Taffy': 'Sugar Shack', 'SweeTARTS': 'Sugar Shack', 'Nerds': 'Sugar Shack',
    'Fun Dip': 'Sugar Shack', 'Fizzy Lifting Drinks': 'Sugar Shack',
    'Everlasting Gobstopper': 'Secret Factory', 'Hair Toffee': 'The Other Factory',
    'Lickable Wallpaper': 'Secret Factory', 'Wonka Gum': 'Secret Factory',
    'Kazookles': 'The Other Factory',
}

# ---------------------------------------------------------------- data ----
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
    df['Lead Time (days)'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Current Factory'] = df['Product Name'].map(PRODUCT_FACTORY)
    df = df[df['State/Province'].isin(STATE_COORDS.keys())].copy()
    df['Distance to Current Factory (mi)'] = df.apply(
        lambda r: distance_to_factory(r['State/Province'], r['Current Factory']), axis=1)
    df['Profit Margin (%)'] = (df['Gross Profit'] / df['Sales']) * 100
    df['Cost per Unit'] = df['Cost'] / df['Units']
    df = df.dropna(subset=['Distance to Current Factory (mi)'])
    return df

@st.cache_resource
def train_models(df):
    features_num = ['Distance to Current Factory (mi)', 'Sales', 'Units', 'Cost per Unit']
    features_cat = ['Ship Mode', 'Region', 'Division', 'Current Factory']
    X = df[features_num + features_cat]
    y = df['Lead Time (days)']
    pre = ColumnTransformer([
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    }
    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([('pre', pre), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            'Model': name,
            'RMSE': mean_squared_error(y_test, pred) ** 0.5,
            'MAE': mean_absolute_error(y_test, pred),
            'R2': r2_score(y_test, pred),
        })
        fitted[name] = pipe
    results_df = pd.DataFrame(results).sort_values('RMSE')
    best_name = results_df.iloc[0]['Model']
    return results_df, fitted, best_name, features_num, features_cat

@st.cache_data
def cluster_routes(df):
    route_agg = df.groupby(['Region', 'Product Name']).agg(
        avg_lead_time=('Lead Time (days)', 'mean'),
        avg_distance=('Distance to Current Factory (mi)', 'mean'),
        avg_margin=('Profit Margin (%)', 'mean'),
        orders=('Order ID', 'count'),
    ).reset_index()
    km_X = StandardScaler().fit_transform(route_agg[['avg_lead_time', 'avg_distance', 'avg_margin']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    route_agg['cluster'] = kmeans.fit_predict(km_X)
    summary = route_agg.groupby('cluster')['avg_lead_time'].mean().sort_values(ascending=False)
    order = summary.index.tolist()
    labels = {order[0]: 'Consistently Slow'}
    for c in order[1:-1]:
        labels[c] = 'Moderate Risk'
    labels[order[-1]] = 'High Performing'
    route_agg['cluster_label'] = route_agg['cluster'].map(labels)
    return route_agg

@st.cache_data
def simulate_scenarios(df, _best_model, cost_per_mile_per_unit=0.0015):
    best_model = _best_model
    products = df[['Product Name', 'Division', 'Current Factory']].drop_duplicates()
    regions = df['Region'].unique()
    rows = []
    for _, prow in products.iterrows():
        pname, division, cur_factory = prow['Product Name'], prow['Division'], prow['Current Factory']
        prod_df = df[df['Product Name'] == pname]
        baseline_sales = prod_df['Sales'].mean()
        baseline_units = prod_df['Units'].mean()
        baseline_cost_unit = prod_df['Cost per Unit'].mean()
        baseline_margin = prod_df['Profit Margin (%)'].mean()
        for region in regions:
            region_states = df[df['Region'] == region]['State/Province']
            if region_states.empty:
                continue
            rep_state = region_states.mode()[0]
            cur_dist = distance_to_factory(rep_state, cur_factory)
            for factory in FACTORY_COORDS.keys():
                dist = distance_to_factory(rep_state, factory)
                if np.isnan(dist):
                    continue
                row = pd.DataFrame([{
                    'Distance to Current Factory (mi)': dist, 'Sales': baseline_sales,
                    'Units': baseline_units, 'Cost per Unit': baseline_cost_unit,
                    'Ship Mode': 'Standard Class', 'Region': region, 'Division': division,
                    'Current Factory': factory,
                }])
                pred_lt = best_model.predict(row)[0]
                dist_delta = dist - cur_dist
                price_per_unit = baseline_sales / baseline_units if baseline_units else np.nan
                cost_delta = dist_delta * cost_per_mile_per_unit
                margin_delta = -(cost_delta / price_per_unit * 100) if price_per_unit else 0
                margin_delta = float(np.clip(margin_delta, -15, 15))
                rows.append({
                    'Product Name': pname, 'Division': division, 'Region': region,
                    'Current Factory': cur_factory, 'Candidate Factory': factory,
                    'Is Current': factory == cur_factory,
                    'Predicted Lead Time (days)': round(pred_lt, 1),
                    'Distance (mi)': round(dist, 0),
                    'Est. Profit Margin (%)': round(baseline_margin + margin_delta, 2),
                    'Margin Delta (pts)': round(margin_delta, 2),
                })
    return pd.DataFrame(rows)

def build_recommendations(scenarios, priority):
    # priority: 0 = pure speed, 100 = pure profit
    speed_w = (100 - priority) / 100 * 1.2
    profit_w = priority / 100 * 5
    recs = []
    for (pname, region), grp in scenarios.groupby(['Product Name', 'Region']):
        cur = grp[grp['Is Current']].iloc[0]
        alt = grp[~grp['Is Current']].copy()
        alt['lead_time_gain_pct'] = (cur['Predicted Lead Time (days)'] - alt['Predicted Lead Time (days)']) / cur['Predicted Lead Time (days)'] * 100
        alt['margin_gain_pts'] = alt['Est. Profit Margin (%)'] - cur['Est. Profit Margin (%)']
        alt['score'] = alt['lead_time_gain_pct'] * speed_w + alt['margin_gain_pts'] * profit_w
        best_alt = alt.sort_values('score', ascending=False).iloc[0]
        recs.append({
            'Product Name': pname, 'Region': region,
            'Current Factory': cur['Current Factory'],
            'Recommended Factory': best_alt['Candidate Factory'],
            'Lead Time Reduction (%)': round(best_alt['lead_time_gain_pct'], 1),
            'Profit Impact (pts)': round(best_alt['margin_gain_pts'], 2),
            'Confidence Score': round(min(99, 55 + abs(best_alt['score'])), 1),
            'Risk Flag': 'High Risk' if best_alt['margin_gain_pts'] < -5 else ('Watch' if best_alt['margin_gain_pts'] < 0 else 'Safe'),
        })
    return pd.DataFrame(recs)

# ---------------------------------------------------------------- load ----
df = load_data()
results_df, fitted_models, best_name, features_num, features_cat = train_models(df)
best_model = fitted_models[best_name]
route_agg = cluster_routes(df)

# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("🏭 Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")
page = st.sidebar.radio("Dashboard", [
    "Overview",
    "Factory Optimization Simulator",
    "What-If Scenario Analysis",
    "Recommendation Dashboard",
    "Risk & Impact Panel",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
region_filter = st.sidebar.multiselect("Region", sorted(df['Region'].unique()), default=list(df['Region'].unique()))
shipmode_filter = st.sidebar.multiselect("Ship Mode", sorted(df['Ship Mode'].unique()), default=list(df['Ship Mode'].unique()))
priority = st.sidebar.slider("Optimization Priority — Speed ↔ Profit", 0, 100, 50,
                              help="0 = prioritize fastest shipping, 100 = prioritize profit margin")

fdf = df[df['Region'].isin(region_filter) & df['Ship Mode'].isin(shipmode_filter)]

scenarios = simulate_scenarios(df, best_model)
recs_df = build_recommendations(scenarios, priority)

# ---------------------------------------------------------------- pages ----
if page == "Overview":
    st.title("Factory Reallocation & Shipping Optimization")
    st.caption("Decision intelligence system — Nassau Candy Distributor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{len(fdf):,}")
    c2.metric("Avg. Lead Time", f"{fdf['Lead Time (days)'].mean():.0f} days")
    c3.metric("Avg. Profit Margin", f"{fdf['Profit Margin (%)'].mean():.1f}%")
    c4.metric("Avg. Shipping Distance", f"{fdf['Distance to Current Factory (mi)'].mean():.0f} mi")

    st.warning(
        "**Data quality note:** The `Ship Date` field in the source data is inconsistent with `Order Date` "
        "(offsets of 2–5+ years across records), which is almost certainly a data-entry/generation artifact "
        "rather than real operational lead time. All lead-time figures in this dashboard should be read as "
        "**relative/comparative indicators only**, not literal day counts. See the Research Report for detail."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(fdf.groupby(['Region', 'Ship Mode'], as_index=False)['Lead Time (days)'].mean(),
                     x='Region', y='Lead Time (days)', color='Ship Mode', barmode='group',
                     title='Average Lead Time by Region & Ship Mode')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(fdf.groupby('Current Factory', as_index=False)['Profit Margin (%)'].mean().sort_values('Profit Margin (%)'),
                     x='Profit Margin (%)', y='Current Factory', orientation='h',
                     title='Average Profit Margin by Factory')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(fdf.sample(min(1500, len(fdf)), random_state=1),
                          x='Distance to Current Factory (mi)', y='Lead Time (days)', color='Division',
                          opacity=0.5, title='Lead Time vs Shipping Distance')
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(fdf, x='Division', y='Lead Time (days)', title='Lead Time Distribution by Division')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance")
    st.dataframe(results_df.style.format({'RMSE': '{:.1f}', 'MAE': '{:.1f}', 'R2': '{:.3f}'}), use_container_width=True)
    st.caption(f"Best model selected: **{best_name}**. Note the low/negative R² — this reflects the Ship Date "
               "data-quality issue above rather than a modeling failure; see Research Report §4.")

elif page == "Factory Optimization Simulator":
    st.title("🔧 Factory Optimization Simulator")
    st.caption("Select a product to view predicted performance across all five factories.")

    product = st.selectbox("Product", sorted(df['Product Name'].unique()))
    region = st.selectbox("Customer Region", sorted(df['Region'].unique()))

    sub = scenarios[(scenarios['Product Name'] == product) & (scenarios['Region'] == region)].copy()
    sub = sub.sort_values('Predicted Lead Time (days)')
    cur_factory = df[df['Product Name'] == product]['Current Factory'].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sub, x='Candidate Factory', y='Predicted Lead Time (days)',
                     color=sub['Candidate Factory'].eq(cur_factory).map({True: 'Current', False: 'Alternative'}),
                     title=f'Predicted Lead Time by Factory — {product} ({region})',
                     labels={'color': 'Assignment'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(sub, x='Candidate Factory', y='Est. Profit Margin (%)',
                     color=sub['Candidate Factory'].eq(cur_factory).map({True: 'Current', False: 'Alternative'}),
                     title=f'Estimated Profit Margin by Factory — {product} ({region})',
                     labels={'color': 'Assignment'})
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sub[['Candidate Factory', 'Distance (mi)', 'Predicted Lead Time (days)', 'Est. Profit Margin (%)', 'Is Current']],
                 use_container_width=True, hide_index=True)

elif page == "What-If Scenario Analysis":
    st.title("🔁 What-If Scenario Analysis")
    st.caption("Compare current vs. recommended factory assignments.")

    product = st.selectbox("Product", sorted(df['Product Name'].unique()), key='whatif_product')
    region = st.selectbox("Customer Region", sorted(df['Region'].unique()), key='whatif_region')

    rec_row = recs_df[(recs_df['Product Name'] == product) & (recs_df['Region'] == region)]
    cur_factory = df[df['Product Name'] == product]['Current Factory'].iloc[0]
    cur = scenarios[(scenarios['Product Name'] == product) & (scenarios['Region'] == region) & (scenarios['Is Current'])].iloc[0]

    if not rec_row.empty:
        rec = rec_row.iloc[0]
        rec_scenario = scenarios[(scenarios['Product Name'] == product) & (scenarios['Region'] == region) &
                                  (scenarios['Candidate Factory'] == rec['Recommended Factory'])].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Factory", cur_factory)
        c2.metric("Recommended Factory", rec['Recommended Factory'])
        c3.metric("Confidence", f"{rec['Confidence Score']:.0f}%")

        colA, colB = st.columns(2)
        colA.metric("Lead Time (Current)", f"{cur['Predicted Lead Time (days)']:.0f} d")
        colA.metric("Lead Time (Recommended)", f"{rec_scenario['Predicted Lead Time (days)']:.0f} d",
                     delta=f"{rec['Lead Time Reduction (%)']:+.1f}%")
        colB.metric("Profit Margin (Current)", f"{cur['Est. Profit Margin (%)']:.1f}%")
        colB.metric("Profit Margin (Recommended)", f"{rec_scenario['Est. Profit Margin (%)']:.1f}%",
                     delta=f"{rec['Profit Impact (pts)']:+.2f} pts")

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Lead Time (days)', x=['Current', 'Recommended'],
                              y=[cur['Predicted Lead Time (days)'], rec_scenario['Predicted Lead Time (days)']]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reassignment currently outperforms the existing factory for this product/region combination "
                "under the selected optimization priority — the current assignment already looks efficient.")

elif page == "Recommendation Dashboard":
    st.title("📋 Recommendation Dashboard")
    st.caption("Ranked factory reassignment suggestions across all products & regions.")

    sort_col = st.selectbox("Sort by", ['Lead Time Reduction (%)', 'Profit Impact (pts)', 'Confidence Score'])
    top_recs = recs_df.sort_values(sort_col, ascending=False)

    st.dataframe(top_recs, use_container_width=True, hide_index=True)

    st.subheader("Expected Efficiency Gains")
    fig = px.scatter(top_recs, x='Lead Time Reduction (%)', y='Profit Impact (pts)',
                      size='Confidence Score', color='Product Name', hover_data=['Region', 'Recommended Factory'],
                      title='Reassignment Opportunities: Speed Gain vs Profit Impact')
    st.plotly_chart(fig, use_container_width=True)

    csv = top_recs.to_csv(index=False).encode('utf-8')
    st.download_button("Download Recommendations (CSV)", csv, "recommendations.csv", "text/csv")

elif page == "Risk & Impact Panel":
    st.title("⚠️ Risk & Impact Panel")
    st.caption("Profit impact alerts and high-risk reassignment warnings.")

    risky = recs_df[recs_df['Risk Flag'] != 'Safe'].sort_values('Profit Impact (pts)')
    safe = recs_df[recs_df['Risk Flag'] == 'Safe']

    c1, c2, c3 = st.columns(3)
    c1.metric("Safe Recommendations", len(safe))
    c2.metric("Watch List", (recs_df['Risk Flag'] == 'Watch').sum())
    c3.metric("High Risk", (recs_df['Risk Flag'] == 'High Risk').sum())

    if not risky.empty:
        st.subheader("⚠️ Flagged Reassignments")
        st.dataframe(risky[['Product Name', 'Region', 'Current Factory', 'Recommended Factory',
                             'Lead Time Reduction (%)', 'Profit Impact (pts)', 'Risk Flag']],
                     use_container_width=True, hide_index=True)
    else:
        st.success("No high-risk reassignments detected under current priority settings.")

    st.subheader("Route Cluster Risk Map")
    fig = px.scatter(route_agg, x='avg_distance', y='avg_lead_time', color='cluster_label',
                      size='orders', hover_data=['Product Name', 'Region'],
                      title='Route Clusters: Consistently Slow / Moderate Risk / High Performing')
    st.plotly_chart(fig, use_container_width=True)
