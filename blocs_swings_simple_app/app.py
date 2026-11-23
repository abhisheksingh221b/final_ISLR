# Blocs and Swings - Streamlit dashboard
# Run with:  streamlit run app.py

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Blocs and Swings', page_icon='🗳️', layout='wide')

# The data files can sit next to app.py or inside an "app_data" folder
FOLDER = Path(__file__).parent
DATA = FOLDER / 'app_data' if (FOLDER / 'app_data').exists() else FOLDER

VOTE_COLORS = {'yes': 'green', 'abstain': 'orange', 'no': 'red'}


# ---------------------------------------------------------------
# Load the files once (Streamlit remembers them between clicks)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    blocs = pd.read_csv(DATA / 'blocs.csv')
    preds = pd.read_parquet(DATA / 'predictions.parquet')
    resolutions = pd.read_csv(DATA / 'resolutions.csv')
    return blocs, preds, resolutions


blocs, preds, resolutions = load_data()


# ---------------------------------------------------------------
# Page 1: Voting blocs (Q1)
# ---------------------------------------------------------------
def blocs_page():
    st.title('Voting blocs at the UN')
    st.write('Countries grouped by how they vote (PCA + K-Means, 3 groups per decade).')

    decade = st.select_slider('Decade', options=sorted(blocs['decade'].unique()), value=1980)
    d = blocs[blocs['decade'] == decade]

    world_map = px.choropleth(d, locations='iso3', color='bloc', hover_name='country',
                              title=f'Voting blocs in the {decade}s')
    st.plotly_chart(world_map)

    pca_map = px.scatter(d, x='x', y='y', color='bloc', hover_name='country',
                         title='PCA map: countries close together vote alike')
    st.plotly_chart(pca_map)

    st.subheader('Who is in each bloc?')
    for bloc_name in d['bloc'].unique():
        members = sorted(d[d['bloc'] == bloc_name]['country'])
        with st.expander(f'{bloc_name} ({len(members)} countries)'):
            st.write(', '.join(members))


# ---------------------------------------------------------------
# Page 2: Predict a resolution (Q2 and Q3)
# ---------------------------------------------------------------
def predict_page():
    st.title('Predict how countries vote on a resolution')
    st.write('The model learned from 1980-2009 and never saw these 2010-2019 votes.')

    year = st.selectbox('Year', sorted(preds['year'].unique(), reverse=True))
    ids_this_year = preds[preds['year'] == year]['rcid'].unique()
    options = resolutions[resolutions['rcid'].isin(ids_this_year)]
    titles = dict(zip(options['rcid'], options['descr'].fillna(options['short']).str[:110]))
    rcid = st.selectbox('Resolution', list(titles), format_func=lambda r: titles[r])

    info = resolutions[resolutions['rcid'] == rcid].iloc[0]
    st.write('**Resolution:**', info['descr'])
    st.caption(f"Topic: {info['main_topic']} · Similarity to the closest earlier resolution: {info['similarity']:.2f}")

    r = preds[preds['rcid'] == rcid]
    col1, col2, col3 = st.columns(3)
    col1.metric('Countries voting', len(r))
    col2.metric('Correct predictions', f"{(r['vote'] == r['predicted']).mean():.0%}")
    col3.metric('Swing countries flagged', int(r['swing_flag'].sum()))

    # Q2: predicted vs actual votes
    which = st.radio('Show on the map', ['predicted', 'vote'], horizontal=True,
                     format_func=lambda c: 'Predicted vote' if c == 'predicted' else 'Actual vote')
    vote_map = px.choropleth(r, locations='iso3', color=which, hover_name='country',
                             color_discrete_map=VOTE_COLORS)
    st.plotly_chart(vote_map)

    # Q3: swing countries
    st.subheader('Swing countries: most likely to change their old vote')
    st.caption('Old vote = the country\'s vote on the most similar earlier resolution. '
               'A country is flagged when its chance of change is above 50%.')
    swing = r.sort_values('p_change', ascending=False).head(10)
    swing = swing[['country', 'old_vote', 'vote', 'p_change', 'swing_flag', 'changed']].copy()
    swing['p_change'] = (swing['p_change'] * 100).round(0)
    swing.columns = ['Country', 'Old vote', 'Actual vote', 'Chance of change (%)', 'Flagged', 'Really changed']
    st.dataframe(swing, hide_index=True)

    changed = r[r['changed']]
    if len(changed) > 0:
        st.write(f"**{len(changed)}** countries really changed their vote on this resolution. "
                 f"The flags caught **{int(changed['swing_flag'].sum())}** of them.")


# ---------------------------------------------------------------
# Sidebar menu
# ---------------------------------------------------------------
st.sidebar.title('🗳️ Blocs and Swings')
page = st.sidebar.radio('Go to', ['Voting blocs', 'Predict a resolution'])
st.sidebar.caption('Data: UN General Assembly votes, 1946-2019 (Voeten et al., via TidyTuesday).')

if page == 'Voting blocs':
    blocs_page()
else:
    predict_page()
