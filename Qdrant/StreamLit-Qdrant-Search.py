# ## Build the app in Streamlit

# ### 1. Install Streamlit
#!pip install streamlit

#!streamlit hello

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ### 2. Create Store and Effects
import os
import pandas as pd
from dataclasses import dataclass
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import torch

@dataclass
class AppStore:
	api_url = "http://localhost:6333" #API_KEY
	index_name = "HospitalCharges" #INDEX_NAME

# Use the AppEffect class to connect the app to QDrant (with init) and to the index (docs):

class AppEffect:

    def __init__(self, store: AppStore):
        self.store = store

    def init_qdrant(self):
        self.qdrant = QdrantClient(self.store.api_url) 

    def init_qdrant_searchCollection(self):        
        self.qdrant.get_collection(collection_name=self.store.index_name)        


SUGGESTED_QUERY = ['Influenza', 'backache', 'pnuemonia', 'appendicitis', 'Admin of Flumist Influenza Vaccine']

class PageHome:
	
    def __init__(self, app):
        self.app = app

    @property
    def index(self):
        return self.app.effect.init_qdrant_searchCollection()
	
    def render(self):
        self.render_suggested_queries()
        submitted = self.render_search_form()
        if submitted:
            self.render_search_results()

    def render_suggested_queries(self):
        st.markdown("Try one of these queries:")
        columns = st.columns(len(SUGGESTED_QUERY))
        for col, query in zip(columns, SUGGESTED_QUERY):
            with col:
                if st.button(query):
                    st.session_state.queryname = query
    
    def render_search_form(self):
        st.markdown("Or enter a query:")
        with st.form("search_form"):
            if st.session_state.get('queryname'):
               st.session_state.queryname = st.text_input("Query", value=st.session_state.queryname)
            else:
                st.session_state.queryname = st.text_input("Query")
            return st.form_submit_button("Search")
	
	
    def render_search_results(self):
        dfResults = []
        with st.spinner("Searching for " + st.session_state.queryname):
            result = self.app.effect.qdrant.search(collection_name="HospitalCharges", 
                                        query_vector=self.app.model.encode(st.session_state.queryname).tolist(),
                                        limit=300)
            for hit in result:
                print(hit.payload, "score:", hit.score)

        if (len(result) == 0):
            return st.markdown("This query found no match. Somethin seems to be off")

        dfResults = pd.DataFrame([hit.payload for hit in result])
        dfResults['query_score'] = [hit.score for hit in result]
	        
        with st.container():
            st.dataframe(dfResults.style.highlight_max(axis=0))
        
        
        col1, col2, col3 = st.columns(3)
        a4_dims = (4, 2)
        sns.set(font_scale=0.5)
        
        with col1:
            # # Display the DataFrame in Streamlit
            st.dataframe(dfResults[['procedure_description', 'payer', 'ip_price', 'ip_expected_reimbursement']])    

        with col2:
            # Select features to display scatter plot
            # feature_x = st.selectbox('Select feature for x axis', dfResults.columns)
            # feature_y = st.selectbox('Select feature for y axis', dfResults.columns)

            # Display scatter plot
            fig, ax = plt.subplots(figsize=a4_dims)
            # sns.scatterplot(data=dfResults, x=feature_x, y=feature_y, hue=dfResults.payer, ax=ax)
            sns.scatterplot(data=dfResults, x=dfResults.ip_price, y=dfResults.ip_expected_reimbursement, hue=dfResults.ip_expected_reimbursement/dfResults.ip_price, ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col3:
            fig2, ax2 = plt.subplots(figsize=a4_dims)
            sns.histplot(data=dfResults, x="ip_price", y="ip_expected_reimbursement", hue="payer", legend=False)
            st.pyplot(fig2, use_container_width=False)


        # # Display the DataFrame in Streamlit
        # st.dataframe(dfResults)

        # # Show general information about the dataset
        # st.text(dfResults.info())

        # # Show statistical information about the dataset
        # st.write(dfResults.describe())

        # # Select a feature to display histogram
        # feature = st.selectbox('Select a feature', dfResults.columns)

        # # Plot histogram
        # fig, ax = plt.subplots()
        # ax.hist(dfResults[feature], bins=20)

        # # Set the title and labels
        # ax.set_title(f'Histogram of {feature}')
        # ax.set_xlabel(feature)
        # ax.set_ylabel('Frequency')

        # # Display the plot
        # st.pyplot(fig)

class App:
    title = "Healthcare Charges"

    def __init__(self):
        self.store = AppStore()
        self.effect = AppEffect(self.store)
        self.effect.init_qdrant()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print('Sorry no cuda.')
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def render(self):        
        st.set_page_config(layout="wide")
        st.title(self.title)
        PageHome(self).render()

if __name__ == "__main__":
    App().render()