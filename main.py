import streamlit as st
import pandas as pd 
import yfinance as yf 
import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC

def main():
    st.title('PortFolio Diversification')
    with st.sidebar:
        st.title("PortFolio Diversification")
        with st.expander("Navigation",expanded=True):
            page = st.selectbox(
            label="Choose your ML backend",
            options=["Classical Machine Learning", "Quantum Machine Learning"]
            )
    
    def load_df():
        tk_list = pd.read_html("https://en.wikipedia.org/wiki/list_of_S%26P_500_companies",header=0)
        df= tk_list[0]
        return df

    def get_metrics(tkr,close_df):
        for i in tkr:
            data = yf.download(i,start,end)
            close_df[i] = data['Adj Close']
            
        close_df = close_df.fillna(method="ffill")
        mu = expected_returns.mean_historical_return(close_df)
        s = risk_models.sample_cov(close_df)
        ef = EfficientFrontier(mu,s)
        return mu,s,ef

    def get_details():
        with st.expander("**Details of Stock**",expanded=False):
            tr = st.selectbox("Choose the Stock Ticker",tkr)
            if tr:
                st.session_state.ticker=True
                if st.session_state.ticker:
                    resp = yf.Ticker(tr)
                    info = resp.info
                    name = info.get('longName')
                    country = info.get('country')
                    ceo = info.get('companyOfficers')[0]['name']
                    currency = info.get('currency')
                    summ = info.get('longBusinessSummary')
                    ind = info.get('industry')
                    website = info.get('website')
                    rev = info.get('totalRevenue')

                    st.subheader(name)
                    st.write(f'**Industry** : {ind}')
                    st.write(F'**Chief Executive Officer**: {ceo}')
                    st.write(f'**Country** : {country}')
                    st.write(f'**Currency** : {currency}')
                    st.write(f'**Total Revenue** : {rev}')
                    st.write(f'**Summary** : {summ}')

                    
                    if st.button("View Price Table and Graph"):
                        st.subheader("Price Table")
                        stock = yf.download(tr,start,end)
                        stock.reset_index(inplace=True)
                        st.write(stock)
                        
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(x=stock['Date'],y=stock['Open'],name="Stock Open Price"))
                        fig.add_trace(go.Scatter(x=stock['Date'],y=stock['Close'],name="Stock Close Price "))
                        fig.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig.update_layout(xaxis_title="Price",yaxis_title="Date")        
                        st.plotly_chart(fig)
    
    def plot_results(preds):
        fig1 = go.Figure()
        for i in set(preds):
            cluster_points = X[preds == i]
            fig1.add_trace(go.Scatter(x=cluster_points[:, 0],y=cluster_points[:, 1],mode='markers',name=f"Cluster {i}"))

        fig1.layout.update(title_text="Cluster plots")
        fig1.update_layout(xaxis_title="Returns",yaxis_title="Variances")
        st.plotly_chart(fig1)

    if page == "Classical Machine Learning":
        st.header("Portfolio Classification using Classical Machine Learning")
        n = st.number_input("Enter the number of tickers : ",min_value=0,step=1)
        strategy = st.selectbox("Select Clustering type", options=[
            "K-Means",
            "Agglomerative",
            "DBSCAN",
            "Gaussian Mixture"
            ])

        df = load_df()
        close_df = pd.DataFrame()

        if strategy == "K-Means":
            clster = st.number_input("Enter the number of clusters : ",min_value=2)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("K Means Clustering")
                    kmeans = KMeans(n_clusters = clster,random_state=42)
                    kmeans.fit(X)
                    km = kmeans.predict(X)
                    
                    df["KMeans Categories"] = km
                    pdf = df[["Tickers","KMeans Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    sc = silhouette_score(X,km)
                    st.write(f"**Silhouette Score** : {sc}")

                    tables_km = {}

                    for category in df["KMeans Categories"].unique():
                        category_df = df[df["KMeans Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(km)

                    st.header("Cluster Details")
                    for i in df["KMeans Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["KMeans Categories"]==i]["Tickers"])

        elif strategy == "Agglomerative":
            clster = st.number_input("Enter the number of clusters : ",min_value=2)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("Agglomerative Clustering")
                    Agglomerative = AgglomerativeClustering(n_clusters = clster)
                    agg = Agglomerative.fit_predict(X)
                    df["Agglomerative Categories"] = agg
                    pdf = df[["Tickers","Agglomerative Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    sc = silhouette_score(X,agg)
                    st.write(f"**Silhouette Score** : {sc}")

                    tables_km = {}

                    for category in df["Agglomerative Categories"].unique():
                        category_df = df[df["Agglomerative Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(agg)

                    st.header("Cluster Details")
                    for i in df["Agglomerative Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["Agglomerative Categories"]==i]["Tickers"])
                    
        elif strategy == "DBSCAN":
            clster = st.number_input("Enter the number of clusters : ",min_value=2)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("DBSCAN Clustering")
                    dbs = DBSCAN(eps=0.2,min_samples=clster)
                    dbsc = dbs.fit_predict(X)
                    df["DBSCAN Categories"] = dbsc
                    pdf = df[["Tickers","DBSCAN Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    sc = silhouette_score(X,dbsc)
                    st.write(f"**Silhouette Score** : {sc}")

                    tables_km = {}

                    for category in df["DBSCAN Categories"].unique():
                        category_df = df[df["DBSCAN Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(dbsc)

                    st.header("Cluster Details")
                    for i in df["DBSCAN Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["DBSCAN Categories"]==i]["Tickers"])

        elif strategy == "Gaussian Mixture":
            clster = st.number_input("Enter the number of clusters : ",min_value=2)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("Gaussian Mixture Model Clustering")
                    GMM = GaussianMixture(n_components = clster)
                    GMM.fit(X)
                    gmmm = GMM.predict(X)
                    df["GMM Categories"] = gmmm
                    pdf = df[["Tickers","GMM Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    sc = silhouette_score(X,gmmm)
                    st.write(f"**Silhouette Score** : {sc}")

                    tables_km = {}

                    for category in df["GMM Categories"].unique():
                        category_df = df[df["GMM Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(gmmm)

                    st.header("Cluster Details")
                    for i in df["GMM Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["GMM Categories"]==i]["Tickers"])

    elif page == "Quantum Machine Learning":
        st.header("Portfolio Classification using Quantum Machine Learning")
        n = st.number_input("Enter the number of tickers : ",min_value=0,step=1)
        strategy = st.selectbox("Select Clustering type", options=[
            "QSVC",
            "VQC"
            ])

        df = load_df()
        close_df = pd.DataFrame()

        if strategy == "QSVC":
            clster = st.number_input("Enter the number of clusters : ",min_value=2,max_value=n)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("QSVC (Quantum Support Vector Classification) Clustering")
                    kmeans = KMeans(n_clusters = clster,random_state=42)
                    kmeans.fit(X)
                    labels = kmeans.labels_

                    feature_map = ZZFeatureMap(feature_dimension =2,reps = 3)
                    kernel = FidelityQuantumKernel(feature_map = feature_map)
                    qsvc = QSVC(quantum_kernel = kernel)
                    qsvc.fit(X,labels)
                    qsr = qsvc.predict(X)
                    
                    df["QSVC Categories"] = qsr
                    pdf = df[["Tickers","QSVC Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    try:
                        sc = silhouette_score(X,qsr)
                        st.write(f"**Silhouette Score** : {sc}")
                    except Exception as e:
                        st.write("No Silhouette Score will be provided for One Cluster")

                    tables_km = {}

                    for category in df["QSVC Categories"].unique():
                        category_df = df[df["QSVC Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(qsr)

                    st.header("Cluster Details")
                    for i in df["QSVC Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["QSVC Categories"]==i]["Tickers"])

        elif strategy == "VQC":
            clster = st.number_input("Enter the number of clusters : ",min_value=2,max_value=n)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            
            if tkr and len(tkr)==n and strategy and clster: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                returns = close_df.pct_change()

                ann_returns = returns.mean()*252
                ann_variance = returns.var()*252

                df = pd.DataFrame(close_df.columns,columns=["Tickers"])
                df['Variances']=ann_variance.values
                df['Returns']=ann_returns.values

                df['Returns'] = pd.to_numeric(df['Returns'])
                df['Variances'] = pd.to_numeric(df['Variances'])
                X = df[['Returns','Variances']].values

                if st.button("Get Clusters"):
                    st.header("VQC (Variational Quantum Classifier) Clustering")
                    kmeans = KMeans(n_clusters = clster,random_state=42)
                    kmeans.fit(X)
                    labels = kmeans.labels_

                    feature_map = ZZFeatureMap(feature_dimension =2,reps = 3)
                    kernel = FidelityQuantumKernel(feature_map = feature_map)
                    vqc = VQC(num_qubits = 2, feature_map = feature_map,optimizer=COBYLA(maxiter=300))
                    vqc.fit(X,labels)
                    vqcr = vqc.predict(X)
                    
                    df["VQC Categories"] = vqcr
                    pdf = df[["Tickers","VQC Categories"]]

                    st.subheader("Cluster Table")
                    st.write(pdf)
                    try:
                        sc = silhouette_score(X,qsr)
                        st.write(f"**Silhouette Score** : {sc}")
                    except Exception as e:
                        st.write("No Silhouette Score will be provided for One Cluster")

                    tables_km = {}

                    for category in df["VQC Categories"].unique():
                        category_df = df[df["VQC Categories"] ==category]
                        K_table = category_df
                        tables_km[category]=K_table

                    def get_Kinfo(n,tkrlst):
                        st.write(f"Mean Returns of Cluster {n} : {tables_km[n]['Returns'].mean()}")
                        st.write(f"Mean Variance of Cluster {n} : {tables_km[n]['Variances'].mean()}\n")
                        st.write(tables_km[n])

                        fig1 = go.Figure()
                        for i in tkrlst:
                            fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))
                        fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
                        st.plotly_chart(fig1)

                    st.subheader("Plots")
                    plot_results(vqcr)

                    st.header("Cluster Details")
                    for i in df["VQC Categories"].unique():
                        with st.expander(f"Cluster {i}",expanded = False):
                            st.subheader(f"Cluster {i}")
                            get_Kinfo(i,df[df["VQC Categories"]==i]["Tickers"])

if __name__ == '__main__':
    main()