""" Joshua Tran
    ITP-449
    HW12
    This assignment finds the optimal number of clusters using KMeans and checks it against the quality 
"""
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score

def main():
    #Data Wrangling
    file_name = "wineQualityReds.csv"
    wine_df = pd.read_csv(file_name)
    quality = wine_df["quality"]
    wine_df = wine_df.drop(columns = "quality")
    wine_df = wine_df.dropna()
    wine_df = wine_df.drop_duplicates()
    
    #normalize data 
    x = pd.DataFrame(Normalizer().fit_transform(wine_df), columns = wine_df.columns)

    #the amount of clusters we are looking at 
    cluster_range = range(1,10)
    inertia_values = []

    #create KMeans models at different cluster values and set their inertia values  
    for cluster_num in cluster_range:
        model_kMeans = KMeans(n_clusters = cluster_num, random_state=42)
        distances = model_kMeans.fit_transform(x)
        cluster_labels = model_kMeans.labels_
        inertia_values.append(model_kMeans.inertia_)
    #plotting
    fig, ax = plt.subplots(1,1)
    ax.plot(cluster_range, inertia_values)
    ax.set_title("Red Wines: Inertia vs Number of clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    plt.savefig("cluster and inertia") 

    #base the optimal cluster amount off the graph and create the best Kmeans model 
    optimal_clusters = 3
    true_model_kMeans = KMeans(n_clusters = optimal_clusters, random_state=42)
    cluster_labels = true_model_kMeans.fit_predict(x)
    #create a dataframe of results from Cluster labels and quality 
    results = pd.DataFrame({'Cluster': cluster_labels, 'Quality': quality})
    #create crosstab 
    crosstab = pd.crosstab(results['Cluster'], results['Quality'])
    print(crosstab)
    cond_probabilities = crosstab.div(crosstab.sum(axis=1), axis=0)
    print(cond_probabilities)

    #I think these clusters do not represent the quality of the wine since clusters 0, 1 and 2
    #seem to share both majority in the data around the 5-6 quality mark. 

if __name__ == '__main__':
    main()