import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from seaborn import histplot

rawDiversitate=pd.read_csv('./dateIN/Diversitate.csv',index_col=0)
rawCoduri=pd.read_csv('./dateIN/Coduri_Localitati.csv',index_col=0)

labels=list(rawDiversitate.columns.values[1:])

merged=rawDiversitate.merge(rawCoduri,left_index=True,right_index=True)\
    .drop('Localitate_y',axis=1)\
    .rename(columns={'Localitate_x':'Localitate'})[['Judet','Localitate'] + labels]
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)

#CALCUL IERARHIE
x=StandardScaler().fit_transform(merged[labels])
HC=linkage(x,method='ward')

#CALCUL PARTITIE OPTIMALA
n=HC.shape[0]
print(n)
dist_1=HC[1:n,2]
dist_2=HC[0:n-1,2]
diff=dist_1-dist_2
j=np.argmax(diff)
t=(HC[j,2]+HC[j+1,2])/2
cat=fcluster(HC,n-j,criterion='maxclust')
labels_clusters=['C'+str(i)for i in cat]

#CALCUL PARTITIE OARECARE
kmeans=KMeans(n_clusters=5,n_init=10)
k_labels=['C'+str(i+1)for i in kmeans.fit_predict(x)]
merged['Clusters']=k_labels

#CALCUL INDECSI SILHOUETTE LA NIVEL DE PARTITIE SI DE INSTANTE
silhouette_opt=silhouette_score(x, cat)

#TRASARE PLOT DENDOGRAMACU EVIDENTIEREA PARTITIEI PRIN CULOARE
#PARTITIE OPTIMALA
plt.figure(figsize=(12,12))
plt.title('dendograma')
dendrogram(HC, leaf_rotation=30, labels=merged.index.values)
plt.axhline(t,c='r')
plt.show()

#PARTITIE-K
k=n-j
plt.figure(figsize=(12,12))
plt.title('dendograma')
dendrogram(HC, leaf_rotation=30, labels=merged.index.values)
plt.axhline(HC[-(k-1),2], c='r')
plt.show()

#TRASARE PLOT SILHOUETTE PARTITIE OPTIMALA SI PARTITIE-K
plt.figure(figsize=(12,12))
plt.title('Silhouette partitie optima')
silhouette_opt_obs=silhouette_samples(x, cat)
plt.scatter(x[:,0], x[:,1], c=silhouette_opt_obs, cmap='viridis')
plt.show()

plt.figure(figsize=(12,12))
plt.title('Silhouette partitie 5 clusteri')
silhouette_fixed_obs=silhouette_samples(x,kmeans.fit_predict(x))
plt.scatter(x[:,0], x[:,1], c=silhouette_fixed_obs, cmap='viridis')
plt.show()

#TRASARE HISTOGRAME CLUSTERI PENTRU FIECARE VARIABILA OBSERVATA
variabila='2015'

plt.figure(figsize=(12,12))
plt.title('histograma pentru o variabila')
histplot(data=merged, x=variabila, hue='Clusters', kde=True, bins=30)
plt.show()

#TRASARE PLOT PARTITIE IN AXE PRINCIPALE
#PARTITIE OPTIMA
pca=PCA(n_components=2)
pca_scores=pca.fit_transform(x)

plt.figure(figsize=(12,12))
plt.title("Reprezentare 2D a Clusterelor folosind PCA")
plt.scatter(pca_scores[:,0], pca_scores[:,1], c=cat, cmap='viridis')
plt.colorbar(label='etichete clustere')
plt.show()

#K-PARTITII
pca=PCA(n_components=2)
C=pca.fit_transform(x)
kmeans=KMeans(n_clusters=5, n_init=10)
labels_kmeans=kmeans.fit_predict(C)

plt.figure(figsize=(12,12))
plt.title("Partitie a 5 clusteri pe cele 2 componente principale")
plt.scatter(C[:,0], C[:,1], c=labels_kmeans, cmap='viridis')
plt.colorbar(label='Etichete Clustere')
plt.show()







