import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from seaborn import heatmap

rawRata=pd.read_csv('./dateIN/Rata.csv', index_col=0)
rawCoduri=pd.read_csv('./dateIN/CoduriTariExtins.csv', index_col=0)
labels=list(rawRata.columns.values[1:])

merged=rawRata.merge(rawCoduri, left_index=True, right_index=True).drop('Country_Name',axis=1)[['Continent','Country']+labels]
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)

#STANDARDIZARE
x=StandardScaler().fit_transform(merged[labels])

#PCA
pca=PCA()
C=pca.fit_transform(x)

#TABELUL VARIANTEI / VARIANTA COMPONENTE
alpha=pca.explained_variance_
pve=pca.explained_variance_ratio_
alpha_cum=np.cumsum(alpha)
pve_cum=np.cumsum(pve)

df_varianta_cimponente=pd.DataFrame(data={
    'varianta componentelor':alpha,
    'procentul de varianta explicata': pve,
    'varianta cumulata': alpha_cum,
    'procentul cumulat': pve_cum
})
df_varianta_cimponente.to_csv('./dateOUT/PCA_varianta_componente', index=False)

#PLOT VARIANTA COMPONENTE CU EVIDENTA CRITERIILOR DE RELEVANTA
plt.figure(figsize=(12,12))
plt.title('Plot varianta componente')
Xindex=['C'+str(k+1)for k in range(len(alpha))]
plt.plot(Xindex, alpha, ' bo-')

#CRITERIUL KAISER
plt.axhline(1,c='r', label='kaiser')

#CRITERIUL CATTELL
eps=np.diff(alpha)
d=np.diff(eps)
if(d<0).any():
    j_Cattel=np.where(d<0)[0][0]+2
    plt.axhline(alpha[j_Cattel-1], c='m', label='cattell')

#CRITERIUL PROCENT MINIMAL
procent_cumulat=np.cumsum(alpha)*100/np.sum(alpha)
j_procent_minimal=np.where(procent_cumulat>80)[0][0]+1
plt.axhline(alpha[j_procent_minimal-1], c='c', label='procent minimal > 80%')

plt.legend()
plt.show()

#CALCUL CORELATII FACTORIALE (CORELATII VARIABILE OBSERVANTE - COMPONENTE)
a=pca.components_.T
rxc=a*np.sqrt(alpha)

#TRASARE CORELOGRAMA CORELATII FACTORIALE
plt.figure(figsize=(12,12))
plt.title('corelograma')
rxc_df=pd.DataFrame(data=rxc, index=labels, columns=['C'+str(i+1)for i in range(rxc.shape[1])])
heatmap(rxc_df, vmin=-1, vmax=1, cmap='bwr', annot=True)
plt.show()

#TRASARE CERCUL CORELATII (PRIMELE 2 COMPONENTE PRINCIPALE)
plt.figure(figsize=(12,12))
plt.title('correlation circle')
T=[t for t in np.arange(0, np.pi*2, 0.01)]
X=[np.cos(t)for t in T]
Y=[np.sin(t)for t in T]
plt.plot(X,Y)
plt.axhline(0, c='r')
plt.axvline(0, c='r')
plt.scatter(rxc[:,0], rxc[:,1])
for i in range(rxc.shape[0]):
    plt.text(rxc[i,0], rxc[i,1], labels[i], fontsize=12, ha='right')
plt.show()

#CALCUL COMPONENTE SI/SAU SCORURI
components=pca.components_
scores=C/np.sqrt(alpha)

#TRASARE PLOT COMPONENTE SCORURI
plt.figure(figsize=(12,12))
plt.title('plot scoruri')
plt.scatter(scores[:,0], scores[:,1])
plt.xlabel('componenta principala 1')
plt.ylabel('componenta principala 2')
plt.show()

######################################
C2=C*C

#CALCUL COSINUSURI
quality=np.transpose(C2.T/np.sum(C2,axis=1))

#CALCUL CONTRIBUTII
contributions=C2/(x.shape[0]*alpha)

#CALCUL COMUNALITATI
communalities=np.cumsum(rxc*rxc, axis=1)

#TRASARE CORELOGRAMA COMUNALITATI
communalities_df=pd.DataFrame(data=communalities, index=labels, columns=['C'+str(i+1)for i in range(communalities.shape[1])])
communalities_df.to_csv('./dateOUT/PCA_comunalitati.csv')

plt.figure(figsize=(12,12))
plt.title('corelograma')
heatmap(communalities_df, vmin=-1, vmax=1, cmap='veridis', annot=True)
plt.show()



