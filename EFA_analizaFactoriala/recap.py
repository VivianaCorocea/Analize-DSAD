import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler

rawDiversitate=pd.read_csv('./dateIN/Diversitate.csv', index_col=0)
rawCoduri=pd.read_csv('./dateIN/Coduri_localitati.csv', index_col=0)
labels=list(rawDiversitate.columns.values[1:])

merged=rawDiversitate.merge(rawCoduri,right_index=True, left_index=True).drop('Localitate_y',axis=1).rename(columns={'Localitate_x':'Localitate'})[['Judet', 'Localitate']+labels]
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)

#EFA
#STANDARDIZARE
x=StandardScaler().fit_transform(merged[labels])
bartlett_stat,bartlett_p=calculate_bartlett_sphericity(x)

#TEST BARTLETT
if(bartlett_p>0.001):
    print('nu exista factori comuni semnificativi')
    exit(0)

#TEST KMO
kmo_all,kmo=calculate_kmo(x)
if(kmo<0.6):
    print('valoare kmo prea mica, nu se recomanda analiza factoriala')
    exit(0)

#EFA
efa=FactorAnalyzer(n_factors=x.shape[1]-1, rotation='varimax')

#CALCUL SCORURI
scores=efa.fit_transform(x)
df_scores=pd.DataFrame(scores,columns=['Factor-'+str(i)for i in range(scores.shape[1])], index=merged['Localitate'])
df_scores.to_csv('./dateOUT/EFA_scoruri.csv')

#TRASARE PLOT SCORURI - graficul scorurilor fact pt 2 factori (ex f0 si f1)
plt.figure(figsize=(12,12))
plt.title('Graficul scorurilor pentru primii 2 factori')
plt.scatter(scores[:,0],scores[:,1],color='blue')

