import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from seaborn import heatmap

rawDiversitate=pd.read_csv('./dateIN/Diversitate.csv',index_col=0)
rawCoduri=pd.read_csv('./dateIN/Coduri_Localitati.csv',index_col=0)
labels=list(rawDiversitate.columns.values[1:])

merged=rawDiversitate.merge(rawCoduri,right_index=True,left_index=True)\
    .drop('Localitate_y',axis=1)\
    .rename(columns={'Localitate_x':'Localitate'})[['Judet','Localitate']+labels]
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)
#print(merged)

#EFA
#STANDARDIZARE
x=StandardScaler().fit_transform(merged[labels])

#TEST BARTLETT DE RELEVANTA - ANALIZA FACTORABILITATII
bartlett_stat,bartlett_p=calculate_bartlett_sphericity(x)
if(bartlett_p>0.001):
    print('NU exista factori comuni semnificativi')
    exit(0)

#KMO PT A VERIFICA DACA DATELE SUNT POTRIVITE PT EFA - analiza factorabilitatii
kmo_all,kmo=calculate_kmo(x)
if(kmo<0.6):
    print('Indicele KMO este prea mic. Analiza factoriala NU este recomandata')
    exit(0)

#daca vrem FARA rotatie => rotation=None
efa=FactorAnalyzer(n_factors=x.shape[1]-1,rotation='varimax') #x.shape[1] = nr de coloane
print(x.shape)

#CALCUL SCORURI
scores=efa.fit_transform(x) # antreneaza modelul EFA pe datele x dar si returneaza „scorurile factorilor”
# daca nu ne cere scorurile la vreo cerinta => putem folosi efa.fit() pt a antrena modelul
df_scores=pd.DataFrame(scores,columns=["Factor_"+str(i) for i in range (scores.shape[1])], index=merged['Localitate'])
df_scores.to_csv('./dateOUT/EFA_scoruri.csv')

#TRASARE PLOT SCORURI - graficul scorurilor fact pt 2 factori (ex f0 si f1)
plt.figure(figsize=(12,12))
plt.title('Graficul scorurilor pentru primii 2 factori')
plt.scatter(scores[:,0],scores[:,1],color='blue')
plt.xlabel('factor 1')
plt.ylabel('factor 2')
plt.axhline(0,c='r')
plt.axvline(0,c='r')
for i,ind in enumerate(merged['Localitate']):
    plt.annotate(ind,(scores[i,0], scores[i,1]))
plt.show()

#VARIANTA FACTORI COMUNI (VARIANTA, PROCENTUL DE VARIANTA EXTRASA, PROCENTUL CUMULAT DE VARIANTA)
varianta_fact, proc_var_extrasa, proc_var_cum=efa.get_factor_variance() #tuplu cu cele 3 valori cerute
print(varianta_fact, proc_var_extrasa, proc_var_cum)
df_varianta=pd.DataFrame(data={
    'Varianta factorilor': varianta_fact,
    'Procentul de varianta extrasa':proc_var_extrasa*100,
    'Procentul de varianta cumulat': proc_var_cum*100
}).to_csv('./dateOUT/EFA_Varianta.csv', index=False)

#CORELATII FACTORIALE
factor_loadings=efa.loadings_
df_factor_loadings=pd.DataFrame(factor_loadings,index=labels).to_csv('./dateOUT/EFA_cor_fact.csv')

#CORELOGRAMA CORELATII FACTORIALE
plt.figure(figsize=(12,12))
plt.title('Corelograma corelatiilor factoriale')
heatmap(data=factor_loadings,vmin=-1, vmax=1, cmap='bwr', annot=True)
plt.xlabel('factori')
plt.ylabel('variabile')
# plt.show()
# print(df_scores.var()) ?

#CERCUL CORELATIILOR PT 2 FACTORI
plt.figure(figsize=(8,8))
plt.title("Cercul de corelatie pentru primii 2 factori comuni")
T=[t for t in np.arange(0, np.pi*2, 0.01)]
X=[np.cos(t) for t in T]
Y=[np.sin(t) for t in T]
plt.plot(X,Y)
plt.axhline(0,c='g')
plt.axvline(0,c='g')
plt.scatter(factor_loadings[:,0],factor_loadings[:,1])
for i in range(factor_loadings.shape[0]):
    plt.text(factor_loadings[i,0],factor_loadings[i,1],labels[i],fontsize=12, ha='right')
# plt.show()

#CALCUL COMUNALITATI SI VARIANTA SPECIFICA
communalities=efa.get_communalities() # cat din variabila este explicat de factori
print(communalities)

specificFactors=efa.get_uniquenesses() # cat din varianta variabilei nu e explicat de factori
print(specificFactors)

df_comunalitati_varianta=pd.DataFrame(data={
    'Comunalitati': communalities,
    'Varianta specifica': specificFactors
}, index=labels) # Folosim labels pentru index
df_comunalitati_varianta.to_csv('./dateOUT/EFA_comunalitati_varianta_specifica.csv')

#CORELOGRAMA COMUNALITATI SI VARIANTA SPECIFICA
plt.figure(figsize=(12,12))
plt.title('Corelogramă - Comunalități și varianță specifică')
heatmap(data=df_comunalitati_varianta, cmap='viridis', annot=True)
plt.show()