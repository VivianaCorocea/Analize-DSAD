import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

x=pd.read_csv('./dateIN/Pacienti.csv', index_col=0)
x_apply=pd.read_csv('./dateIN/Pacienti_apply.csv', index_col=0)

tinta='DECISION'
labels_lda=list(x.columns.values[:-1])

dict={'I':1, 'S':2, 'A':3}
x[tinta]=x[tinta].map(dict)

x_train,x_test,y_train,y_test=train_test_split(x[labels_lda], x[tinta], train_size=0.4)

#ANTRENAM MODELUL
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)

#SCORURI DISCRIMINANTE
scores=lda.transform(x_train)
pd.DataFrame(scores).to_csv('./dateOUT/LDA_scoruri.csv')

#PLOT INSTANTE SCORURI
plt.figure(figsize=(10,10))
plt.title('Instantele in primele 2 axe discriminante')
plt.scatter(scores[:,0], scores[:,1], c=y_train, cmap='viridis', alpha=0.7)
plt.xlabel('prima axa disc')
plt.ylabel('a doua axa disc')
plt.colorbar(label='clase')
plt.show()

#PLOT DISTRIBUTII IN AXE DISCRIMINANTE
plt.figure(figsize=(10,10))
plt.title('Distributiile in primele 2 axe discriminante')
kdeplot(scores, fill=True)
plt.show()

#PREDICTII SI EVALUARE
############################ MODEL LINIAR ##################

#PREDICTII IN SETUL DE DATE
predict_test=lda.predict(x_test)
pd.DataFrame(data=predict_test).to_csv('./dateOUT/LDA_predict_test.csv')

#PREDICTII IN SETUL DE APLICARE
predict_apply=lda.predict(x_apply)
pd.DataFrame(data=predict_apply).to_csv('./dateOUT/LDA_predict_apply.csv')

#EVALUARE
cm=confusion_matrix(y_test,predict_test)
accuracy=accuracy_score(y_test,predict_test)
class_accuracy=cm.diagonal()/cm.sum(axis=1)
medie_accuracy=np.mean(class_accuracy)

print('Matrice de acuratete: ',cm)
print('Acuratetea: ', accuracy)
print('Acuratetea medie: ',medie_accuracy)

################# MODEL BAYESIAN ##############
x_gaussian=GaussianNB()
x_gaussian.fit(x_train, y_train)

#PREDICTII IN SETUL DE TESTARE
gaussian_predict_test=x_gaussian.predict(x_test)

#PREDICTII IN SETUL DE APLICARE
gaussian_predict_apply=x_gaussian.predict(x_apply)

#EVALUARE
cm_bayesian=confusion_matrix(y_test, gaussian_predict_test)
accuracy_bayesian=accuracy_score(y_test, gaussian_predict_test)
class_accuracy_bayesian=cm_bayesian.diagonal()/cm_bayesian.sum(axis=1)
medie_accuracy_bayesian=np.mean(class_accuracy_bayesian)

print('Matrice de acuratete: ',cm_bayesian)
print('Acuratetea:' , accuracy_bayesian)
print('Acuratetea medie: ',medie_accuracy_bayesian)