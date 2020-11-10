#!/usr/bin/env python
# coding: utf-8

# # EL YOUSFI OMAR
# # La classification de données

# # Plan:
# ##  Introduction
# ##  Importations
# ## 1- Analyse Discriptive
# ## 2- Analyce ACP
# ## 3- Modèle KNN et évaluation les performances de généralisation de chaque modèle
# ## 4- La sélection des hyper-paramètres

# # Introduction

# Cet ensemble de données donne un certain nombre de variables avec une condition cible d’avoir ou non une maladie cardiaque.

# C’est un ensemble de données propre et facile à comprendre. Cependant, le sens de certains des en-têtes de colonne ne sont pas évidents. Voici ce qu’ils signifient: <br/>
# <ul>
#     <li><b>age:</b> L’âge de la personne en années </li><br/>
#     <li><b>sex:</b> 1 = male, 0 = female </li><br/>
#     <li><b>cp:</b> The chest pain experienced(=La douleur ressentie à la poitrine) (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic) </li><br/>
#     <li><b> trestbps:</b> The person's resting blood pressure(= La tension artérielle au repos de la personne)(mm Hg on admission to the hospital) </li><br/>
#     <li><b> chol:</b> The person's cholesterol measurement in mg/dl </li><br/>
#     <li><b> fbs:</b> The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) </li><br/>
#     <li><b> restecg:</b> Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria) </li><br/>
#     <li><b> thalach:</b> The person's maximum heart rate achieved </li><br/>
#     <li><b> exang:</b> Exercise induced angina (1 = yes; 0 = no) </li><br/>
#     <li><b> oldpeak:</b> ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here) </li><br/> 
#     <li><b> slope:</b> the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) <br/>
#     <li><b> ca:</b> The number of major vessels (0-3) </li><br/>
#     <li><b> thal:</b> A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) </li><br/>
#     <li><b> target:</b> Heart disease(maladie cardiaque) (0 = no, 1 = yes)</li></ul><br/>

# # Importation des packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels
from scipy import stats
import scipy.stats as stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier


from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors, metrics,model_selection

from scipy.stats import t, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## Importation de fichier

# In[2]:


heart = pd.read_csv('heart.csv')


# In[3]:


heart.head()


# # 1- Analyse Descriptive

# In[4]:


print('Nombre de lignes est :' , heart.shape[0])
print('Nombre de colonnes est :' , heart.shape[1])


# ## Missing Values

# In[5]:


heart.isnull().sum()


# ## Visualisation de données

# ### Produire des statistiques descriptives qui résument la tendance centrale, la dispersion et la forme de la distribution d’un ensemble de données.

# In[7]:


plt.figure(figsize=(12,10))
sns.heatmap(heart.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=3,cmap=sns.color_palette("Set2"))
plt.title("Sommaire")
plt.show()


# ### Histogramme de chaque colonnes:

# In[8]:


heart[['age']].hist()
plt.show()
heart[['sex']].hist()
plt.show()
heart[['cp']].hist()
plt.show()
heart[['trestbps']].hist()
plt.show()
heart[['chol']].hist()
plt.show()
heart[['fbs']].hist()
plt.show()
heart[['restecg']].hist()
plt.show()
heart[['thalach']].hist()
plt.show()
heart[['exang']].hist()
plt.show()
heart[['oldpeak']].hist()
plt.show()
heart[['slope']].hist()
plt.show()
heart[['ca']].hist()
plt.show()
heart[['thal']].hist()
plt.show()
heart[['target']].hist()
plt.show()


# ### Distribution des variables:

# In[9]:


plt.figure(figsize=(12,6))
plt.subplot(122)
plt.pie(heart["target"].value_counts().values,
        labels=["Pas malade","Malade"],
        autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.subplots_adjust(wspace = .2)
plt.title("Proportion de target variables dans dataset")
plt.show()


# ### Correlation entre les variables:

# In[10]:


correlation = heart.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=True,
            cmap=sns.color_palette("magma"),
            linewidth=2,edgecolor="k")
plt.title("CORRELATION ENTRE LES VARIABLES")
plt.show()


# In[11]:


ax=sns.scatterplot(x="chol",y="target", data=heart)
ax.set(xlabel='chol',ylabel='target')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))


# Il y a des O et des 1, mais il est ici difficile de dire si l'on est plus ou moins malade en fonction
# de l'âge. On voit également qu'une régression linéaire sur un tel nuage de points n'aurait
# aucun sens, car elle nous donnerait des valeurs qui ne seraient quasiment jamais sur O ni 1.

# In[12]:


Y=heart["target"].values                      # la variable a expliqué
X=heart[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']] #Les variables explicatives


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.2, random_state=42)


# ### Data Training

# In[14]:


print(X_train)
print(y_train)


# ### Data Testing

# In[15]:


print(X_test)
print(y_test)


# # 2- Analyse ACP:

# In[16]:


acp = decomposition.PCA(n_components=2)
acp.fit(X)

X_acp = acp.transform(X)# la réduction de dimensionnalité à X
print(" la data avant la réduction de dimensionnalité\n")
print(X)

print("\n la data apres la réduction de dimensionnalité \n")
print(X_acp)


# # 3- Modèle knn et évaluation les performances de généralisation de chaque modèle

# ##  Applique l'Algorithme de KNN

# In[17]:



X_train_acp, X_test_acp, y_train_acp, y_test_acp = train_test_split(X_acp, Y, test_size=.2, random_state=42)

# Phase Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_acp, y_train_acp)
# Test de performance
y_pred_acp = knn.predict(X_test_acp)

print("Le processus KNN est terminé .\n")
print('Evaluation du modele :Précision de data prédit avec la data test :{:.5f}%'.format(accuracy_score(y_test_acp, y_pred_acp)*100))
print('\nEvaluation du modele :Précision de training data :{:.5f}%'.format(knn.score(X_train_acp, y_train_acp)*100))
print('Evaluation du modele :Précision de Testing data => {:.5f}%'.format(knn.score(X_test_acp, y_test_acp)*100))


# In[18]:


# Matrice de confusion
pd.crosstab(y_test_acp, y_pred_acp, rownames=['True'], colnames=['Predicted'], margins=True)


# # 4- La sélection des hyper-paramètres

# In[19]:



param_grid = {'n_neighbors':[1 , 3, 5, 7, 9, 11, 13, 15, 17]}

score = 'accuracy'


# Créer un classifieur KNN avec recherche d'hyperparamètre par validation croisée
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), 
    param_grid,                       
    cv=5,                             
    scoring=score                     
)

clf.fit(X_train_acp, y_train_acp)


print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)


# In[20]:



X_train_acp, X_test_acp, y_train_acp, y_test_acp = train_test_split(X_acp, Y, test_size=.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train_acp, y_train_acp)
y_pred_acp = knn.predict(X_test_acp)
print('Evaluation du modele :Précision de data prévu avec la data test => {:.5f}%'.format(accuracy_score(y_test_acp, y_pred_acp)*100))
print('\nEvaluation du modele :Précision de training data => {:.5f}%'.format(knn.score(X_train_acp, y_train_acp)*100))
print('Evaluation du modele :Précision de Testing data => {:.5f}%'.format(knn.score(X_test_acp, y_test_acp)*100))
print("Le processus KNN est terminé .\n")


# # FIN

# In[ ]:




