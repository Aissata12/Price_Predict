#!/usr/bin/env python
# coding: utf-8

# In[194]:


# Étape 1 : Importation des bibliothèques
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[197]:


# 2. Charger les données"
df = pd.read_csv("C:\\Users\\A\\Desktop\\IA MASTER2\\House Price prediction\\house_prices.csv")

df.head()


# In[198]:


# Étape 3 : Prétraitement simple
# Garder les colonnes numériques pour simplifier
#df = df.select_dtypes(include=["int64", "float64"])
#df = df.dropna(axis=1, how="any")  # Supprimer les colonnes avec valeurs manquantes

df.info()
df.describe()
df.isnull().sum()


# In[199]:


df.info()


# In[200]:


df.fillna(df.mean(numeric_only=True), inplace=True)


# In[201]:


print(df.columns.tolist())


# In[202]:


# Sélectionner les colonnes de type object avec peu de modalités (ex: < 100)
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 100]

print("Colonnes catégorielles à encoder :", cat_cols)

# Puis encoder uniquement celles-ci
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# In[166]:


# Séparer variables indépendantes et cible
#X = df.drop("Price (in rupees)", axis=1)
#y = df["Price (in rupees)"]


# In[203]:


print("df shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print(df.head())


# In[204]:


df.fillna(0, inplace=True)  # Replace NaNs with 0

# Now redo X and y
X = df.drop(['Index', 'Price (in rupees)'], axis=1)
y = df['Price (in rupees)']

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[205]:


print(df.shape)
print(X.shape)


# In[206]:


df = df.dropna()  # ou df.dropna()


# In[207]:


if df.empty:
    print("Le DataFrame est vide.")
else:
    print("Le DataFrame contient :", df.shape[0], "lignes")


# In[209]:


# Garder uniquement les colonnes numériques mais pas les lignes vides
df = df.select_dtypes(include=["int64", "float64"])
df = df.dropna()


# In[210]:


print(len(X), len(y))  # Les deux doivent avoir plus que 0 lignes et même taille


# In[211]:


#Affichage de la distribution des prix

st.subheader("Distribution des prix des maisons")
fig, ax = plt.subplots()
sns.histplot(df["Price (in rupees)"], kde=True, ax=ax)
st.pyplot(fig)


# In[212]:


# Define X and y
X = df.drop(['Index', 'Price (in rupees)'], axis=1)
y = df['Price (in rupees)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[213]:


# Étape 5 : Entraînement du modèle
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[214]:


# Étape 7 : Application Streamlit
# Enregistrer le modèle
import joblib
joblib.dump(model, "house_price_model.pkl")


# In[ ]:




