# CIENCIA DE DATOS: MOD-2 CLASE 1
# Fuente de datos se descargo titanic.csv desde Kaggle  
# autor: regis munez fecha: octubre 25
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. CARGAR DATASET
df = pd.read_csv('titanic.csv')  

print("=" * 60)
print("PREPROCESAMIENTO TITANIC CON SCIKIT-LEARN")
print("=" * 60)

# Ver datos originales
print("\n1. DATOS ORIGINALES:")
print(f"Forma: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nValores faltantes:\n{df.isnull().sum()}")

# 2. IMPUTACIÓN - TÉCNICA 1
print("\n" + "=" * 60)
print("TÉCNICA 1: IMPUTACIÓN DE VALORES FALTANTES")
print("=" * 60)

# Imputar Age con mediana por género y clase
for class_val in df['Pclass'].unique():
    for gender in df['Sex'].unique():
        mask = (df['Pclass'] == class_val) & (df['Sex'] == gender)
        median_age = df[mask]['Age'].median()
        df.loc[mask & df['Age'].isnull(), 'Age'] = median_age

print(f"✓ Age imputada con mediana por género y clase")

# Imputar Embarked con moda
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
print(f"✓ Embarked imputada con moda: {embarked_mode}")

# Crear variable HasCabin (para aprovechar info de Cabin)
df['HasCabin'] = df['Cabin'].notna().astype(int)
print(f"✓ Variable HasCabin creada (0=sin camarote, 1=con camarote)")

print(f"\nValores faltantes después de imputación:\n{df.isnull().sum()}")

# 3. INGENIERÍA DE CARACTERÍSTICAS - TÉCNICA 4
print("\n" + "=" * 60)
print("TÉCNICA 4: INGENIERÍA DE CARACTERÍSTICAS")
print("=" * 60)

# Crear FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("✓ FamilySize = SibSp + Parch + 1")

# Crear IsAlone
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("✓ IsAlone = 1 si está solo, 0 si no")

# Extraer Title del nombre
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.')
rare_titles = df[~df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])]['Title'].unique()
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
print(f"✓ Title extraído de Name: {df['Title'].unique()}")

# Extraer Deck de Cabin
df['Deck'] = df['Cabin'].str[0]
print(f"✓ Deck extraído de Cabin: {sorted(df['Deck'].dropna().unique())}")

# 4. CODIFICACIÓN - TÉCNICA 2
print("\n" + "=" * 60)
print("TÉCNICA 2: CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("=" * 60)

# Label Encoding para Sex
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
print(f"✓ Sex codificada: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# One-Hot Encoding para Embarked
embarked_encoded = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=False)
df = pd.concat([df, embarked_encoded], axis=1)
print(f"✓ Embarked (One-Hot): {embarked_encoded.columns.tolist()}")

# One-Hot Encoding para Title
title_encoded = pd.get_dummies(df['Title'], prefix='Title', drop_first=False)
df = pd.concat([df, title_encoded], axis=1)
print(f"✓ Title (One-Hot): {title_encoded.columns.tolist()}")

# One-Hot Encoding para Deck
deck_encoded = pd.get_dummies(df['Deck'], prefix='Deck', drop_first=False)
df = pd.concat([df, deck_encoded], axis=1)
print(f"✓ Deck (One-Hot): {deck_encoded.columns.tolist()}")

# Label Encoding para Pclass (ya es numérica y ordinal)
print(f"✓ Pclass mantenida como ordinal: 1=Primera, 2=Segunda, 3=Tercera")

# 5. NORMALIZACIÓN - TÉCNICA 3
print("\n" + "=" * 60)
print("TÉCNICA 3: NORMALIZACIÓN/ESTANDARIZACIÓN")
print("=" * 60)

# Variables a normalizar
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

# Estandarización (Z-score)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
print(f"✓ Estandarización Z-score aplicada a: {numeric_features}")
print(f"  Media después: {df_scaled[numeric_features].mean().round(3).to_dict()}")
print(f"  Desv.Est después: {df_scaled[numeric_features].std().round(3).to_dict()}")

# 6. SELECCIÓN DE CARACTERÍSTICAS
print("\n" + "=" * 60)
print("CARACTERÍSTICAS FINALES SELECCIONADAS")
print("=" * 60)

# Columnas útiles para el modelo
features_to_keep = ['Survived', 'Pclass', 'Age', 'Sex_encoded', 'Fare', 'SibSp', 
                    'Parch', 'FamilySize', 'IsAlone', 'HasCabin', 
                    'Embarked_C', 'Embarked_Q', 'Embarked_S',
                    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                    'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G']

# Asegurar que todas las columnas existan
features_to_keep = [f for f in features_to_keep if f in df_scaled.columns]
df_final = df_scaled[features_to_keep].copy()

print(f"\nDataset final preparado:")
print(f"Forma: {df_final.shape}")
print(f"Columnas: {df_final.columns.tolist()}")
print(f"\nÚltimas filas:\n{df_final.tail()}")
print(f"\nEstadísticas:\n{df_final.describe()}")

# 7. GUARDAR DATASET PROCESADO
df_final.to_csv('titanic_preprocessed_sklearn.csv', index=False)
print(f"\n✓ Dataset procesado guardado: titanic_preprocessed_sklearn.csv")

print("\n" + "=" * 60)
print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE DE TAREA 1: CON SCIKIT-LEARN ")
print("=" * 60)
