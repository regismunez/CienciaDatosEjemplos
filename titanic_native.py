import pandas as pd
import numpy as np
from collections import Counter
import re

print("=" * 60)
print("PREPROCESAMIENTO TITANIC CON PYTHON NATIVO")
print("(Sin Scikit-learn, implementación desde cero)")
print("=" * 60)

# 1. CARGAR DATASET
df = pd.read_csv('titanic.csv')

print("\n1. DATOS ORIGINALES:")
print(f"Forma: {df.shape}")
print(f"Valores faltantes:\n{df.isnull().sum()}\n")

# 2. IMPUTACIÓN MANUAL - TÉCNICA 1
print("=" * 60)
print("TÉCNICA 1: IMPUTACIÓN MANUAL DE VALORES FALTANTES")
print("=" * 60)

# Función para imputar Age con mediana por grupo
def imputar_age_por_grupo(dataframe):
    """Imputar Age con mediana por género y clase"""
    df_copy = dataframe.copy()
    
    for pclass in df_copy['Pclass'].unique():
        for sex in df_copy['Sex'].unique():
            mask = (df_copy['Pclass'] == pclass) & (df_copy['Sex'] == sex)
            ages_grupo = df_copy[mask]['Age'].dropna().values
            
            if len(ages_grupo) > 0:
                mediana = np.median(ages_grupo)
                df_copy.loc[mask & df_copy['Age'].isnull(), 'Age'] = mediana
    
    return df_copy

df = imputar_age_por_grupo(df)
print("✓ Age imputada con mediana por género y clase")

# Imputar Embarked con moda (valor más frecuente)
def obtener_moda(valores):
    """Obtener el valor más frecuente"""
    contador = Counter(valores)
    return contador.most_common(1)[0][0]

embarked_values = df['Embarked'].dropna().values
embarked_moda = obtener_moda(embarked_values)
df['Embarked'].fillna(embarked_moda, inplace=True)
print(f"✓ Embarked imputada con moda: {embarked_moda}")

# Crear HasCabin (variable binaria)
df['HasCabin'] = [1 if pd.notna(x) else 0 for x in df['Cabin']]
print("✓ Variable HasCabin creada")

print(f"\nValores faltantes después de imputación:\n{df.isnull().sum()}\n")

# 3. INGENIERÍA DE CARACTERÍSTICAS MANUAL - TÉCNICA 4
print("=" * 60)
print("TÉCNICA 4: INGENIERÍA DE CARACTERÍSTICAS MANUAL")
print("=" * 60)

# FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("✓ FamilySize creada")

# IsAlone
df['IsAlone'] = [1 if x == 1 else 0 for x in df['FamilySize']]
print("✓ IsAlone creada")

# Extraer Title del nombre manualmente
def extraer_title(nombre):
    """Extraer título de nombre usando regex"""
    match = re.search(r'([A-Za-z]+)\.', nombre)
    if match:
        return match.group(1)
    return 'Unknown'

df['Title'] = [extraer_title(name) for name in df['Name']]

# Agrupar títulos raros
titulos_comunes = ['Mr', 'Miss', 'Mrs', 'Master']
df['Title'] = [t if t in titulos_comunes else 'Rare' for t in df['Title']]
print(f"✓ Title extraído. Valores únicos: {df['Title'].unique()}")

# Extraer Deck de Cabin
def extraer_deck(cabin):
    """Extraer primera letra de camarote"""
    if pd.notna(cabin):
        return cabin[0]
    return 'Unknown'

df['Deck'] = [extraer_deck(c) for c in df['Cabin']]
print(f"✓ Deck extraído. Valores únicos: {sorted(df['Deck'].unique())}")

# 4. CODIFICACIÓN MANUAL - TÉCNICA 2
print("\n" + "=" * 60)
print("TÉCNICA 2: CODIFICACIÓN MANUAL DE VARIABLES")
print("=" * 60)

# Label Encoding para Sex (manualmente)
sex_mapping = {'female': 0, 'male': 1}
df['Sex_encoded'] = [sex_mapping[x] for x in df['Sex']]
print(f"✓ Sex codificada: {sex_mapping}")

# One-Hot Encoding para Embarked (manual)
embarked_unique = df['Embarked'].unique()
for port in embarked_unique:
    df[f'Embarked_{port}'] = [1 if x == port else 0 for x in df['Embarked']]
print(f"✓ Embarked (One-Hot): {[f'Embarked_{p}' for p in embarked_unique]}")

# One-Hot Encoding para Title (manual)
title_unique = df['Title'].unique()
for title in title_unique:
    df[f'Title_{title}'] = [1 if x == title else 0 for x in df['Title']]
print(f"✓ Title (One-Hot): {[f'Title_{t}' for t in title_unique]}")

# One-Hot Encoding para Deck (manual)
deck_unique = df['Deck'].unique()
for deck in deck_unique:
    df[f'Deck_{deck}'] = [1 if x == deck else 0 for x in df['Deck']]
print(f"✓ Deck (One-Hot): {[f'Deck_{d}' for d in deck_unique]}")

# 5. NORMALIZACIÓN MANUAL - TÉCNICA 3
print("\n" + "=" * 60)
print("TÉCNICA 3: NORMALIZACIÓN/ESTANDARIZACIÓN MANUAL")
print("=" * 60)

def estandarizar_zcore(valores):
    """Estandarización Z-score: (x - media) / desv.est"""
    valores_array = np.array(valores)
    media = np.mean(valores_array)
    desv_est = np.std(valores_array)
    
    if desv_est == 0:
        return np.zeros_like(valores_array)
    
    return (valores_array - media) / desv_est

# Aplicar estandarización a variables numéricas
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

for col in numeric_cols:
    df[col] = estandarizar_zcore(df[col].values)

print(f"✓ Estandarización Z-score aplicada a: {numeric_cols}")
print(f"Media después de estandarización:")
for col in numeric_cols:
    print(f"  {col}: {df[col].mean():.6f}")
print(f"Desv.Est después de estandarización:")
for col in numeric_cols:
    print(f"  {col}: {df[col].std():.6f}")

# 6. SELECCIÓN FINAL DE CARACTERÍSTICAS
print("\n" + "=" * 60)
print("CARACTERÍSTICAS FINALES SELECCIONADAS")
print("=" * 60)

# Crear lista de columnas finales
columnas_finales = ['Survived', 'Pclass', 'Age', 'Sex_encoded', 'Fare', 
                    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'HasCabin']

# Agregar columnas codificadas
columnas_finales.extend([col for col in df.columns if col.startswith('Embarked_')])
columnas_finales.extend([col for col in df.columns if col.startswith('Title_')])
columnas_finales.extend([col for col in df.columns if col.startswith('Deck_')])

# Asegurar que existan
columnas_finales = [col for col in columnas_finales if col in df.columns]

df_final = df[columnas_finales].copy()

print(f"\nDataset final preparado:")
print(f"Forma: {df_final.shape}")
print(f"Número de características: {len(columnas_finales) - 1}")  # -1 por Survived
print(f"\nPrimeras filas:\n{df_final.head()}")
print(f"\nÚltimas filas:\n{df_final.tail()}")

# 7. GUARDAR DATASET
df_final.to_csv('titanic_preprocessed_native.csv', index=False)
print(f"\n✓ Dataset procesado guardado: titanic_preprocessed_native.csv")

print("\n" + "=" * 60)
print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 60)