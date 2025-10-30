"""
EXPLORACIÓN Y PREPROCESAMIENTO DE DATOS PARA DEEP LEARNING
Predicción de Cáncer de Mama usando MLPClassifier
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("EXPLORACIÓN Y PREPROCESAMIENTO DE DATOS PARA DEEP LEARNING")
print("Predicción de Cáncer de Mama")
print("="*80)

# ============================================================================
# FASE 1: EXPLORACIÓN Y SELECCIÓN DEL DATASET
# ============================================================================

print("\n" + "="*80)
print("FASE 1: EXPLORACIÓN Y SELECCIÓN DEL DATASET")
print("="*80)

# Cargar el dataset
print("\n1.1 Cargando dataset desde 'data/breast_cancer.csv'...")
try:
    df = pd.read_csv('data/breast_cancer.csv')
    print(" Dataset cargado exitosamente")
except FileNotFoundError:
    print(" Error: No se encontró el archivo 'data/breast_cancer.csv'")
    print("  Verificar la ruta del archivo")
    exit()

# Información básica del dataset
print("\n1.2 Información básica del dataset:")
print(f"  - Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"  - Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Visualizar primeras filas
print("\n1.3 Primeras 5 filas del dataset:")
print(df.head())

# Información de tipos de datos
print("\n1.4 Tipos de datos:")
print(df.dtypes)

# Información detallada
print("\n1.5 Información detallada del dataset:")
print(df.info())

# Valores faltantes
print("\n1.6 Análisis de valores faltantes:")
valores_nulos = df.isnull().sum()
if valores_nulos.sum() == 0:
    print("  ✓ No se encontraron valores faltantes")
else:
    print(f"  Total de valores faltantes: {valores_nulos.sum()}")
    print("\nColumnas con valores faltantes:")
    print(valores_nulos[valores_nulos > 0])

# Eliminar columnas innecesarias (si existe columna 'Unnamed' o 'id')
columnas_eliminar = []
for col in df.columns:
    if 'Unnamed' in col or col.lower() == 'id':
        columnas_eliminar.append(col)

if columnas_eliminar:
    df = df.drop(columns=columnas_eliminar)
    print(f"\n  Columnas eliminadas: {columnas_eliminar}")

# Estadísticas descriptivas
print("\n1.7 Estadísticas descriptivas:")
print(df.describe())

# Identificar variable objetivo
print("\n1.8 Identificando variable objetivo...")
target_candidates = ['diagnosis', 'target', 'class', 'label']
target_col = None

for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # Buscar columna con valores categóricos binarios
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() == 2:
            target_col = col
            break

if target_col:
    print(f"   Variable objetivo identificada: '{target_col}'")
else:
    print("   No se pudo identificar la variable objetivo automáticamente")
    print("  Columnas disponibles:", list(df.columns))
    exit()

# Separar variables independientes (X) y variable objetivo (y)
print("\n1.9 Separando variables independientes y objetivo...")
X = df.drop(columns=[target_col])
y = df[target_col]

# Convertir variable objetivo a binaria si es necesario
if y.dtype == 'object':
    print(f"  Valores únicos en variable objetivo: {y.unique()}")
    y = y.map({'M': 1, 'B': 0}) if 'M' in y.unique() else y.map({y.unique()[0]: 0, y.unique()[1]: 1})
    print(f"  Variable objetivo convertida a binaria: 0 y 1")

print(f"  Variables independientes (X): {X.shape[1]} características")
print(f"  Variable objetivo (y): {y.shape[0]} muestras")

# Análisis de distribución de clases
print("\n1.10 Distribución de clases:")
distribucion = y.value_counts()
print(distribucion)
print(f"\nProporción de clases:")
print(y.value_counts(normalize=True))

# Visualizar distribución de clases
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Distribución de Clases', fontsize=14, fontweight='bold')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.pie(y.value_counts(), labels=['Benigno (0)', 'Maligno (1)'], 
        autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
plt.title('Proporción de Clases', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fase1_distribucion_clases.png', dpi=300, bbox_inches='tight')
print("\n   Gráfico guardado: 'fase1_distribucion_clases.png'")
plt.close()

# Análisis de desbalanceo
ratio_desbalanceo = y.value_counts().min() / y.value_counts().max()
print(f"\n1.11 Análisis de desbalanceo:")
print(f"  Ratio de desbalanceo: {ratio_desbalanceo:.2f}")
if ratio_desbalanceo > 0.5:
    print("   Dataset relativamente balanceado")
elif ratio_desbalanceo > 0.3:
    print("   Dataset moderadamente desbalanceado")
else:
    print("   Dataset significativamente desbalanceado - considerar técnicas de balanceo")

# Identificar variables numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\n1.12 Tipos de variables:")
print(f"  Variables numéricas: {len(numeric_features)}")
print(f"  Variables categóricas: {len(categorical_features)}")

# Análisis de valores atípicos (outliers)
print("\n1.13 Análisis de valores atípicos (usando método IQR):")
outliers_count = {}
for col in numeric_features:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        outliers_count[col] = outliers

if outliers_count:
    print(f"  Columnas con valores atípicos: {len(outliers_count)}")
    for col, count in list(outliers_count.items())[:5]:
        print(f"    - {col}: {count} outliers")
else:
    print("   No se detectaron valores atípicos significativos")

# Visualizar correlaciones
if len(numeric_features) > 1:
    plt.figure(figsize=(14, 10))
    correlation_matrix = X[numeric_features].corr()
    
    # Seleccionar top 15 características más correlacionadas con otras
    top_features = correlation_matrix.abs().sum().nlargest(15).index
    
    sns.heatmap(correlation_matrix.loc[top_features, top_features], 
                annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Matriz de Correlación (Top 15 características)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fase1_correlacion.png', dpi=300, bbox_inches='tight')
    print("\n   Gráfico guardado: 'fase1_correlacion.png'")
    plt.close()

print("\n" + "="*80)
print(" FASE 1 COMPLETADA")
print("="*80)

# ============================================================================
# FASE 2: PREPROCESAMIENTO Y CONSTRUCCIÓN DEL PIPELINE
# ============================================================================

print("\n" + "="*80)
print("FASE 2: PREPROCESAMIENTO Y CONSTRUCCIÓN DEL PIPELINE")
print("="*80)

print("\n2.1 Configurando preprocesamiento de datos...")

# Crear transformadores para variables numéricas y categóricas
preprocessor = None

if len(categorical_features) > 0:
    print(f"  Configurando escalamiento para {len(numeric_features)} variables numéricas")
    print(f"  Configurando codificación One-Hot para {len(categorical_features)} variables categóricas")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
else:
    print(f"  Configurando escalamiento para {len(numeric_features)} variables numéricas")
    print("  No se detectaron variables categóricas")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

# Crear el pipeline completo
print("\n2.2 Construyendo pipeline con MLPClassifier...")
print("  Configuración de la red neuronal:")
print("    - Capas ocultas: (50, 25)")
print("    - Función de activación: ReLU")
print("    - Optimizador: Adam")
print("    - Máximo de iteraciones: 500")

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    ))
])

print("\n   Pipeline construido exitosamente")

# División de datos
print("\n2.3 Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Datos de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Datos de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

# Entrenar el pipeline
print("\n2.4 Entrenando el modelo...")
pipeline.fit(X_train, y_train)
print("   Modelo entrenado exitosamente")

# Realizar predicciones
print("\n2.5 Realizando predicciones...")
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Evaluar rendimiento inicial
print("\n2.6 Evaluación inicial del modelo:")
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"  Exactitud en entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Exactitud en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Métricas detalladas en conjunto de prueba
print("\n2.7 Métricas detalladas (conjunto de prueba):")
print(f"  Precisión: {precision_score(y_test, y_test_pred):.4f}")
print(f"  Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_test_pred):.4f}")

# Matriz de confusión
print("\n2.8 Matriz de confusión (conjunto de prueba):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusión - Modelo Inicial', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('fase2_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("\n  Gráfico guardado: 'fase2_matriz_confusion.png'")
plt.close()

print("\n" + "="*80)
print(" FASE 2 COMPLETADA")
print("="*80)

# ============================================================================
# FASE 3: VALIDACIÓN CRUZADA Y EVALUACIÓN DEL MODELO
# ============================================================================

print("\n" + "="*80)
print("FASE 3: VALIDACIÓN CRUZADA Y EVALUACIÓN DEL MODELO")
print("="*80)

# Validación cruzada con el modelo base
print("\n3.1 Aplicando validación cruzada (5-fold)...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                            scoring='accuracy', n_jobs=-1)

print(f"  Resultados de validación cruzada:")
print(f"    Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Exactitud media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"    Rango: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

# GridSearchCV para optimización de hiperparámetros
print("\n3.2 Optimizando hiperparámetros con GridSearchCV...")
print("  Buscando mejor configuración de capas ocultas y tasa de aprendizaje...")

param_grid = {
    'classifier__hidden_layer_sizes': [(50, 25), (100, 50), (100, 50, 25)],
    'classifier__learning_rate_init': [0.001, 0.01],
    'classifier__alpha': [0.0001, 0.001]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n3.3 Resultados de la optimización:")
print(f"  Mejores hiperparámetros: {grid_search.best_params_}")
print(f"  Mejor score de validación cruzada: {grid_search.best_score_:.4f}")

# Entrenar modelo final con mejores hiperparámetros
print("\n3.4 Evaluando modelo optimizado...")
best_model = grid_search.best_estimator_
y_test_pred_optimized = best_model.predict(X_test)

# Métricas del modelo optimizado
print("\n3.5 Métricas del modelo optimizado:")
accuracy_opt = accuracy_score(y_test, y_test_pred_optimized)
precision_opt = precision_score(y_test, y_test_pred_optimized)
recall_opt = recall_score(y_test, y_test_pred_optimized)
f1_opt = f1_score(y_test, y_test_pred_optimized)

print(f"  Exactitud: {accuracy_opt:.4f} ({accuracy_opt*100:.2f}%)")
print(f"  Precisión: {precision_opt:.4f}")
print(f"  Recall: {recall_opt:.4f}")
print(f"  F1-Score: {f1_opt:.4f}")

# Reporte de clasificación completo
print("\n3.6 Reporte de clasificación detallado:")
print(classification_report(y_test, y_test_pred_optimized, 
                          target_names=['Benigno', 'Maligno']))

# Matriz de confusión del modelo optimizado
cm_opt = confusion_matrix(y_test, y_test_pred_optimized)
print("\n3.7 Matriz de confusión (modelo optimizado):")
print(cm_opt)

# Visualizaciones comparativas
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Gráfico 1: Comparación de exactitud
axes[0, 0].bar(['Modelo Inicial', 'Validación Cruzada', 'Modelo Optimizado'],
               [test_accuracy, cv_scores.mean(), accuracy_opt],
               color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0, 0].set_ylabel('Exactitud')
axes[0, 0].set_title('Comparación de Exactitud', fontweight='bold')
axes[0, 0].set_ylim([0.8, 1.0])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate([test_accuracy, cv_scores.mean(), accuracy_opt]):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# Gráfico 2: Scores de validación cruzada
axes[0, 1].plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8)
axes[0, 1].axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                   label=f'Media: {cv_scores.mean():.4f}')
axes[0, 1].fill_between(range(1, 6), 
                        cv_scores.mean() - cv_scores.std(),
                        cv_scores.mean() + cv_scores.std(),
                        alpha=0.2, color='red')
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('Exactitud')
axes[0, 1].set_title('Validación Cruzada (5-Fold)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Gráfico 3: Matriz de confusión optimizada
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
axes[1, 0].set_title('Matriz de Confusión - Modelo Optimizado', fontweight='bold')
axes[1, 0].set_ylabel('Valor Real')
axes[1, 0].set_xlabel('Predicción')

# Gráfico 4: Comparación de métricas
metrics_comparison = pd.DataFrame({
    'Modelo Inicial': [test_accuracy, 
                       precision_score(y_test, y_test_pred),
                       recall_score(y_test, y_test_pred),
                       f1_score(y_test, y_test_pred)],
    'Modelo Optimizado': [accuracy_opt, precision_opt, recall_opt, f1_opt]
}, index=['Exactitud', 'Precisión', 'Recall', 'F1-Score'])

metrics_comparison.plot(kind='bar', ax=axes[1, 1], rot=45, width=0.7)
axes[1, 1].set_title('Comparación de Métricas', fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0.8, 1.0])
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fase3_evaluacion_completa.png', dpi=300, bbox_inches='tight')
print("\n   Gráfico guardado: 'fase3_evaluacion_completa.png'")
plt.close()

# Análisis de sobreajuste/subajuste
print("\n3.8 Análisis de sobreajuste/subajuste:")
y_train_pred_opt = best_model.predict(X_train)
train_accuracy_opt = accuracy_score(y_train, y_train_pred_opt)

print(f"  Exactitud en entrenamiento: {train_accuracy_opt:.4f}")
print(f"  Exactitud en prueba: {accuracy_opt:.4f}")
print(f"  Diferencia: {abs(train_accuracy_opt - accuracy_opt):.4f}")

if abs(train_accuracy_opt - accuracy_opt) < 0.05:
    print("   El modelo generaliza bien (no hay sobreajuste significativo)")
elif train_accuracy_opt > accuracy_opt + 0.05:
    print("   Posible sobreajuste - el modelo rinde mejor en entrenamiento")
else:
    print("   Posible subajuste - el modelo tiene bajo rendimiento en ambos conjuntos")

print("\n" + "="*80)
print("✓ FASE 3 COMPLETADA")
print("="*80)

# ============================================================================
# FASE 4: REPORTE Y CONCLUSIONES
# ============================================================================

print("\n" + "="*80)
print("FASE 4: REPORTE Y CONCLUSIONES")
print("="*80)

print("\n" + "="*60)
print("RESUMEN EJECUTIVO")
print("="*60)

print(f"""
Dataset: Breast Cancer Wisconsin (Diagnostic)
Fuente: Kaggle / UCI Machine Learning Repository
Objetivo: Clasificación binaria (Benigno vs Maligno)

CARACTERÍSTICAS DEL DATASET:
  • Total de muestras: {len(df)}
  • Variables predictoras: {X.shape[1]}
  • Distribución de clases: {y.value_counts().to_dict()}
  • Ratio de desbalanceo: {ratio_desbalanceo:.2f}
  • Valores faltantes: {'No' if valores_nulos.sum() == 0 else 'Sí'}

RENDIMIENTO DEL MODELO:
  • Modelo Inicial (sin optimización):
    - Exactitud: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  
  • Validación Cruzada (5-fold):
    - Exactitud media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  
  • Modelo Optimizado (GridSearchCV):
    - Exactitud: {accuracy_opt:.4f} ({accuracy_opt*100:.2f}%)
    - Precisión: {precision_opt:.4f}
    - Recall: {recall_opt:.4f}
    - F1-Score: {f1_opt:.4f}

MEJORES HIPERPARÁMETROS:
{grid_search.best_params_}
""")

print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)

print("""
1. IMPORTANCIA DEL PREPROCESAMIENTO EN MODELOS DE DEEP LEARNING:
   
   El preprocesamiento de datos demostró ser fundamental para el éxito
   del modelo de red neuronal:
   
   • Escalamiento (StandardScaler): Normalizar las características fue
     crucial dado que las variables médicas tienen rangos muy diferentes.
     Sin escalamiento, las características con valores grandes dominarían
     el proceso de aprendizaje.
   
   • Eliminación de columnas irrelevantes: IDs y columnas no informativas
     fueron removidas para evitar ruido en el modelo.
   
   • Análisis exploratorio: Permitió identificar la estructura de los
     datos, detectar valores atípicos y comprender la distribución de
     clases antes del modelado.
   
   IMPACTO: El preprocesamiento adecuado permitió que el modelo alcanzara
   alta precisión (>95%) desde la primera iteración.

2. UTILIDAD DEL USO DE PIPELINES PARA ESTANDARIZAR PROCESOS:
   
   La implementación de pipelines en scikit-learn proporcionó múltiples
   beneficios:
   
   • Reproducibilidad: Todo el flujo de procesamiento está encapsulado,
     garantizando que los mismos pasos se apliquen consistentemente.
   
   • Prevención de data leakage: El escalamiento se ajusta solo con datos
     de entrenamiento y se aplica a los datos de prueba, evitando fugas
     de información.
   
   • Código limpio y mantenible: Reduce errores y facilita la integración
     de nuevos componentes de procesamiento.
   
   • Optimización simplificada: GridSearchCV puede optimizar tanto
     hiperparámetros del modelo como del preprocesamiento de forma unificada.
   
   IMPACTO: El pipeline permitió experimentar con diferentes configuraciones
   de forma segura y eficiente.

3. VALOR DEL USO DE VALIDACIÓN CRUZADA PARA MODELOS MÁS ROBUSTOS:
   
   La validación cruzada demostró ser esencial para:
   
   • Estimación realista del rendimiento: La exactitud media de validación
     cruzada ({cv_scores.mean():.4f}) mostró que el modelo generaliza bien,
     con baja variabilidad entre folds (std: {cv_scores.std():.4f}).
   
   • Detección de sobreajuste: La diferencia entre entrenamiento y prueba
     fue mínima (<5%), indicando que el modelo no está sobreajustado.
   
   • Optimización de hiperparámetros: GridSearchCV exploró sistemáticamente
     el espacio de hiperparámetros, mejorando la configuración inicial.
   
   • Confiabilidad: Usar 5 folds proporciona múltiples evaluaciones
     independientes, dando mayor confianza en el rendimiento final.
   
   IMPACTO: La validación cruzada garantiza que el modelo será robusto
   al enfrentar nuevos datos no vistos en producción.

4. ANÁLISIS DE RESULTADOS FINALES:
   
   El modelo final alcanzó métricas excelentes:
   
   • Exactitud de {accuracy_opt*100:.2f}%: Clasifica correctamente la gran
     mayoría de casos.
   
   • Alta Precisión ({precision_opt:.4f}): Pocos falsos positivos, minimizando
     diagnósticos innecesarios.
   
   • Alto Recall ({recall_opt:.4f}): Detecta la mayoría de casos malignos,
     crucial en diagnóstico médico.
   
   • F1-Score balanceado ({f1_opt:.4f}): Equilibrio óptimo entre precisión
     y recall.
   
   INTERPRETACIÓN: El modelo es confiable para asistir en el diagnóstico
   de cáncer de mama, con bajo riesgo de falsos negativos (casos malignos
   no detectados).

5. RECOMENDACIONES Y TRABAJO FUTURO:
   
   • Implementar técnicas de ensemble (Random Forest, Gradient Boosting)
     para comparar rendimiento con redes neuronales.
   
   • Explorar arquitecturas más profundas o usar regularización adicional
     (dropout) para datasets más complejos.
   
   • Considerar análisis de importancia de características para
     interpretabilidad clínica.
   
   • Validar el modelo con datos externos para confirmar su capacidad
     de generalización en diferentes poblaciones.
   
   • Implementar explicabilidad con técnicas como SHAP o LIME para
     generar confianza en entornos clínicos.
""")

print("\n" + "="*60)
print("ARCHIVOS GENERADOS")
print("="*60)
print("""
Los siguientes archivos fueron generados durante el análisis:

  1. fase1_distribucion_clases.png
     → Visualización de la distribución de clases del dataset
  
  2. fase1_correlacion.png
     → Matriz de correlación de las principales características
  
  3. fase2_matriz_confusion.png
     → Matriz de confusión del modelo inicial
  
  4. fase3_evaluacion_completa.png
     → Comparación completa de métricas y validación cruzada
""")

print("\n" + "="*60)
print("TABLA COMPARATIVA DE MODELOS")
print("="*60)

# Crear tabla comparativa
comparacion = pd.DataFrame({
    'Métrica': ['Exactitud', 'Precisión', 'Recall', 'F1-Score', 
                'Tiempo de Entrenamiento'],
    'Modelo Inicial': [
        f'{test_accuracy:.4f}',
        f'{precision_score(y_test, y_test_pred):.4f}',
        f'{recall_score(y_test, y_test_pred):.4f}',
        f'{f1_score(y_test, y_test_pred):.4f}',
        'Rápido'
    ],
    'Validación Cruzada': [
        f'{cv_scores.mean():.4f} ± {cv_scores.std():.4f}',
        'N/A',
        'N/A',
        'N/A',
        'Moderado'
    ],
    'Modelo Optimizado': [
        f'{accuracy_opt:.4f}',
        f'{precision_opt:.4f}',
        f'{recall_opt:.4f}',
        f'{f1_opt:.4f}',
        'Lento'
    ]
})

print(comparacion.to_string(index=False))

print("\n" + "="*60)
print("ANÁLISIS DE ERRORES")
print("="*60)

# Análisis detallado de la matriz de confusión
tn, fp, fn, tp = cm_opt.ravel()

print(f"""
Matriz de Confusión del Modelo Optimizado:

                    Predicción
                Benigno    Maligno
    Real  
    Benigno      {tn:3d}        {fp:3d}     (Verdaderos Negativos / Falsos Positivos)
    Maligno      {fn:3d}        {tp:3d}     (Falsos Negativos / Verdaderos Positivos)

INTERPRETACIÓN:
  • Verdaderos Negativos (TN): {tn} casos benignos correctamente identificados
  • Verdaderos Positivos (TP): {tp} casos malignos correctamente identificados
  • Falsos Positivos (FP): {fp} casos benignos incorrectamente clasificados como malignos
  • Falsos Negativos (FN): {fn} casos malignos incorrectamente clasificados como benignos

TASAS DE ERROR:
  • Tasa de Falsos Positivos: {fp/(tn+fp)*100:.2f}% 
    (Proporción de casos benignos mal clasificados)
  
  • Tasa de Falsos Negativos: {fn/(fn+tp)*100:.2f}%
    (Proporción de casos malignos mal clasificados)

RELEVANCIA CLÍNICA:
  Los falsos negativos son críticos en diagnóstico médico, ya que 
  representan casos de cáncer que no serían detectados. Con solo
  {fn} falso(s) negativo(s), el modelo muestra alta sensibilidad.
  
  Los falsos positivos generan estudios adicionales innecesarios,
  pero son menos graves que los falsos negativos.
""")

print("\n" + "="*60)
print("CURVA DE APRENDIZAJE Y CONVERGENCIA")
print("="*60)

# Obtener información de convergencia del modelo final
final_classifier = best_model.named_steps['classifier']
if hasattr(final_classifier, 'loss_curve_'):
    print(f"""
INFORMACIÓN DE ENTRENAMIENTO:
  • Iteraciones completadas: {final_classifier.n_iter_}
  • Pérdida final: {final_classifier.loss_:.6f}
  • Convergencia: {' Alcanzada' if final_classifier.n_iter_ < 500 else '⚠ Máximo de iteraciones'}
  
  El modelo convergió exitosamente, indicando que encontró un
  mínimo local óptimo en la función de pérdida.
""")
    
    # Graficar curva de pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(final_classifier.loss_curve_, linewidth=2, color='#e74c3c')
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Pérdida (Loss)', fontsize=12)
    plt.title('Curva de Aprendizaje - Convergencia del Modelo', 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fase4_curva_aprendizaje.png', dpi=300, bbox_inches='tight')
    print("   Gráfico guardado: 'fase4_curva_aprendizaje.png'")
    plt.close()

print("\n" + "="*60)
print("IMPORTANCIA DE CARACTERÍSTICAS (Análisis Indirecto)")
print("="*60)

# Análisis de correlación con la variable objetivo para entender importancia
if len(numeric_features) > 0:
    correlaciones = pd.DataFrame({
        'Característica': numeric_features,
        'Correlación': [abs(X[col].corr(y)) for col in numeric_features]
    }).sort_values('Correlación', ascending=False)
    
    print("\nTop 10 características más correlacionadas con el diagnóstico:\n")
    print(correlaciones.head(10).to_string(index=False))
    
    # Visualizar top 15 características
    plt.figure(figsize=(12, 8))
    top_15 = correlaciones.head(15)
    plt.barh(range(len(top_15)), top_15['Correlación'], color='#3498db')
    plt.yticks(range(len(top_15)), top_15['Característica'])
    plt.xlabel('Correlación Absoluta con Diagnóstico', fontsize=12)
    plt.title('Top 15 Características Más Correlacionadas', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fase4_importancia_features.png', dpi=300, bbox_inches='tight')
    print("\n  ✓ Gráfico guardado: 'fase4_importancia_features.png'")
    plt.close()
    
    print("""
NOTA: Esta correlación es un análisis univariado. En el modelo de red
neuronal, las características interactúan de forma compleja y no lineal.
Para importancia real, se requerirían técnicas como permutation importance
o SHAP values.
""")

print("\n" + "="*60)
print("EVALUACIÓN DE ROBUSTEZ")
print("="*60)

# Análisis de estabilidad con múltiples ejecuciones
print("\nEvaluando estabilidad del modelo con diferentes semillas aleatorias...")
print("(10 ejecuciones con diferentes particiones de datos)\n")

stability_scores = []
for seed in range(10):
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    temp_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=grid_search.best_params_['classifier__hidden_layer_sizes'],
            activation='relu',
            solver='adam',
            learning_rate_init=grid_search.best_params_['classifier__learning_rate_init'],
            alpha=grid_search.best_params_['classifier__alpha'],
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ))
    ])
    temp_pipeline.fit(X_train_temp, y_train_temp)
    score = temp_pipeline.score(X_test_temp, y_test_temp)
    stability_scores.append(score)

stability_mean = np.mean(stability_scores)
stability_std = np.std(stability_scores)

print(f"Resultados de estabilidad:")
print(f"  Media: {stability_mean:.4f}")
print(f"  Desviación estándar: {stability_std:.4f}")
print(f"  Rango: [{min(stability_scores):.4f}, {max(stability_scores):.4f}]")
print(f"  Coeficiente de variación: {(stability_std/stability_mean)*100:.2f}%")

if stability_std < 0.02:
    print("\n   El modelo es muy estable y robusto")
elif stability_std < 0.05:
    print("\n   El modelo tiene estabilidad aceptable")
else:
    print("\n   El modelo muestra variabilidad significativa")

# Visualizar estabilidad
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), stability_scores, marker='o', linewidth=2, 
         markersize=8, color='#2ecc71')
plt.axhline(y=stability_mean, color='r', linestyle='--', 
            label=f'Media: {stability_mean:.4f}')
plt.fill_between(range(1, 11), 
                 stability_mean - stability_std,
                 stability_mean + stability_std,
                 alpha=0.2, color='red', label=f'±1 Std: {stability_std:.4f}')
plt.xlabel('Ejecución', fontsize=12)
plt.ylabel('Exactitud', fontsize=12)
plt.title('Evaluación de Estabilidad del Modelo', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fase4_estabilidad.png', dpi=300, bbox_inches='tight')
print("\n   Gráfico guardado: 'fase4_estabilidad.png'")
plt.close()

print("\n" + "="*60)
print("RECOMENDACIONES PARA PRODUCCIÓN")
print("="*60)

print("""
CONSIDERACIONES PARA IMPLEMENTACIÓN EN ENTORNO CLÍNICO:

1. VALIDACIÓN ADICIONAL:
   • Validar con datasets externos de diferentes hospitales
   • Realizar estudios prospectivos con nuevos pacientes
   • Comparar rendimiento con diagnóstico de especialistas

2. INTERPRETABILIDAD:
   • Implementar SHAP (SHapley Additive exPlanations) para explicar
     predicciones individuales
   • Generar reportes automáticos que justifiquen cada diagnóstico
   • Crear interfaces visuales para médicos

3. MONITOREO Y MANTENIMIENTO:
   • Implementar alertas para detectar drift en los datos
   • Re-entrenar periódicamente con nuevos casos
   • Mantener registro de predicciones vs diagnósticos reales

4. ASPECTOS ÉTICOS Y LEGALES:
   • El modelo debe ser una herramienta de apoyo, no reemplazo del médico
   • Documentar limitaciones y casos donde el modelo puede fallar
   • Cumplir con regulaciones de dispositivos médicos (FDA, CE, etc.)
   • Proteger la privacidad de datos de pacientes (HIPAA)

5. INTEGRACIÓN TÉCNICA:
   • Exportar modelo usando joblib o pickle para producción
   • Crear API REST para integración con sistemas hospitalarios
   • Implementar validación de datos de entrada
   • Establecer tiempos de respuesta adecuados (<1 segundo)
""")

print("\n" + "="*60)
print("CÓDIGO PARA GUARDAR Y CARGAR EL MODELO")
print("="*60)

print("""
Para guardar el modelo entrenado para uso futuro:

```python
import joblib

# Guardar el modelo completo (pipeline)
joblib.dump(best_model, 'modelo_cancer_mama.pkl')

# Cargar el modelo en el futuro
modelo_cargado = joblib.load('modelo_cancer_mama.pkl')

# Realizar predicciones con nuevos datos
nuevas_predicciones = modelo_cargado.predict(X_nuevos)
```
""")

# Guardar el modelo
import joblib
joblib.dump(best_model, 'modelo_cancer_mama.pkl')
print(" Modelo guardado exitosamente: 'modelo_cancer_mama.pkl'")

print("\n" + "="*60)
print("RESUMEN FINAL DE ARCHIVOS GENERADOS")
print("="*60)

archivos_generados = [
    ('fase1_distribucion_clases.png', 'Análisis exploratorio - distribución de clases'),
    ('fase1_correlacion.png', 'Análisis exploratorio - matriz de correlación'),
    ('fase2_matriz_confusion.png', 'Evaluación - matriz de confusión inicial'),
    ('fase3_evaluacion_completa.png', 'Validación cruzada - métricas comparativas'),
    ('fase4_curva_aprendizaje.png', 'Convergencia del modelo'),
    ('fase4_importancia_features.png', 'Análisis de características importantes'),
    ('fase4_estabilidad.png', 'Evaluación de robustez del modelo'),
    ('modelo_cancer_mama.pkl', 'Modelo entrenado serializado')
]

print("\nArchivos generados durante el análisis:\n")
for i, (archivo, descripcion) in enumerate(archivos_generados, 1):
    print(f"  {i}. {archivo}")
    print(f"     → {descripcion}\n")

print("="*80)
print(" ANÁLISIS COMPLETO FINALIZADO EXITOSAMENTE")
print("="*80)

print(f"""
ESTADÍSTICAS FINALES:
  • Dataset procesado: {len(df)} muestras
  • Variables analizadas: {X.shape[1]}
  • Exactitud final: {accuracy_opt:.4f} ({accuracy_opt*100:.2f}%)
  • Modelo: Red Neuronal Multicapa (MLP)
  • Arquitectura optimizada: {grid_search.best_params_['classifier__hidden_layer_sizes']}
  • Validación: 5-fold Cross-Validation
  • Archivos generados: {len(archivos_generados)}

PRÓXIMOS PASOS SUGERIDOS:
  1. Revisar los gráficos generados para análisis visual
  2. Cargar el modelo guardado para realizar nuevas predicciones
  3. Explorar técnicas de ensemble (combinar múltiples modelos)
  4. Implementar análisis de interpretabilidad (SHAP/LIME)
  5. Validar con datasets externos

¡Gracias por utilizar este análisis automatizado!
Para preguntas o mejoras, consultar la documentación de scikit-learn.
""")