import Data_Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from fastapi import FastAPI
from scipy.stats import chi2_contingency
from mlxtend.classifier import StackingCVClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import unify_exoplanet_catalogs as uec
from unify_exoplanet_catalogs import unify

# Leemos los datos
KOI_raw_data = pd.read_csv('Data/KOI_data_cumulative_2025.09.22_07.28.01.csv',skiprows=144)
TOI_raw_data = pd.read_csv('Data/TOI_data_2025.09.22_07.28.57.csv', skiprows=90)
K2_raw_data = pd.read_csv('Data/K2_data_k2pandc_2025.09.22_07.31.46.csv',skiprows=298)

# Conservamos las columnas importantes de cada caso:
TOI_columns = ['tfopwg_disp','pl_orbper','pl_trandurh','pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
K2_columns  = ['disposition',   'pl_orbper','pl_trandur', 'pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
KOI_columns = ['koi_disposition','koi_period','koi_duration','koi_depth','koi_prad','koi_teq','koi_insol','koi_steff','koi_slogg','koi_srad','ra','dec','koi_time0']

TOI_df = TOI_raw_data[TOI_columns]
KOI_df = KOI_raw_data[KOI_columns]
K2_df = K2_raw_data[K2_columns]

# Eliminamos filas inecesarias
K2_useless_values = ['REFUTED']
TOI_useless_values = ['KP','FA','APC']

TOI_values = TOI_df[~TOI_df['tfopwg_disp'].isin(TOI_useless_values)]
K2_values = K2_df[~K2_df['disposition'].isin(K2_useless_values)]

# Convertimos los valores a formato universal
name_values = {'CP':'CONFIRMED','PC':'CANDIDATE','FP':'FALSE POSITIVE'}
TOI_values.loc[:, 'tfopwg_disp'] = TOI_values['tfopwg_disp'].replace(name_values)

koi_std = unify(KOI_df,None,None,priority=('koi','toi','k2'))
toi_std = unify(None,TOI_values, None,priority=('toi','koi','k2'))
k2_std  = unify(None,None,K2_values,priority=('k2','toi','koi'))

# Concatenamos los datos
df_concat = pd.concat([koi_std, toi_std, k2_std], ignore_index=True, sort=False)

original_columns = df_concat.columns
df_encoded = pd.get_dummies(df_concat, columns=['disposition'], dummy_na=False, prefix='state')
df_encoded



imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
print(df_imputed)

encoded_columns = [col for col in df_imputed.columns if col.startswith('state_')]

columnas_originales = [col for col in df_imputed.columns if col not in encoded_columns]

decoded_series = df_imputed[encoded_columns].idxmax(axis=1)

df_concat['diposition_imputed'] = decoded_series.str.replace('state_', '')

final_df = df_imputed[columnas_originales].copy()
final_df['disposition'] = df_concat['diposition_imputed']

# 4. AJUSTE DE HIPERPARÁMETROS PARA XGBOOST

X = final_df.drop('disposition', axis=1)
y_text = final_df['disposition']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("Mapeo de etiquetas (El índice es el número, el valor es la etiqueta):")
print(list(label_encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)
print(f"Datos listos. Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")
# --- Paso 1: Definir el espacio de búsqueda de parámetros ---
# Aquí definimos los rangos de valores que queremos probar para cada hiperparámetro.
param_grid = {
    'n_estimators': [100, 200, 300, 400], # Número de árboles en el bosque
    'max_depth': [3, 4, 5, 6, 7],         # Profundidad máxima de cada árbol
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
    'subsample': [0.7, 0.8, 0.9, 1.0],      # Porcentaje de muestras usadas por árbol
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Porcentaje de características usadas por árbol
    'gamma': [0, 0.1, 0.2]                  # Parámetro de regularización
}

# --- Paso 2: Configurar la Búsqueda Aleatoria ---
# Creamos el modelo base de XGBoost que vamos a optimizar
xgb_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Configuramos RandomizedSearchCV
# n_iter: Número de combinaciones aleatorias a probar. 50 es un buen punto de partida.
# cv: Número de pliegues para la validación cruzada. 5 es estándar.
# scoring: La métrica que queremos maximizar. 'f1_weighted' es ideal para tus datos.
# n_jobs: -1 para usar todos los núcleos de tu CPU y acelerar el proceso.
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1  # Muestra el progreso
)

# --- Paso 3: Ejecutar la Búsqueda ---

print("\nIniciando búsqueda de hiperparámetros para XGBoost...")
random_search.fit(X_train, y_train)

# --- Paso 4: Obtener los Mejores Parámetros ---
print("\nBúsqueda completada.")
print("Mejores parámetros encontrados:")
print(random_search.best_params_)

# Guardamos el mejor modelo encontrado
best_xgb_model = random_search.best_estimator_

# OPTIMIZACIÓN DE HIPERPARÁMETROS PARA SVM

# --- Paso 1: Definir el Pipeline y el Espacio de Búsqueda ---
# Es CRUCIAL escalar los datos para el SVM, por eso lo metemos en un Pipeline.
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# Definimos el espacio de búsqueda. 'svm__' es el prefijo para acceder a los parámetros del paso 'svm' en el pipeline.
param_grid_svm = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svm__kernel': ['rbf'] # El kernel radial (rbf) es el más común y potente
}

# --- Paso 2: Configurar y Ejecutar la Búsqueda Aleatoria ---
random_search_svm = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions=param_grid_svm,
    n_iter=20,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nIniciando búsqueda de hiperparámetros para SVM...")
random_search_svm.fit(X_train, y_train)

# --- Paso 3: Obtener el Mejor Modelo SVM ---
print("\nBúsqueda completada para SVM.")
print("Mejores parámetros encontrados:")
print(random_search_svm.best_params_)
best_svm_model = random_search_svm.best_estimator_

from sklearn.neural_network import MLPClassifier

# OPTIMIZACIÓN DE HIPERPARÁMETROS PARA MLP

# --- Paso 1: Definir el Pipeline y el Espacio de Búsqueda ---
# La Red Neuronal también necesita datos escalados.
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1500, random_state=42)) # Aumentamos las iteraciones
])

# Definimos arquitecturas de red y parámetros de aprendizaje a probar.
param_grid_mlp = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__activation': ['relu', 'tanh']
}

# --- Paso 2: Configurar y Ejecutar la Búsqueda Aleatoria ---
random_search_mlp = RandomizedSearchCV(
    estimator=mlp_pipeline,
    param_distributions=param_grid_mlp,
    n_iter=20, # Probaremos 20 arquitecturas y configuraciones
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nIniciando búsqueda de hiperparámetros para MLP...")
random_search_mlp.fit(X_train, y_train)

# --- Paso 3: Obtener el Mejor Modelo MLP ---
print("\nBúsqueda completada para MLP.")
print("Mejores parámetros encontrados:")
print(random_search_mlp.best_params_)
best_mlp_model = random_search_mlp.best_estimator_

X = final_df.drop('disposition', axis=1)
y_text = final_df['disposition']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("Mapeo de etiquetas (El índice es el número, el valor es la etiqueta):")
print(list(label_encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)
print(f"Datos listos. Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")

# 4. DEFINICIÓN Y CREACIÓN DEL ENSAMBLE DE MODELOS

# Modelo 1: XGBoost
model1 = best_xgb_model

# Modelo 2: SVM con escalado de datos
model2 = best_svm_model

# Modelo 3: Red Neuronal con escalado de datos
model3 = best_mlp_model

# --- Meta-Modelo (Capa 1) ---
meta_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# --- Creación del Ensamble con StackingCVClassifier ---
stacking_model = StackingCVClassifier(
    classifiers=[model1, model2, model3],
    meta_classifier=meta_model,
    use_probas=True,
    cv=5,
    random_state=42,
    n_jobs=-1 # Usar todos los núcleos de CPU disponibles
)
print("Ensamble de modelos definido.")

# 5. ENTRENAMIENTO DEL MODELO
print("\nIniciando entrenamiento del ensamble de stacking...")
stacking_model.fit(X_train, y_train)
print("Entrenamiento completado.")

# 6. GUARDADO DEL MODELO ENTRENADO
model_filename = 'exoplanet_stacking_model.pkl'
joblib.dump(stacking_model, model_filename)
print(f"Modelo guardado en el archivo: {model_filename}")

# 7. EVALUACIÓN DEL MODELO
#print("\nEvaluando el modelo en el conjunto de prueba...")
y_pred = stacking_model.predict(X_test)

# Calcular y mostrar métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión (Accuracy): {accuracy:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión (Conteos Absolutos):")
print(confusion_matrix(y_test, y_pred))

class_names = label_encoder.classes_ 

# 2. Calcula la matriz de confusión normalizada
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

# 3. Crea la figura y el mapa de calor
plt.figure(figsize=(10, 8)) 
heatmap = sns.heatmap(
    cm_normalized, 
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=class_names, 
    yticklabels=class_names
)

plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.title('Matriz de Confusión Normalizada')
plt.show()