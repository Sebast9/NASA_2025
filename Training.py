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

KOI_raw_data = pd.read_csv('Data/KOI_data_cumulative_2025.09.22_07.28.01.csv',skiprows=144)
TOI_raw_data = pd.read_csv('Data/TOI_data_2025.09.22_07.28.57.csv', skiprows=90)
K2_raw_data = pd.read_csv('Data/K2_data_k2pandc_2025.09.22_07.31.46.csv',skiprows=298)

useless_columns_KOI = [
    'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
    'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_disp_prov', 'koi_parm_prov', 'koi_sparprov',
    'koi_vet_stat', 'koi_vet_date', 'koi_comment', 'koi_tce_delivname', 'koi_tce_plnt_num', 'ra', 'dec',
    'koi_quarters', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_time0', 'koi_time0_err1',
    'koi_time0_err2', 'koi_limbdark_mod', 'koi_fittype', 'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec',
    'koi_dicco_mdec_err', 'koi_dicco_msky', 'koi_dicco_msky_err', 'koi_dikco_mra', 'koi_dikco_mra_err',
    'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky', 'koi_dikco_msky_err', 'koi_fwm_stat_sig', 'koi_fwm_sra',
    'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err', 'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco',
    'koi_fwm_sdeco_err', 'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err', 'koi_period_err1',
    'koi_period_err2', 'koi_eccen_err1', 'koi_eccen_err2', 'koi_longp_err1', 'koi_longp_err2', 'koi_impact_err1',
    'koi_impact_err2', 'koi_duration_err1', 'koi_duration_err2', 'koi_ingress_err1', 'koi_ingress_err2',
    'koi_depth_err1', 'koi_depth_err2', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho_err1', 'koi_srho_err2',
    'koi_prad_err1', 'koi_prad_err2', 'koi_sma_err1', 'koi_sma_err2', 'koi_incl_err1', 'koi_incl_err2', 'koi_teq_err1',
    'koi_teq_err2', 'koi_insol_err1', 'koi_insol_err2', 'koi_dor_err1', 'koi_dor_err2', 'koi_steff_err1',
    'koi_steff_err2', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_smet_err1', 'koi_smet_err2', 'koi_srad_err1',
    'koi_srad_err2', 'koi_smass_err1', 'koi_smass_err2', 'koi_sage_err1', 'koi_sage_err2',
]
useless_columns_TOI = ['toi','raerr1','raerr2','decerr1','decerr2','st_pmralim','st_pmrasymerr','st_pmdeclim','st_pmdecsymerr','pl_tranmidlim','pl_tranmidsymerr','pl_orbperlim','pl_orbpersymerr','pl_trandurhlim','pl_trandurhsymerr','pl_trandeplim','pl_trandepsymerr','pl_radelim','pl_radesymerr','pl_insolerr1','pl_insolerr2',
                   'pl_insollim','pl_insolsymerr','pl_eqterr1','pl_eqterr2','pl_eqtlim','pl_eqtsymerr','st_tmaglim','st_tmagsymerr', 'st_distlim','st_distsymerr','st_tefflim','st_teffsymerr','st_logglim','st_loggsymerr','st_radlim','st_radsymerr','toi_created','rowupdate','rowid','pl_pnum','st_pmraerr2','st_pmdecerr2',
                   'pl_tranmiderr2','pl_orbpererr2','pl_trandurherr2','pl_trandeperr2','pl_radeerr2','st_tmagerr2','st_disterr2','st_tefferr2','st_loggerr2','st_raderr2','rastr','decstr']

useless_columns_K2 = ['disp_refname','discoverymethod','disc_year','disc_refname','disc_pubdate','disc_locale','disc_facility','disc_telescope','disc_instrument','pul_flag','ptv_flag','tran_flag','ast_flag','obm_flag','micro_flag','etv_flag','ima_flag','dkin_flag','soltype','pl_controv_flag','pl_refname','pl_orbperlim','pl_orbsmax',
                      'pl_orbsmaxerr1','pl_orbsmaxerr2','pl_orbsmaxlim','pl_radelim','pl_radjlim','pl_masse','pl_masseerr1','pl_masseerr2','pl_masselim','pl_massj','pl_massjerr1','pl_massjerr2','pl_massjlim','pl_msinie','pl_msinieerr1','pl_msinieerr2','pl_msinielim','pl_msinij','pl_msinijerr1','pl_msinijerr2','pl_msinijlim',
                      'pl_cmasse','pl_cmasseerr1','pl_cmasseerr2','pl_cmasselim','pl_cmassj','pl_cmassjerr1','pl_cmassjerr2','pl_cmassjlim','pl_bmasse','pl_bmasse','pl_bmasseerr1','pl_bmasseerr2','pl_bmasselim','pl_bmassj','pl_bmassjerr1','pl_bmassjerr2','pl_bmassjlim','pl_bmassprov','pl_dens','pl_denserr1','pl_denserr2','pl_denslim',
                      'pl_orbeccen','pl_orbeccenerr1','pl_orbeccenerr2','pl_orbeccenlim','pl_insol','pl_insolerr1','pl_insolerr2','pl_insollim','pl_eqt','pl_eqterr1','pl_eqterr2','pl_eqtlim','pl_orbincl','pl_orbinclerr1','pl_orbinclerr2','pl_orbincllim','pl_tranmidlim','pl_tsystemref','ttv_flag','pl_imppar','pl_impparerr1',
                      'pl_impparerr2','pl_impparlim','pl_trandep','pl_trandeperr1','pl_trandeperr2','pl_trandeplim','pl_trandurerr1','pl_trandurerr2','pl_trandurlim','pl_ratdor','pl_ratdorerr1','pl_ratdorerr2','pl_ratdorlim','pl_ratrorerr1','pl_ratrorerr2','pl_ratrorlim','pl_occdep','pl_occdeperr1','pl_occdeperr2','pl_occdeplim',
                      'pl_orbtper','pl_orbtpererr1','pl_orbtpererr2','pl_orbtperlim','pl_orblper','pl_orblpererr1','pl_orblpererr2','pl_orblperlim','pl_rvamp','pl_rvamperr1','pl_rvamperr2','pl_rvamplim','pl_projobliq','pl_projobliqerr1','pl_projobliqerr2','pl_projobliqlim','pl_trueobliq','pl_trueobliqerr1','pl_trueobliqerr2',
                      'pl_trueobliqlim','st_refname','st_spectype','st_tefferr1','st_tefferr2','st_tefflim','st_radlim','st_mass','st_masserr1','st_masserr2','st_masslim','st_met','st_meterr1','st_meterr2','st_metlim','st_metratio','st_lum','st_lumerr1','st_lumerr2','st_lumlim','st_loggerr1','st_loggerr2','st_logglim','st_age',
                      'st_ageerr1','st_ageerr2','st_agelim','st_dens','st_denserr1','st_denserr2','st_denslim','st_vsin','st_vsinerr1','st_vsinerr2','st_vsinlim','st_rotp','st_rotperr1','st_rotperr2','st_rotplim','st_radv','st_radverr1','st_radverr2','st_radvlim','sy_refname','sy_pmerr2','sy_pmraerr2','sy_pmdecerr2','sy_disterr2',
                      'sy_plxerr2','sy_bmagerr2','sy_vmagerr2','sy_jmagerr2','sy_hmagerr2','sy_kmagerr2','sy_umag','sy_umagerr1','sy_umagerr2','sy_gmag','sy_gmagerr1','sy_gmagerr2','sy_rmag','sy_rmagerr1','sy_rmagerr2','sy_imag','sy_imagerr1','sy_imagerr2','sy_zmag','sy_zmagerr1','sy_zmagerr2','sy_w1magerr2','sy_w2magerr2',
                      'sy_w3magerr2','sy_w4magerr1','sy_w4magerr2','sy_gaiamagerr2','sy_icmag','sy_icmagerr1','sy_icmagerr2','sy_tmagerr2','sy_kepmagerr1','sy_kepmagerr2','rowupdate','pl_pubdate','releasedate','pl_nnotes','k2_campaigns','k2_campaigns_num','st_nphot','st_nrvc','st_nspec','pl_nespec','pl_ntranspec','pl_ndispec']

df_KOI = KOI_raw_data.drop(columns=useless_columns_KOI)
df_TOI = TOI_raw_data.drop(columns=useless_columns_TOI)
df_K2 = K2_raw_data.drop(columns=useless_columns_K2)

original_columns = df_TOI.columns
df_TOI_encoded = pd.get_dummies(df_TOI, columns=['tfopwg_disp'], dummy_na=False, prefix='state')
df_TOI_encoded


imputer = IterativeImputer(max_iter=10, random_state=0)
df_TOI_imputado = pd.DataFrame(imputer.fit_transform(df_TOI_encoded), columns=df_TOI_encoded.columns)
print(df_TOI_imputado)

encoded_columns = [col for col in df_TOI_imputado.columns if col.startswith('state_')]

columnas_originales = [col for col in df_TOI_imputado.columns if col not in encoded_columns]

decoded_series = df_TOI_imputado[encoded_columns].idxmax(axis=1)

df_TOI['tfopwg_disp_imputed'] = decoded_series.str.replace('state_', '')

final_df = df_TOI_imputado[columnas_originales].copy()
final_df['tfopwg_disp'] = df_TOI['tfopwg_disp_imputed']

final_df

# 4. AJUSTE DE HIPERPARÁMETROS PARA XGBOOST

X = final_df.drop('tfopwg_disp', axis=1)
y_text = final_df['tfopwg_disp']

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

X = final_df.drop('tfopwg_disp', axis=1)
y_text = final_df['tfopwg_disp']

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