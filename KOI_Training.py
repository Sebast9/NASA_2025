import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import joblib

KOI_raw_data = pd.read_csv('Data/KOI_data_cumulative_2025.09.22_07.28.01.csv',skiprows=144)

useless_columns_KOI = [
    'kepid', 'rowid', 'koi_datalink_dvr', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
    'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_disp_prov', 'koi_parm_prov', 'koi_sparprov',
    'koi_vet_stat', 'koi_vet_date', 'koi_comment', 'koi_tce_delivname', 'koi_tce_plnt_num', 'ra', 'dec',
    'koi_quarters', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_time0', 'koi_time0_err1',
    'koi_time0_err2', 'koi_limbdark_mod', 'koi_fittype', 'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec',
    'koi_dicco_mdec_err', 'koi_dicco_msky', 'koi_dicco_msky_err', 'koi_dikco_mra', 'koi_dikco_mra_err',
    'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky', 'koi_dikco_msky_err', 'koi_fwm_stat_sig', 'koi_fwm_sra',
    'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err', 'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco',
    'koi_fwm_sdeco_err', 'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err', 'koi_period_err1',
    'koi_period_err2', 'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2', 'koi_longp_err1', 'koi_longp_err2', 'koi_impact_err1',
    'koi_impact_err2', 'koi_duration_err1', 'koi_duration_err2', 'koi_ingress_err1', 'koi_ingress_err2',
    'koi_depth_err1', 'koi_depth_err2', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho_err1', 'koi_srho_err2',
    'koi_prad_err1', 'koi_prad_err2', 'koi_sma_err1', 'koi_sma_err2', 'koi_incl_err1', 'koi_incl_err2', 'koi_teq_err1',
    'koi_teq_err2', 'koi_insol_err1', 'koi_insol_err2', 'koi_dor_err1', 'koi_dor_err2', 'koi_steff_err1',
    'koi_steff_err2', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_smet_err1', 'koi_smet_err2', 'koi_srad_err1',
    'koi_srad_err2', 'koi_smass_err1', 'koi_smass_err2', 'koi_sage_err1', 'koi_sage_err2', 'koi_sage', 'koi_model_dof',
    'koi_ingress', 'koi_model_chisq', 'koi_longp', 'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_trans_mod', 'koi_datalink_dvr',
    'koi_datalink_dvs'
]

df_KOI = KOI_raw_data.drop(columns=useless_columns_KOI)
df_KOI_encoded = pd.get_dummies(df_KOI, columns=['koi_disposition'], dummy_na=False, prefix='state')
imputer = IterativeImputer(max_iter=10, random_state=0)
df_KOI_imputado = pd.DataFrame(imputer.fit_transform(df_KOI_encoded), columns=df_KOI_encoded.columns)
encoded_columns = [c for c in df_KOI_imputado.columns if c.startswith('state_')]
columnas_originales = [c for c in df_KOI_imputado.columns if c not in encoded_columns]
decoded_series = df_KOI_imputado[encoded_columns].idxmax(axis=1)
df_KOI['koi_disposition_imputed'] = decoded_series.str.replace('state_', '')
final_df = df_KOI_imputado[columnas_originales].copy()
final_df['koi_disposition'] = df_KOI['koi_disposition_imputed']

X = final_df.drop('koi_disposition', axis=1)
y_text = final_df['koi_disposition']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

param_grid = {
    'n_estimators': [200, 400, 600, 800, 1000, 1500],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.05, 0.1, 0.2, 0.3]
}

xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss', tree_method='hist', objective='multi:softprob')
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=80,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)
best_xgb_model = random_search.best_estimator_

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])
param_grid_svm = {
    'svm__C': [0.1, 1, 3, 10, 30, 100, 300],
    'svm__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1],
    'svm__kernel': ['rbf']
}
random_search_svm = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions=param_grid_svm,
    n_iter=30,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search_svm.fit(X_train, y_train)
best_svm_model = random_search_svm.best_estimator_

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=2000, early_stopping=True, random_state=42))
])
param_grid_mlp = {
    'mlp__hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64), (128, 128)],
    'mlp__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'mlp__learning_rate_init': [1e-4, 5e-4, 1e-3, 5e-3],
    'mlp__activation': ['relu', 'tanh']
}
random_search_mlp = RandomizedSearchCV(
    estimator=mlp_pipeline,
    param_distributions=param_grid_mlp,
    n_iter=30,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search_mlp.fit(X_train, y_train)
best_mlp_model = random_search_mlp.best_estimator_

model1 = best_xgb_model
model2 = best_svm_model
model3 = best_mlp_model

meta_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
stacking_model = StackingClassifier(
    estimators=[('xgb', model1), ('svm', model2), ('mlp', model3)],
    final_estimator=meta_model,
    stack_method='predict_proba',
    passthrough=False,
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
joblib.dump(stacking_model, 'exoplanet_stacking_model.pkl')

y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

class_names = label_encoder.classes_
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.title('Matriz de Confusión Normalizada')
plt.tight_layout()
plt.show()
