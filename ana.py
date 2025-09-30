#Para el correcto funcionamiento se necesita descargar recomendablemente desde la terminal:
#  python -m pip install -U pip setuptools wheel
#  python -m pip install -U scikit-learn

#   python -m pip install -U pip setuptools wheel
#   python -m pip install -U numpy pandas scipy scikit-learn matplotlib seaborn



import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
KOI_raw_data = pd.read_csv('Data/KOI_data_cumulative_2025.09.22_07.28.01.csv',skiprows=144)

useless_columns_KOI = [
    'kepid', 'koi_datalink_dvr', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
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

useful_columns_KOI = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_model_snr', 'koi_impact', 'koi_ror', 'koi_prad', 'koi_sma',
    'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smet', 'koi_smass', 'koi_srho', 'koi_kepmag',
    'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag', 'koi_kmag', 'koi_incl', 'koi_dor','koi_disposition'
]

df_KOI = KOI_raw_data.drop(columns=useless_columns_KOI)


# Convert into dummies the categorical target
original_columns = df_KOI.columns
df_KOI_encoded = pd.get_dummies(df_KOI, columns=['koi_disposition'], dummy_na=False, prefix='state')

# Impute the missing Data

imputer = IterativeImputer(max_iter=10, random_state=0)
df_KOI_imputado = pd.DataFrame(imputer.fit_transform(df_KOI_encoded), columns=df_KOI_encoded.columns)
print(df_KOI_imputado)

encoded_columns = [col for col in df_KOI_imputado.columns if col.startswith('state_')]

columnas_originales = [col for col in df_KOI_imputado.columns if col not in encoded_columns]

decoded_series = df_KOI_imputado[encoded_columns].idxmax(axis=1)

df_KOI['tfopwg_disp_imputed'] = decoded_series.str.replace('state_', '')

final_df = df_KOI_imputado[columnas_originales].copy()
final_df['tfopwg_disp'] = df_KOI['tfopwg_disp_imputed']

# Split Data for the model

X = final_df.drop('tfopwg_disp', axis=1)
y = final_df['tfopwg_disp']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 1,
                                                    stratify = y)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = (sc.transform(X_train))
X_test_std = (sc.transform(X_test))

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=600,
                                criterion='log_loss',
                                max_features='sqrt',
                                max_depth=60,
                                min_samples_leaf = 6,
                                min_samples_split = 10,
                                max_samples = 0.9,
                                oob_score = True,
                                class_weight = 'balanced',
                                )      
                      
forest.fit(X_train_std, y_train)
print('Train Accuracy : %.5f' % forest.score(X_train_std, y_train))
print('Test Accuracy : %.5f' % forest.score(X_test_std, y_test))

y_pred = forest.predict(X_test_std)
cm = confusion_matrix(y_test, y_pred, normalize='true')

Train_Accuracy = forest.score(X_train_std, y_train)
Test_Accuracy = forest.score(X_test_std, y_test)

from sklearn.metrics import confusion_matrix
r = confusion_matrix(y_test, y_pred, normalize='true').diagonal(); print(f"Balanced accuracy = {r.mean():.3f} = ({' + '.join(f'{x:.2f}' for x in r)})/{r.size}")

Overfiting_gap = Train_Accuracy - Test_Accuracy
print(f'Overfiting gap: {Overfiting_gap}')
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()