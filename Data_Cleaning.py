import pandas as pd

KOI_raw_data = pd.read_csv('Data/KOI_data_cumulative_2025.09.22_07.28.01.csv',skiprows=144)
TOI_raw_data = pd.read_csv('Data/TOI_data_2025.09.22_07.28.57.csv', skiprows=90)
K2_raw_data = pd.read_csv('Data/K2_data_k2pandc_2025.09.22_07.31.46.csv',skiprows=298)

useless_columns_TOI = ['toi','raerr1','raerr2','decerr1','decerr2','st_pmralim','st_pmrasymerr','st_pmdeclim','st_pmdecsymerr','pl_tranmidlim','pl_tranmidsymerr','pl_orbperlim','pl_orbpersymerr','pl_trandurhlim','pl_trandurhsymerr','pl_trandeplim','pl_trandepsymerr','pl_radelim','pl_radesymerr','pl_insolerr1','pl_insolerr2',
                   'pl_insollim','pl_insolsymerr','pl_eqterr1','pl_eqterr2','pl_eqtlim','pl_eqtsymerr','st_tmaglim','st_tmagsymerr', 'st_distlim','st_distsymerr','st_tefflim','st_teffsymerr','st_logglim','st_loggsymerr','st_radlim','st_radsymerr','toi_created','rowupdate']

print(len(useless_columns_TOI))

df_TOI = TOI_raw_data.drop(columns=useless_columns_TOI)