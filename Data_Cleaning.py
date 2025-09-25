import pandas as pd

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
                   'pl_insollim','pl_insolsymerr','pl_eqterr1','pl_eqterr2','pl_eqtlim','pl_eqtsymerr','st_tmaglim','st_tmagsymerr', 'st_distlim','st_distsymerr','st_tefflim','st_teffsymerr','st_logglim','st_loggsymerr','st_radlim','st_radsymerr','toi_created','rowupdate']

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

print(df_KOI.shape)
print(df_TOI.shape)
print(df_K2.shape)