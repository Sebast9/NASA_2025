
import pandas as pd
import numpy as np

COMMON_SCHEMA = [
    'disposition','pl_orbper','pl_trandur_h','pl_trandep_ppm','pl_rade',
    'pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid_bjd',
]

TOI_COLUMNS = ['tfopwg_disp','pl_orbper','pl_trandurh','pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
K2_COLUMNS  = ['disposition','pl_orbper','pl_trandur','pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
KOI_COLUMNS = ['koi_disposition','koi_period','koi_duration','koi_depth','koi_prad','koi_teq','koi_insol','koi_steff','koi_slogg','koi_srad','ra','dec','koi_time0']

MAP_TOI = dict(zip(TOI_COLUMNS, [
    'disposition_toi','pl_orbper_toi','pl_trandur_h_toi','pl_trandep_ppm_toi','pl_rade_toi','pl_eqt_toi','pl_insol_toi','st_teff_toi','st_logg_toi','st_rad_toi','ra_toi','dec_toi','pl_tranmid_bjd_toi'
]))
MAP_K2 = dict(zip(K2_COLUMNS, [
    'disposition_k2','pl_orbper_k2','pl_trandur_h_k2','pl_trandep_pct_k2','pl_rade_k2','pl_eqt_k2','pl_insol_k2','st_teff_k2','st_logg_k2','st_rad_k2','ra_k2','dec_k2','pl_tranmid_bjd_k2'
]))
MAP_KOI = dict(zip(KOI_COLUMNS, [
    'disposition_koi','pl_orbper_koi','pl_trandur_h_koi','pl_trandep_ppm_koi','pl_rade_koi','pl_eqt_koi','pl_insol_koi','st_teff_koi','st_logg_koi','st_rad_koi','ra_koi','dec_koi','pl_tranmid_bjd_koi'
]))

def _normalize_disposition(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().lower()
    if 'confirm' in v or v in ['cp','cpf']: return 'Confirmed'
    if 'false'   in v or v=='fp' or 'false positive' in v: return 'False Positive'
    if 'cand'    in v or 'candidate' in v: return 'Candidate'
    return val

def _coalesce_row(row):
    vals = [x for x in row if pd.notna(x)]
    return vals[0] if vals else np.nan

def _safe_mul(x, factor):
    try: return float(x)*factor if pd.notna(x) else np.nan
    except: return np.nan

def unify(df_koi, df_toi, df_k2, priority=('toi','koi','k2')):
    koi = df_koi.copy() if df_koi is not None else pd.DataFrame()
    toi = df_toi.copy() if df_toi is not None else pd.DataFrame()
    k2  = df_k2.copy()  if df_k2  is not None else pd.DataFrame()

    # KOI: BKJD -> BJD si hace falta
    if 'koi_time0' not in koi.columns and 'koi_time0bk' in koi.columns:
        koi['koi_time0'] = koi['koi_time0bk'] + 2454833.0

    koi_r = koi.rename(columns={c: MAP_KOI[c] for c in KOI_COLUMNS if c in koi.columns})
    toi_r = toi.rename(columns={c: MAP_TOI[c] for c in TOI_COLUMNS if c in toi.columns})
    k2_r  = k2 .rename(columns={c: MAP_K2[c]  for c in K2_COLUMNS  if c in k2.columns})

    df = koi_r.join(toi_r, how='outer').join(k2_r, how='outer')

    if 'pl_trandep_pct_k2' in df.columns:
        df['pl_trandep_ppm_k2'] = df['pl_trandep_pct_k2'].apply(lambda x: _safe_mul(x, 1e4))

    for c in ['disposition_toi','disposition_koi','disposition_k2']:
        if c in df.columns: df[c+'_n'] = df[c].map(_normalize_disposition)

    def pick(cols):
        cols = [c for c in cols if c in df.columns]
        return df[cols].apply(_coalesce_row, axis=1) if cols else np.nan

    prio = {'toi':'_toi','koi':'_koi','k2':'_k2'}
    def pcols(base): return [f'{base}{prio[src]}' for src in priority]

    out = pd.DataFrame(index=df.index)

    disp_cols = [f'disposition_{src}_n' for src in priority if f'disposition_{src}_n' in df.columns]
    if disp_cols:
        order = ['Confirmed','Candidate','False Positive']
        def best(row):
            vals = [x for x in row if pd.notna(x)]
            for o in order:
                if o in vals: return o
            return vals[0] if vals else np.nan
        out['disposition'] = df[disp_cols].apply(best, axis=1)
    else:
        out['disposition'] = np.nan

    out['pl_orbper']    = pick(pcols('pl_orbper'))
    out['pl_trandur_h'] = pick(pcols('pl_trandur_h'))

    depth_cols = pcols('pl_trandep_ppm')
    if 'pl_trandep_ppm_k2' in df.columns and 'pl_trandep_ppm_k2' not in depth_cols:
        depth_cols.append('pl_trandep_ppm_k2')
    out['pl_trandep_ppm'] = pick(depth_cols)

    for base in ['pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad']:
        out[base] = pick(pcols(base))

    out['ra']  = pick(pcols('ra'))
    out['dec'] = pick(pcols('dec'))

    out['pl_tranmid_bjd'] = pick(pcols('pl_tranmid_bjd'))

    return out[[c for c in COMMON_SCHEMA if c in out.columns]]

def unify_with_sources(df_koi, df_toi, df_k2, **kwargs):
    final = unify(df_koi, df_toi, df_k2, **kwargs)
    koi_r = df_koi.rename(columns={c: MAP_KOI[c] for c in KOI_COLUMNS if c in df_koi.columns}) if df_koi is not None else pd.DataFrame()
    toi_r = df_toi.rename(columns={c: MAP_TOI[c] for c in TOI_COLUMNS if c in df_toi.columns}) if df_toi is not None else pd.DataFrame()
    k2_r  = df_k2 .rename(columns={c: MAP_K2[c]  for c in K2_COLUMNS  if c in df_k2.columns}) if df_k2  is not None else pd.DataFrame()
    return final, koi_r.join(toi_r, how='outer').join(k2_r, how='outer')
