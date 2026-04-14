#!/usr/bin/env python3
"""
drs_homology_aware_ml_v5.py
============================
Homology-Aware Random Forest Classification for Deep Receptor Scanning (DRS)

Implements reviewer-requested homology-based train/test splitting for all four
model configurations from Tedman et al. Uses GroupShuffleSplit so closely
related GPCRs (by sequence identity) never span train and test sets.

Reads from Doc_S2_Machine_Learning_Features.xlsx (--features_xlsx) or from
the legacy CSVs (GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv and
HIGH_EXPRESSION_GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv via --data_dir).

Produces figures and CSV data files for BOTH homology-aware and random splits
(ROC curves, feature importances, SHAP beeswarms, per-split metrics).

Usage:
    python drs_homology_aware_ml_v5.py --features_xlsx Doc_S2_Machine_Learning_Features.xlsx
    python drs_homology_aware_ml_v5.py --features_xlsx Doc_S2_Machine_Learning_Features.xlsx --seq_identity 0.5 --n_splits 50
    python drs_homology_aware_ml_v5.py --data_dir . --seq_identity 0.5 --n_splits 50
    python drs_homology_aware_ml_v5.py --data_dir . --cluster_csv clusters.csv
    python drs_homology_aware_ml_v5.py --data_dir . --sensitivity

Requirements (environment.yaml compatible):
    python=3.10, numpy, pandas, scikit-learn=1.4, matplotlib, seaborn, shap=0.45, openpyxl
    MMseqs2 on PATH (conda install -c bioconda mmseqs2)
"""
import os, sys, argparse, subprocess, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False; print("WARNING: shap not installed — SHAP plots skipped.")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS — exact column names from the CSVs
# ══════════════════════════════════════════════════════════════════════════════
META_COLS = ['compound_name', 'GPCR_PME']

TOPOLOGY_FEATURES = [
    'relative_tmd_1','relative_tmd_2','relative_tmd_3','relative_tmd_4',
    'relative_tmd_5','relative_tmd_6','relative_tmd_7',
    'relative_N_terminal_loop_length',
    'relative_downstream_loop_length_1','relative_upstream_loop_length_1',
    'relative_downstream_loop_length_2','relative_upstream_loop_length_2',
    'relative_downstream_loop_length_3','relative_upstream_loop_length_3',
    'relative_C_terminal_loop_length',
    'Molecular_Weight','Isoelectric_Point','Aromaticity','instability_index','gravy',
    'adjusted_tmd_delta_G_1','adjusted_tmd_delta_G_2','adjusted_tmd_delta_G_3',
    'adjusted_tmd_delta_G_4','adjusted_tmd_delta_G_5','adjusted_tmd_delta_G_6',
    'adjusted_tmd_delta_G_7',
]

STRUCTURAL_FEATURES = [
    'sasa_total','sasa_crg','sasa_plr','sasa_aplr',
    'f_crg','f_plr','f_aplr',
    'alpha_helix','3_10_helix','extended_configuration',
    'isolated_beta_bridge','turn','coil',
]

TRANSCRIPT_FEATURES = [
    'CAI','Nc','GC3s','CpG_frame1_2','avgCU','CPS_sum','CPSpL',
    'Global_tAI','tAI10Min','tAI10Max','tAI10q25.25.','tAI10q75.75.',
    'avgCU_first20','avgCU_first10','avgCU_first5',
    'GC','GC10min','GC10q25','GC10q75','GC10max',
    'X40deltaG','X40freqens','plus10valRNAss',
    'zeroto38avgRNAss','zeroto38minRNAss','zeroto38q25RNAss',
    'zeroto38q75RNAss','zeroto38maxRNAss',
    'deltaG','freqens','avgRNAss','minRNAss','q25RNAss','q75RNAss','maxRNAss',
]

RENAME_MAP = {
    'sasa_total':'Total SASA','sasa_crg':'Charged SASA','sasa_plr':'sasa plr',
    'sasa_aplr':'sasa aplr','isolated_beta_bridge':'isolated beta bridge',
    'adjusted_tmd_delta_G_6':r'$\Delta G_{app,pred}$ TMD6',
    'adjusted_tmd_delta_G_2':r'$\Delta G_{app,pred}$ TMD2',
    'adjusted_tmd_delta_G_4':r'$\Delta G_{app,pred}$ TMD4',
    'adjusted_tmd_delta_G_1':r'$\Delta G_{app,pred}$ TMD1',
    'adjusted_tmd_delta_G_3':r'$\Delta G_{app,pred}$ TMD3',
    'adjusted_tmd_delta_G_5':r'$\Delta G_{app,pred}$ TMD5',
    'adjusted_tmd_delta_G_7':r'$\Delta G_{app,pred}$ TMD7',
    'alpha_helix':'Alpha Helix','3_10_helix':r'$3_{10}$ Helix Content',
    'Molecular_Weight':'Molecular weight',
    'relative_upstream_loop_length_2':'ECL2 Length',
    'relative_upstream_loop_length_1':'ECL1 Length',
    'relative_upstream_loop_length_3':'ECL3 Length',
    'relative_downstream_loop_length_1':'ICL1 Length',
    'relative_downstream_loop_length_2':'ICL2 Length',
    'relative_downstream_loop_length_3':'ICL3 Length',
    'relative_C_terminal_loop_length':'C-term Loop Length',
    'relative_N_terminal_loop_length':'N-term Loop Length',
    'relative_tmd_1':'TMD1 Length','relative_tmd_2':'TMD2 Length',
    'relative_tmd_3':'TMD3 Length','relative_tmd_4':'TMD4 Length',
    'relative_tmd_5':'TMD5 Length','relative_tmd_6':'TMD6 Length',
    'relative_tmd_7':'TMD7 Length',
}

RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_split=5,
                 class_weight='balanced', random_state=42)

MODEL_CONFIGS = {
    'All_GPCRs_All_Features': {
        'csv':'GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv',
        'xlsx_sheet':'All Receptors',
        'features': TOPOLOGY_FEATURES+STRUCTURAL_FEATURES+TRANSCRIPT_FEATURES,
        'desc':'All GPCRs, full feature set (n=75)',
    },
    'HighGFP_Structure_Topology': {
        'csv':'HIGH_EXPRESSION_GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv',
        'xlsx_sheet':'High GFP Receptors',
        'features': TOPOLOGY_FEATURES+STRUCTURAL_FEATURES,
        'desc':'High-GFP GPCRs, structure+topology (n=40)',
    },
    'HighGFP_Topology_Only': {
        'csv':'HIGH_EXPRESSION_GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv',
        'xlsx_sheet':'High GFP Receptors',
        'features': TOPOLOGY_FEATURES,
        'desc':'High-GFP GPCRs, topology only (n=27)',
    },
    'HighGFP_Structure_Only': {
        'csv':'HIGH_EXPRESSION_GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv',
        'xlsx_sheet':'High GFP Receptors',
        'features': STRUCTURAL_FEATURES,
        'desc':'High-GFP GPCRs, structure only (n=13)',
    },
}

MANUAL_PME_CUTOFF = 2644

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def extract_fasta(source, out_fasta, id_col='compound_name', seq_col='Sequence'):
    """source can be a CSV path (str) or a pandas DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        df = pd.read_csv(source)
    n = 0
    with open(out_fasta,'w') as f:
        for _,row in df.iterrows():
            s = str(row[seq_col]).strip()
            nm = str(row[id_col]).strip()
            if s and s!='nan' and len(s)>10:
                f.write(f'>{nm}\n{s}\n'); n+=1
    print(f"  Wrote {n} sequences to {out_fasta}")

def cluster_mmseqs2(fasta, sid=0.5, cov=0.8, wdir='mmseqs2_work', binary='mmseqs'):
    os.makedirs(wdir, exist_ok=True)
    pfx = os.path.join(wdir,'clusterRes')
    tmp = os.path.join(wdir,'tmp')
    tsv = pfx+'_cluster.tsv'
    print(f"\n{'='*60}\nCLUSTERING (identity={sid}, coverage={cov})\n{'='*60}")
    cmd = [binary,'easy-cluster',fasta,pfx,tmp,
           '--min-seq-id',str(sid),'-c',str(cov),
           '--cluster-mode','0','--cov-mode','0']
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode!=0:
        print(f"  STDERR: {r.stderr[:500]}")
        raise RuntimeError(f"MMseqs2 failed (code {r.returncode})")
    raw = pd.read_csv(tsv, sep='\t', header=None, names=['rep','member'])
    rmap = {r:i for i,r in enumerate(raw['rep'].unique())}
    raw['cluster_id'] = raw['rep'].map(rmap)
    cdf = raw.rename(columns={'member':'compound_name'})
    cmap = dict(zip(cdf['compound_name'], cdf['cluster_id']))
    nc = cdf['cluster_id'].nunique()
    sz = cdf.groupby('cluster_id').size()
    print(f"  {len(cdf)} seqs -> {nc} clusters (median={sz.median():.0f}, max={sz.max()})")
    out = os.path.join(wdir,'cluster_assignments.csv')
    cdf[['compound_name','cluster_id','rep']].to_csv(out,index=False)
    print(f"  Saved: {out}")
    return cmap, cdf

def load_clusters(path):
    df = pd.read_csv(path)
    cmap = dict(zip(df['compound_name'], df['cluster_id']))
    print(f"Loaded {len(cmap)} assignments ({df['cluster_id'].nunique()} clusters)")
    return cmap, df

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — mirrors notebook cells exactly
# ══════════════════════════════════════════════════════════════════════════════
def load_data(config, data_dir, cmap, xlsx_path=None):
    if xlsx_path:
        df = pd.read_excel(xlsx_path, sheet_name=config['xlsx_sheet'])
    else:
        csv = os.path.join(data_dir, config['csv'])
        df = pd.read_csv(csv)
    sel = META_COLS + config['features']
    ds = df[sel].copy()
    feat = config['features']
    imp = SimpleImputer(strategy='median')
    X_df = pd.DataFrame(imp.fit_transform(ds[feat]), columns=feat, index=ds.index)
    y = (ds['GPCR_PME'].values >= MANUAL_PME_CUTOFF).astype(int)
    names = ds['compound_name'].values
    cids = np.array([cmap.get(n, np.nan) for n in names])
    ok = ~np.isnan(cids)
    if (~ok).sum()>0: print(f"  Dropping {(~ok).sum()} without cluster")
    return X_df.values[ok], y[ok], cids[ok].astype(int), feat, names[ok]

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def _eval_loop(splitter, X, y, groups, rf_params):
    M = {k:[] for k in ['bal_acc','f1','precision','recall','roc_auc']}
    fpr_grid = np.linspace(0,1,100); tprs=[]; imps=[]
    si = {k:[] for k in ['n_tr','n_te','nc_tr','nc_te','fh_tr','fh_te']}
    last = [None,None,None]
    kw = dict(groups=groups) if groups is not None else {}
    for tri,tei in splitter.split(X, y, **kw):
        Xtr,Xte = X[tri],X[tei]; ytr,yte = y[tri],y[tei]
        if groups is not None:
            gtr,gte = groups[tri],groups[tei]
            assert len(set(gtr)&set(gte))==0
            si['nc_tr'].append(len(set(gtr))); si['nc_te'].append(len(set(gte)))
        si['n_tr'].append(len(tri)); si['n_te'].append(len(tei))
        si['fh_tr'].append(ytr.mean()); si['fh_te'].append(yte.mean())
        clf = RandomForestClassifier(**rf_params); clf.fit(Xtr,ytr)
        yp = clf.predict(Xte); ypr = clf.predict_proba(Xte)[:,1]
        M['bal_acc'].append(balanced_accuracy_score(yte,yp))
        M['f1'].append(f1_score(yte,yp,zero_division=0))
        M['precision'].append(precision_score(yte,yp,zero_division=0))
        M['recall'].append(recall_score(yte,yp,zero_division=0))
        if len(np.unique(yte))==2:
            M['roc_auc'].append(roc_auc_score(yte,ypr))
            fp,tp,_ = roc_curve(yte,ypr); tprs.append(np.interp(fpr_grid,fp,tp))
        else: M['roc_auc'].append(np.nan)
        imps.append(clf.feature_importances_)
        last = [clf, Xte, yte]
    return {
        'metrics':{k:(np.nanmean(v),np.nanstd(v)) for k,v in M.items()},
        'metrics_per_split': M,
        'split_info':{k:(np.mean(v),np.std(v)) for k,v in si.items() if v},
        'roc':{'fpr':fpr_grid,
               'tpr_mean':np.mean(tprs,axis=0) if tprs else None,
               'tpr_std':np.std(tprs,axis=0) if tprs else None},
        'imp_mean':np.mean(imps,axis=0) if imps else None,
        'imp_std':np.std(imps,axis=0) if imps else None,
        'last_model':last[0],'last_Xte':last[1],'last_yte':last[2],
    }

def eval_grouped(X,y,g,rf_params,n_splits=50,test_size=0.2,rs=42):
    sp = GroupShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=rs)
    return _eval_loop(sp,X,y,g,rf_params)

def eval_random(X,y,rf_params,n_splits=50,test_size=0.2,rs=42):
    sp = ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=rs)
    return _eval_loop(sp,X,y,None,rf_params)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING — per-split + combined figures, each with companion CSV
# ══════════════════════════════════════════════════════════════════════════════

def _split_tag(split_label):
    """Convert 'Homology-Aware' -> 'homology_aware', etc."""
    return split_label.lower().replace('-','_').replace(' ','_')

# ── ROC ───────────────────────────────────────────────────────────────────────

def plot_roc_single(res, split_label, name, odir, color='steelblue'):
    """ROC curve for one split type. Saves PDF + CSV."""
    if res['roc']['tpr_mean'] is None: return
    m,s = res['metrics']['roc_auc']
    tag = _split_tag(split_label)
    fig,ax = plt.subplots(figsize=(6,6))
    ax.plot(res['roc']['fpr'], res['roc']['tpr_mean'], color=color,
            label=f'AUROC={m:.3f}\u00b1{s:.3f}')
    ax.fill_between(res['roc']['fpr'],
                    res['roc']['tpr_mean']-res['roc']['tpr_std'],
                    res['roc']['tpr_mean']+res['roc']['tpr_std'],
                    alpha=0.15, color=color)
    ax.plot([0,1],[0,1],'k:',alpha=0.4)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           xlim=[-0.02,1.02], ylim=[-0.02,1.02])
    ax.set_title(f'{name.replace("_"," ")}\n{split_label}', fontsize=13)
    ax.legend(loc='lower right', fontsize=9); plt.tight_layout()
    fig.savefig(os.path.join(odir, f'roc_{tag}_{name}.pdf'), dpi=300); plt.close(fig)
    pd.DataFrame({
        'FPR': res['roc']['fpr'],
        'TPR_mean': res['roc']['tpr_mean'],
        'TPR_std': res['roc']['tpr_std'],
    }).to_csv(os.path.join(odir, f'roc_data_{tag}_{name}.csv'), index=False)

def plot_roc_combined(rg, rr, name, odir):
    """Overlay ROC for both splits. Saves PDF + CSV."""
    fig,ax = plt.subplots(figsize=(6,6))
    for res,col,ls,lab in [(rr,'gray','--','Random'),(rg,'steelblue','-','Homology-Aware')]:
        if res['roc']['tpr_mean'] is None: continue
        m,s = res['metrics']['roc_auc']
        ax.plot(res['roc']['fpr'], res['roc']['tpr_mean'], color=col, linestyle=ls,
                label=f'{lab} (AUROC={m:.3f}\u00b1{s:.3f})')
        ax.fill_between(res['roc']['fpr'],
                        res['roc']['tpr_mean']-res['roc']['tpr_std'],
                        res['roc']['tpr_mean']+res['roc']['tpr_std'],
                        alpha=0.12, color=col)
    ax.plot([0,1],[0,1],'k:',alpha=0.4)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           xlim=[-0.02,1.02], ylim=[-0.02,1.02])
    ax.set_title(name.replace('_',' '), fontsize=13)
    ax.legend(loc='lower right', fontsize=9); plt.tight_layout()
    fig.savefig(os.path.join(odir, f'roc_combined_{name}.pdf'), dpi=300); plt.close(fig)
    roc_df = pd.DataFrame({'FPR': rg['roc']['fpr']})
    for res,lab in [(rg,'homology_aware'),(rr,'random')]:
        if res['roc']['tpr_mean'] is not None:
            roc_df[f'TPR_mean_{lab}'] = res['roc']['tpr_mean']
            roc_df[f'TPR_std_{lab}'] = res['roc']['tpr_std']
    roc_df.to_csv(os.path.join(odir, f'roc_data_combined_{name}.csv'), index=False)

# ── Feature Importance ────────────────────────────────────────────────────────

def plot_imp_single(res, split_label, fnames, name, odir, top_n=15):
    """Importance bar chart for one split type. Saves PDF + CSV."""
    imp = res['imp_mean']
    if imp is None: return
    dn = [RENAME_MAP.get(f,f) for f in fnames]
    tag = _split_tag(split_label)
    n = min(top_n, len(imp))
    idx = np.argsort(imp)[::-1][:n]
    fig,ax = plt.subplots(figsize=(8, max(4, n*0.4)))
    bars = imp[idx][::-1]
    errs = res['imp_std'][idx][::-1] if res['imp_std'] is not None else None
    ax.barh(range(n), bars, xerr=errs, color='steelblue', alpha=0.8,
            ecolor='gray', capsize=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([dn[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel('Gini Importance')
    ax.set_title(f'{name.replace("_"," ")}\n{split_label}', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(odir, f'importance_{tag}_{name}.pdf'),
                dpi=300, bbox_inches='tight'); plt.close(fig)
    imp_s = res.get('imp_std')
    pd.DataFrame([{
        'feature': f, 'display_name': dn[i],
        'importance_mean': imp[i],
        'importance_std': imp_s[i] if imp_s is not None else np.nan,
    } for i,f in enumerate(fnames)]).sort_values(
        'importance_mean', ascending=False).to_csv(
        os.path.join(odir, f'importance_data_{tag}_{name}.csv'), index=False)

def plot_imp_combined(rg, rr, fnames, name, odir, top_n=15):
    """Side-by-side importance for both splits. Saves PDF + CSV."""
    dn = [RENAME_MAP.get(f,f) for f in fnames]
    fig,axes = plt.subplots(1,2,figsize=(14,max(6,top_n*0.45)))
    for ax,res,lab in [(axes[0],rr,'Random Split'),(axes[1],rg,'Homology-Aware Split')]:
        imp = res['imp_mean']
        if imp is None: continue
        n = min(top_n, len(imp))
        idx = np.argsort(imp)[::-1][:n]
        ax.barh(range(n),imp[idx][::-1],color='steelblue',alpha=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([dn[i] for i in idx][::-1],fontsize=9)
        ax.set_xlabel('Gini Importance'); ax.set_title(lab)
    plt.suptitle(name.replace('_',' '),fontsize=13,y=1.01); plt.tight_layout()
    fig.savefig(os.path.join(odir,f'importance_combined_{name}.pdf'),
                dpi=300,bbox_inches='tight'); plt.close(fig)
    imp_rows = []
    for res,lab in [(rr,'random'),(rg,'homology_aware')]:
        imp = res['imp_mean']; imp_s = res.get('imp_std')
        if imp is None: continue
        for i,f in enumerate(fnames):
            imp_rows.append({
                'feature': f, 'display_name': dn[i], 'split': lab,
                'importance_mean': imp[i],
                'importance_std': imp_s[i] if imp_s is not None else np.nan,
            })
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(
            os.path.join(odir, f'importance_data_combined_{name}.csv'), index=False)

# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(res, split_label, fnames, name, odir, top_n=10):
    """SHAP beeswarm for one split type. Saves PDF + CSVs."""
    if not HAS_SHAP: return
    mdl, Xte = res['last_model'], res['last_Xte']
    if mdl is None: return
    dn = [RENAME_MAP.get(f,f) for f in fnames]
    tag = _split_tag(split_label)
    print(f"  Computing SHAP ({split_label}) for {name}..."); np.random.seed(42)
    ex = shap.TreeExplainer(mdl); sv = ex.shap_values(Xte)
    if isinstance(sv, list): sv = sv[1]
    elif sv.ndim == 3: sv = sv[:,:,1]
    plt.figure(figsize=(10,6))
    shap.summary_plot(sv, Xte, feature_names=dn, max_display=top_n, show=False)
    plt.title(f'SHAP \u2014 {split_label}\n{name.replace("_"," ")}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(odir, f'shap_{tag}_{name}.pdf'),
                dpi=300, bbox_inches='tight'); plt.close()
    pd.DataFrame(sv, columns=dn).to_csv(
        os.path.join(odir, f'shap_values_{tag}_{name}.csv'), index=False)
    pd.DataFrame({
        'feature': fnames, 'display_name': dn,
        'mean_abs_shap': np.abs(sv).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).to_csv(
        os.path.join(odir, f'shap_summary_{tag}_{name}.csv'), index=False)

# ── Cluster diagnostics ──────────────────────────────────────────────────────

def plot_cluster_diag(cmap, df_all, odir):
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    cs = pd.Series(Counter(cmap.values()))
    axes[0].hist(cs,bins=np.arange(0.5,cs.max()+1.5,1),color='steelblue',edgecolor='white')
    axes[0].set(xlabel='Cluster Size',ylabel='Count',title='Cluster Size Distribution')
    axes[0].axvline(cs.median(),color='red',ls='--',label=f'Median={cs.median():.0f}')
    axes[0].legend()
    dt = df_all.copy(); dt['cid']=dt['compound_name'].map(cmap)
    st = dt.dropna(subset=['cid']).groupby('cid')['GPCR_PME'].agg(['mean','std','count'])
    m = st[st['count']>1]
    if len(m)>0:
        axes[1].scatter(m['mean'],m['std'],alpha=0.5,s=m['count']*8,
                       color='steelblue',edgecolors='white')
        axes[1].set(xlabel='Mean Cluster PME',ylabel='Std Dev',
                    title='Intra-Cluster PME Heterogeneity')
    plt.tight_layout()
    fig.savefig(os.path.join(odir,'cluster_diagnostics.pdf'),dpi=300); plt.close(fig)
    cs.name = 'cluster_size'
    cs.to_csv(os.path.join(odir,'cluster_sizes.csv'), header=True)
    st.to_csv(os.path.join(odir,'cluster_pme_stats.csv'))

# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
def summary_table(AR, odir):
    rows=[]
    for cn,res in AR.items():
        for slab,sk in [('Random','random'),('Homology-Aware','grouped')]:
            if sk not in res: continue
            r=res[sk]; row={'Config':cn,'Split':slab}
            for k in ['bal_acc','f1','precision','recall','roc_auc']:
                m,s=r['metrics'][k]; row[k]=f"{m:.3f} +/- {s:.3f}"
            rows.append(row)
    df=pd.DataFrame(rows)
    df.columns=['Config','Split','Bal Acc','F1','Precision','Recall','ROC-AUC']
    p=os.path.join(odir,'model_comparison_table.csv'); df.to_csv(p,index=False)
    print(f"\n{'='*90}\nMODEL COMPARISON\n{'='*90}")
    print(df.to_string(index=False)); print(f"Saved: {p}")
    return df

def split_table(AR, odir):
    rows=[]
    for cn,res in AR.items():
        for slab,sk in [('Homology-Aware','grouped'),('Random','random')]:
            if sk not in res: continue
            si=res[sk]['split_info']
            row = {'Config':cn, 'Split':slab,
                'Train N':f"{si['n_tr'][0]:.0f}+/-{si['n_tr'][1]:.0f}",
                'Test N':f"{si['n_te'][0]:.0f}+/-{si['n_te'][1]:.0f}",
                'Train %High':f"{si['fh_tr'][0]*100:.1f}%+/-{si['fh_tr'][1]*100:.1f}%",
                'Test %High':f"{si['fh_te'][0]*100:.1f}%+/-{si['fh_te'][1]*100:.1f}%"}
            if 'nc_tr' in si:
                row['Train Clusters'] = f"{si['nc_tr'][0]:.0f}"
                row['Test Clusters'] = f"{si['nc_te'][0]:.0f}"
            rows.append(row)
    if rows:
        df=pd.DataFrame(rows); p=os.path.join(odir,'split_diagnostics.csv')
        df.to_csv(p,index=False); print(f"\n{df.to_string(index=False)}\nSaved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    P = argparse.ArgumentParser(description='Homology-Aware ML for DRS (v5)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument('--data_dir',default='.')
    P.add_argument('--features_xlsx',default=None,
                   help='Path to Doc_S2 XLSX with sheets "All Receptors" and '
                        '"High GFP Receptors". If provided, used instead of CSVs.')
    P.add_argument('--cluster_csv',default=None)
    P.add_argument('--seq_identity',type=float,default=0.5)
    P.add_argument('--coverage',type=float,default=0.8)
    P.add_argument('--n_splits',type=int,default=50)
    P.add_argument('--test_size',type=float,default=0.2)
    P.add_argument('--output_dir',default=None,
                   help='Output dir (default: results_sid{seq_identity})')
    P.add_argument('--mmseqs_binary',default='mmseqs')
    P.add_argument('--skip_random',action='store_true')
    P.add_argument('--skip_shap',action='store_true')
    P.add_argument('--sensitivity',action='store_true')
    args = P.parse_args()

    if args.output_dir is None:
        if args.cluster_csv:
            args.output_dir = 'results_precomputed_clusters'
        else:
            sid_str = str(args.seq_identity).replace('.','')
            args.output_dir = f'results_sid{sid_str}'

    os.makedirs(args.output_dir, exist_ok=True); np.random.seed(42)

    xlsx_path = args.features_xlsx
    if xlsx_path:
        if not os.path.exists(xlsx_path):
            sys.exit(f"ERROR: {xlsx_path} not found.")
        import openpyxl
        print(f"Using XLSX features: {xlsx_path}")
        xl = pd.ExcelFile(xlsx_path)
        for cfg in MODEL_CONFIGS.values():
            sn = cfg['xlsx_sheet']
            if sn not in xl.sheet_names:
                sys.exit(f"ERROR: Sheet '{sn}' not found in {xlsx_path}")
            df_check = pd.read_excel(xl, sheet_name=sn, nrows=0)
            missing = [c for c in META_COLS + cfg['features'] if c not in df_check.columns]
            if missing:
                sys.exit(f"ERROR: Sheet '{sn}' missing columns: {missing}")
    else:
        allcsv = os.path.join(args.data_dir,'GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv')
        highcsv = os.path.join(args.data_dir,'HIGH_EXPRESSION_GPCR_TRANSCRIPT_PROTEIN_FEATURES.csv')
        for f in [allcsv,highcsv]:
            if not os.path.exists(f): sys.exit(f"ERROR: {f} not found. Set --data_dir.")

    # Load "all" DataFrame for FASTA / cluster diagnostics
    if xlsx_path:
        df_all = pd.read_excel(xlsx_path, sheet_name='All Receptors')
    else:
        df_all = pd.read_csv(allcsv)

    # Clustering
    if args.cluster_csv:
        cmap,cdf = load_clusters(args.cluster_csv)
    else:
        fasta = os.path.join(args.output_dir,'gpcr_sequences.fasta')
        extract_fasta(df_all, fasta)
        cmap,cdf = cluster_mmseqs2(fasta, sid=args.seq_identity, cov=args.coverage,
                                    wdir=os.path.join(args.output_dir,'mmseqs2'),
                                    binary=args.mmseqs_binary)
    plot_cluster_diag(cmap, df_all, args.output_dir)

    # ── Run all model configs ─────────────────────────────────────────────────
    AR = {}
    for cn,cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}\n{cn}\n  {cfg['desc']}\n{'='*60}")
        X,y,g,fn,_ = load_data(cfg, args.data_dir, cmap, xlsx_path=xlsx_path)
        nc = len(set(g))
        print(f"  N={len(X)}, clusters={nc}, features={X.shape[1]}, %high={y.mean()*100:.1f}%")
        if nc<5: print("  SKIP: too few clusters"); continue
        AR[cn] = {}

        # Homology-aware
        print(f"\n  --- Homology-Aware ({args.n_splits} splits) ---")
        rg = eval_grouped(X,y,g,RF_PARAMS,n_splits=args.n_splits,test_size=args.test_size)
        AR[cn]['grouped'] = rg
        m,s=rg['metrics']['roc_auc']; print(f"  AUROC: {m:.3f}+/-{s:.3f}")
        m,s=rg['metrics']['bal_acc']; print(f"  BalAcc: {m:.3f}+/-{s:.3f}")
        plot_roc_single(rg, 'Homology-Aware', cn, args.output_dir)
        plot_imp_single(rg, 'Homology-Aware', fn, cn, args.output_dir)
        if not args.skip_shap and HAS_SHAP:
            plot_shap(rg, 'Homology-Aware', fn, cn, args.output_dir)

        # Random
        if not args.skip_random:
            print(f"\n  --- Random ({args.n_splits} splits) ---")
            rr = eval_random(X,y,RF_PARAMS,n_splits=args.n_splits,test_size=args.test_size)
            AR[cn]['random'] = rr
            m,s=rr['metrics']['roc_auc']; print(f"  AUROC: {m:.3f}+/-{s:.3f}")
            m,s=rr['metrics']['bal_acc']; print(f"  BalAcc: {m:.3f}+/-{s:.3f}")
            plot_roc_single(rr, 'Random', cn, args.output_dir, color='gray')
            plot_imp_single(rr, 'Random', fn, cn, args.output_dir)
            if not args.skip_shap and HAS_SHAP:
                plot_shap(rr, 'Random', fn, cn, args.output_dir)
            # Combined overlays
            plot_roc_combined(rg, rr, cn, args.output_dir)
            plot_imp_combined(rg, rr, fn, cn, args.output_dir)

        # Per-split metric CSVs
        for slab,sk in [('homology_aware','grouped'),('random','random')]:
            if sk not in AR[cn]: continue
            mps = AR[cn][sk]['metrics_per_split']
            pd.DataFrame({
                'split': range(len(mps['roc_auc'])),
                'roc_auc': mps['roc_auc'],
                'balanced_accuracy': mps['bal_acc'],
                'f1': mps['f1'],
                'precision': mps['precision'],
                'recall': mps['recall'],
            }).to_csv(os.path.join(args.output_dir,
                      f'per_split_metrics_{slab}_{cn}.csv'), index=False)

    if AR:
        summary_table(AR, args.output_dir)
        split_table(AR, args.output_dir)

    # Sensitivity
    if args.sensitivity and not args.cluster_csv:
        print(f"\n{'='*60}\nSENSITIVITY ANALYSIS\n{'='*60}")
        fasta = os.path.join(args.output_dir,'gpcr_sequences.fasta')
        if not os.path.exists(fasta): extract_fasta(df_all, fasta)
        cfg = MODEL_CONFIGS['HighGFP_Structure_Topology']; srows=[]
        for th in [0.3,0.4,0.5,0.6,0.7]:
            print(f"\n  Threshold: {th}")
            try:
                cm,_ = cluster_mmseqs2(fasta,sid=th,cov=args.coverage,
                    wdir=os.path.join(args.output_dir,f'mmseqs2_{th}'),
                    binary=args.mmseqs_binary)
                X,y,g,fn,_ = load_data(cfg,args.data_dir,cm,xlsx_path=xlsx_path)
                if len(set(g))<5: print("  Too few clusters"); continue
                r = eval_grouped(X,y,g,RF_PARAMS,n_splits=args.n_splits,test_size=args.test_size)
                ma,sa = r['metrics']['roc_auc']; mb,sb = r['metrics']['bal_acc']
                srows.append({'Threshold':th,'Clusters':len(set(g)),
                    'AUROC':f"{ma:.3f}+/-{sa:.3f}",'BalAcc':f"{mb:.3f}+/-{sb:.3f}"})
            except Exception as e: print(f"  ERROR: {e}")
        if srows:
            sdf=pd.DataFrame(srows)
            p=os.path.join(args.output_dir,'sensitivity_analysis.csv')
            sdf.to_csv(p,index=False); print(f"\n{sdf.to_string(index=False)}\nSaved: {p}")

    print(f"\n{'='*60}\nDONE. Outputs: {args.output_dir}\n{'='*60}\n")

if __name__=='__main__': main()
