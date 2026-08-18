"""Config-driven, resumable 2x2 DFA-EHOA experiment runner."""
from __future__ import annotations
import argparse, json, time, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from ehoa import EHOA
from proposed import DFAEHOA
from proposed.interaction import subset_redundancy, subset_interaction_quality
from proposed.stability import compute_jaccard_stability, compute_nogueira_stability
from utils import clean_dataset, evaluate_classifiers

VARIANTS={"EHOA":None,"SF-EHOA":(True,"stability"),"IG-EHOA":(False,"interaction"),"DFA-EHOA":(True,"dual")}
def load_data(name, cfg=None):
    builtins={"breast_cancer":load_breast_cancer,"wine":load_wine}
    if name in builtins:
        # Prefer the committed snapshot so a clone is self-contained. The
        # sklearn loader remains a compatibility fallback for older checkouts.
        snapshot=Path(__file__).resolve().parent/"data"/f"{name}.csv"
        if snapshot.exists():
            frame=pd.read_csv(snapshot)
            if "target" not in frame: raise ValueError(f"Committed dataset {snapshot} has no target column")
            y=frame.pop("target").to_numpy(); X=frame.to_numpy(dtype=float)
            return clean_dataset(X,y)
        bunch=builtins[name](); return clean_dataset(bunch.data,bunch.target)
    external=(cfg or {}).get("external_datasets",{}).get(name)
    if not external: raise ValueError(f"Unknown dataset {name!r}; define it under external_datasets")
    path=Path(external["path"]).expanduser(); frame=pd.read_csv(path,sep=external.get("delimiter",","))
    target=external["target"]
    if target not in frame: raise ValueError(f"Target column {target!r} is absent from {path}")
    y=frame.pop(target).to_numpy(); X=frame.select_dtypes(include=[np.number]).to_numpy()
    if X.shape[1] != frame.shape[1]: raise ValueError("All feature columns must be numeric after removing target")
    return clean_dataset(X,y)

def ci95(values):
    a=np.asarray(values,float); return 0.0 if len(a)<2 else float(1.96*a.std(ddof=1)/np.sqrt(len(a)))

def cliffs_delta(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(sum(x>y for x in a for y in b)-sum(x<y for x in a for y in b))/(len(a)*len(b))

def paired_rank_biserial(a,b):
    differences=np.asarray(a,float)-np.asarray(b,float); nonzero=differences[differences!=0]
    if not len(nonzero): return 0.0
    ranks=rankdata(np.abs(nonzero))
    positive=ranks[nonzero>0].sum(); negative=ranks[nonzero<0].sum()
    return float((positive-negative)/(positive+negative))

def paired_bootstrap_ci(a,b,seed=20260817,repeats=5000):
    differences=np.asarray(a,float)-np.asarray(b,float)
    if not len(differences): return np.nan,np.nan
    rng=np.random.default_rng(seed); indices=rng.integers(0,len(differences),(repeats,len(differences)))
    estimates=np.median(differences[indices],axis=1)
    return tuple(float(v) for v in np.quantile(estimates,[.025,.975]))

def evaluation_splits(X, y, cfg):
    """Yield leakage-safe outer splits while preserving legacy hold-out behavior."""
    protocol=cfg.get("evaluation_protocol","repeated_stratified_holdout")
    if protocol=="repeated_stratified_holdout":
        for seed in cfg["seeds"]:
            train,test=train_test_split(np.arange(len(y)),test_size=cfg.get("test_size",.2),stratify=y,random_state=seed)
            yield int(seed),"repeated_stratified_holdout",int(seed),train,test
        return
    if protocol!="repeated_nested_cv": raise ValueError("evaluation_protocol must be repeated_stratified_holdout or repeated_nested_cv")
    outer_folds=int(cfg.get("outer_folds",5))
    for seed in cfg["seeds"]:
        cv=StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=int(seed))
        for fold_index,(train,test) in enumerate(cv.split(X,y),start=1):
            selector_seed=int(seed)+fold_index*100003
            yield int(seed),f"outer_{fold_index}",selector_seed,train,test

def aggregate(raw):
    numeric=[c for c in ["balanced_accuracy","f1_score","mcc","roc_auc","pr_auc","sensitivity","specificity","selected_features","redundancy","interaction_quality","runtime_seconds","classifier_seconds","peak_memory_mb","jaccard_stability","nogueira_stability"] if c in raw]
    rows=[]
    for keys,g in raw.groupby(["dataset","method"]):
        row={"dataset":keys[0],"method":keys[1],"runs":len(g)}
        for c in numeric:
            row[f"{c}_mean"]=g[c].mean(); row[f"{c}_std"]=g[c].std(); row[f"{c}_median"]=g[c].median(); row[f"{c}_iqr"]=g[c].quantile(.75)-g[c].quantile(.25); row[f"{c}_ci95"]=ci95(g[c])
        rows.append(row)
    return pd.DataFrame(rows)

def statistics(raw):
    rows=[]
    for dataset,g in raw.groupby("dataset"):
        pivot=g.pivot(index=["seed","fold"],columns="method",values="balanced_accuracy").dropna()
        if len(pivot)>=2 and len(pivot.columns)>=3:
            arrays=[pivot[c].to_numpy() for c in pivot]
            if all(np.array_equal(arrays[0],values) for values in arrays[1:]): stat,p=0.0,1.0
            else: stat,p=friedmanchisquare(*arrays)
            rows.append(dict(dataset=dataset,test="friedman",comparison="all",statistic=stat,p_value=p))
        if "EHOA" in pivot:
            for method in pivot.columns:
                if method=="EHOA": continue
                diff=pivot[method]-pivot["EHOA"]; ci_low,ci_high=paired_bootstrap_ci(pivot[method],pivot["EHOA"])
                if np.all(diff==0): stat,p=0.0,1.0
                else:
                    try: stat,p=wilcoxon(pivot[method],pivot["EHOA"])
                    except ValueError: stat,p=0.,1.
                rows.append(dict(dataset=dataset,test="wilcoxon",comparison=f"{method} vs EHOA",statistic=stat,p_value=p,effect_median=diff.median(),effect_ci95_low=ci_low,effect_ci95_high=ci_high,paired_rank_biserial=paired_rank_biserial(pivot[method],pivot["EHOA"]),cliffs_delta=cliffs_delta(pivot[method],pivot["EHOA"]),wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum())))
    frame=pd.DataFrame(rows)
    if not frame.empty:
        pair=frame.test=="wilcoxon"; ordered=frame.loc[pair].sort_values("p_value"); m=len(ordered); adjusted=[]; running=0.0
        for rank,pvalue in enumerate(ordered.p_value): running=max(running,min(1.0,(m-rank)*pvalue)); adjusted.append(running)
        frame.loc[ordered.index,"holm_adjusted_p"]=adjusted; frame.loc[ordered.index,"significant_holm"]=np.asarray(adjusted)<.05
    return frame

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--output",type=Path,default=Path("results")); args=ap.parse_args(argv)
    cfg=yaml.safe_load(args.config.read_text()); root=args.output; [(root/p).mkdir(parents=True,exist_ok=True) for p in ["raw","aggregated","statistics","figures","tables","ablation","sensitivity"]]
    raw_path=root/"raw"/f"{cfg['experiment']}.csv"; existing=pd.read_csv(raw_path) if cfg.get("resume") and raw_path.exists() else pd.DataFrame(); rows=existing.to_dict("records")
    done={(r["dataset"],r["method"],int(r["seed"]),str(r.get("fold","holdout"))) for r in rows}
    method_specs=[]
    for method in cfg["methods"]:
        if method=="DFA-EHOA" and cfg.get("sensitivity"):
            base=cfg.get("dfa",{}).copy()
            for parameter,values in cfg["sensitivity"].items():
                for value in values:
                    method_specs.append((method,f"DFA-EHOA[{parameter}={value}]",base|{parameter:value}))
        else: method_specs.append((method,method,cfg.get("dfa",{})))
    masks={}
    for dataset in cfg["datasets"]:
      X,y=load_data(dataset,cfg)
      for seed,fold,selector_seed,train_indices,test_indices in evaluation_splits(X,y,cfg):
        Xtr,Xte,ytr,yte=X[train_indices],X[test_indices],y[train_indices],y[test_indices]
        for method,label,dfa_parameters in method_specs:
          if (dataset,label,seed,fold) in done: continue
          common=dict(n_hikers=cfg["population"],max_iter=cfg["iterations"],n_folds=cfg["folds"],random_state=selector_seed,verbose=False)
          selector=EHOA(**common) if method=="EHOA" else DFAEHOA(**common,feedback_enabled=VARIANTS[method][0],transition_mode=VARIANTS[method][1],**dfa_parameters)
          tracemalloc.start(); mask,_,features=selector.fit(Xtr,ytr); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
          tick=time.perf_counter(); metric,_,_=evaluate_classifiers(Xtr,ytr,Xte,yte,features,random_state=seed); classifier_seconds=time.perf_counter()-tick; knn=metric[metric.classifier=="knn"].iloc[0]
          key=(dataset,label); masks.setdefault(key,[]).append(mask.astype(int))
          row=dict(dataset=dataset,method=label,seed=seed,selector_seed=selector_seed,fold=fold,evaluation_protocol=cfg.get("evaluation_protocol","repeated_stratified_holdout"),total_features=X.shape[1],selected_features=len(features),feature_ratio=len(features)/X.shape[1],fitness=selector.best_fitness,evaluations=selector.evaluations_,runtime_seconds=selector.runtime_seconds_,classifier_seconds=classifier_seconds,peak_memory_mb=peak/(1024**2),redundancy=subset_redundancy(Xtr,mask),parameters=json.dumps(dfa_parameters,sort_keys=True),selected_indices=json.dumps(features.tolist()),**{k:float(knn[k]) for k in ["accuracy","balanced_accuracy","f1_score","mcc","roc_auc","pr_auc","sensitivity","specificity"]})
          rows.append(row); pd.DataFrame(rows).to_csv(raw_path,index=False)
          safe_label=label.replace("[","_").replace("]","").replace("=","-")
          fold_suffix="" if fold=="repeated_stratified_holdout" else f"_{fold}"
          selector.history_frame().assign(dataset=dataset,method=label,seed=seed,fold=fold).to_csv(root/"raw"/f"trace_{cfg['experiment']}_{dataset}_{safe_label}_{seed}{fold_suffix}.csv",index=False)
    raw=pd.DataFrame(rows)
    if "interaction_quality" not in raw: raw["interaction_quality"]=np.nan
    for index,row in raw[raw.interaction_quality.isna()].iterrows():
        X,y=load_data(row.dataset,cfg)
        matching=[item for item in evaluation_splits(X,y,cfg) if item[0]==int(row.seed) and item[1]==str(row.fold)]
        if not matching: raise ValueError(f"Cannot reconstruct split for {row.dataset}/{row.seed}/{row.fold}")
        _,_,_,train_indices,_=matching[0]; Xtr,ytr=X[train_indices],y[train_indices]
        mask=np.zeros(X.shape[1],bool); mask[np.asarray(json.loads(row.selected_indices),int)]=True
        raw.loc[index,"interaction_quality"]=subset_interaction_quality(Xtr,ytr,mask,int(row.seed))
    for (dataset,method),g in raw.groupby(["dataset","method"]):
        total=int(g.total_features.iloc[0]) if "total_features" in g and pd.notna(g.total_features.iloc[0]) else int(round(g.selected_features.iloc[0]/g.feature_ratio.iloc[0]))
        ms=np.zeros((len(g),total),bool)
        for row_index, value in enumerate(g.selected_indices): ms[row_index,np.asarray(json.loads(value),int)]=True
        raw.loc[g.index,"total_features"]=total; raw.loc[g.index,"jaccard_stability"]=compute_jaccard_stability(ms); raw.loc[g.index,"nogueira_stability"]=compute_nogueira_stability(ms)
    raw.to_csv(raw_path,index=False); agg=aggregate(raw); agg.to_csv(root/"aggregated"/f"{cfg['experiment']}.csv",index=False); stats=statistics(raw); stats.to_csv(root/"statistics"/f"{cfg['experiment']}.csv",index=False)
    shown=["dataset","method","balanced_accuracy_mean","f1_score_mean","mcc_mean","selected_features_mean","jaccard_stability_mean","redundancy_mean","runtime_seconds_mean"]
    table=agg[[c for c in shown if c in agg]].round(4).to_csv(index=False)
    report_name="ablation_report.md" if cfg["experiment"] in {"ablation","final_ablation"} else f"{cfg['experiment']}_report.md"
    (root/report_name).write_text("# Experiment report\n\n```csv\n"+table+"```\n\nStatistical results are exploratory unless run counts and datasets meet the registered protocol.\n",encoding="utf-8")
    print(agg.to_string(index=False)); return raw
if __name__=="__main__": main()
