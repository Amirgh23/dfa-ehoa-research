"""Generate deterministic paper-ready pilot figures and CSV/Markdown/LaTeX tables."""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment",default="pilot"); ap.add_argument("--results",type=Path,default=Path("results")); a=ap.parse_args()
    raw=pd.read_csv(a.results/"raw"/f"{a.experiment}.csv"); agg=pd.read_csv(a.results/"aggregated"/f"{a.experiment}.csv"); figures=a.results/"figures"; tables=a.results/"tables"
    for metric,name,ylabel in [("balanced_accuracy","performance_boxplot","Balanced accuracy"),("runtime_seconds","runtime_comparison","Runtime (s)")]:
        fig,ax=plt.subplots(figsize=(9,5)); raw.boxplot(column=metric,by=["dataset","method"],rot=35,ax=ax); fig.suptitle(""); ax.set_title(ylabel+" by method"); ax.set_ylabel(ylabel); save(fig,figures/f"{a.experiment}_{name}.png")
    fig,ax=plt.subplots(figsize=(7,5));
    for method,g in raw.groupby("method"): ax.scatter(g.selected_features,g.balanced_accuracy,label=method,alpha=.8)
    ax.set(xlabel="Selected feature count",ylabel="Balanced accuracy",title="Predictive performance vs subset size"); ax.legend(); ax.grid(alpha=.2); save(fig,figures/f"{a.experiment}_accuracy_vs_features.png")
    fig,ax=plt.subplots(figsize=(7,5));
    for method,g in raw.groupby("method"): ax.scatter(g.jaccard_stability,g.balanced_accuracy,label=method,alpha=.8)
    ax.set(xlabel="Mean Jaccard stability",ylabel="Balanced accuracy",title="Stability vs predictive performance"); ax.legend(); ax.grid(alpha=.2); save(fig,figures/f"{a.experiment}_stability_vs_performance.png")
    traces=list((a.results/"raw").glob(f"trace_{a.experiment}_*_DFA-EHOA_*.csv"))
    if traces:
        trace=pd.read_csv(traces[0]); columns=[("fitness","Convergence"),("nogueira_stability","Stability"),("population_diversity","Population diversity"),("sweep_factor","Sweep factor"),("inertia","Inertia"),("selected_features","Selected features")]
        fig,axes=plt.subplots(3,2,figsize=(11,10))
        for ax,(column,title) in zip(axes.flat,columns): ax.plot(trace.iteration,trace[column]); ax.set(title=title,xlabel="Iteration",ylabel=column); ax.grid(alpha=.2)
        save(fig,figures/f"{a.experiment}_dfa_controller_trace.png")
    agg.to_csv(tables/f"{a.experiment}_overall.csv",index=False)
    (tables/f"{a.experiment}_overall.md").write_text("```csv\n"+agg.round(4).to_csv(index=False)+"```\n",encoding="utf-8")
    rounded=agg.round(4); cols=list(rounded.columns)
    latex="\\begin{tabular}{"+"l"*len(cols)+"}\n"+" & ".join(cols)+" \\\\\n\\hline\n"
    latex+="\n".join(" & ".join(str(v).replace("_","\\_") for v in row)+" \\\\" for row in rounded.astype(str).to_numpy())
    latex+="\n\\end{tabular}\n"
    (tables/f"{a.experiment}_overall.tex").write_text(latex,encoding="utf-8")
if __name__=="__main__": main()
