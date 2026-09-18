"""Export loss analytics and referral evidence from the synthetic fixture book."""
from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from losslens.generate import generate,OUT
from losslens import triage,loss_metrics

def main():
    generate();claims=pd.read_csv(OUT/'claims.csv');policies=pd.read_csv(OUT/'policies.csv');notes=pd.read_csv(OUT/'adjuster_notes.csv')
    scores=triage.build_scores(claims,notes);out=ROOT/'outputs';out.mkdir(exist_ok=True)
    for basis in ['accident','calendar']:
        loss_metrics.loss_ratios(claims,policies,basis).to_csv(out/f'{basis}_year_paid_loss.csv')
    triangle=loss_metrics.development_triangle(claims)
    triangle.to_csv(out/'paid_triangle.csv');loss_metrics.age_to_age(triangle).to_csv(out/'development_factors.csv')
    scores.nlargest(50,'referral_score').to_csv(out/'referral_queue.csv',index=False)
    report={'data':'Synthetic policies and planted rings; not real-person fraud decisions.',
        'policies':len(policies),'claims':len(claims),'precision_at_50':triage.ablation(scores),
        'loss_definition':'Paid amounts; no reserves or IBNR. Full annual premium allocated to policy year.',
        'triangle_cutoff':str(pd.to_datetime(claims.report_date).max().date())}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__': main()
