import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from losslens.loss_metrics import development_triangle, age_to_age, loss_ratios

def test_future_development_is_unknown_not_zero():
    claims=pd.DataFrame({'loss_date':['2025-01-01','2025-04-01'],
        'report_date':['2025-04-15','2025-07-01'],'paid_amount':[100,200]})
    tri=development_triangle(claims,3,as_of='2025-06-30')
    assert tri.loc[pd.Period('2025Q1'),1]==100
    assert pd.isna(tri.loc[pd.Period('2025Q2'),1])
    assert pd.isna(tri.loc[pd.Period('2025Q1'),2])

def test_factors_compare_only_matched_observed_cells():
    tri=pd.DataFrame({0:[100,1000],1:[200,float('nan')]})
    assert age_to_age(tri)['0->1']==2

def test_paid_calendar_and_accident_year_are_distinct():
    claims=pd.DataFrame({'loss_date':['2024-12-01'],'report_date':['2025-01-02'],'paid_amount':[100]})
    policies=pd.DataFrame({'effective_date':['2024-01-01','2025-01-01'],'annual_premium':[200,200]})
    assert loss_ratios(claims,policies).loc[2024,'loss_ratio']==0.5
    assert loss_ratios(claims,policies,'calendar').loc[2025,'loss_ratio']==0.5

def test_missing_premium_is_not_an_infinite_ratio():
    claims=pd.DataFrame({'loss_date':['2025-01-01'],'report_date':['2025-01-02'],'paid_amount':[100]})
    policies=pd.DataFrame({'effective_date':['2024-01-01'],'annual_premium':[200]})
    assert pd.isna(loss_ratios(claims,policies).loc[2025,'loss_ratio'])
