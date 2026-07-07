# Power BI report spec (build after the Python layer)

Import claims.csv + policies.csv + scores export. Four pages:
1. **Loss performance** — AY loss ratio trend, frequency/severity decomposition
   (measures: Frequency = Claims/Policies, Severity = Paid/Claims), premium vs
   incurred by state map.
2. **Development** — triangle as a matrix visual (acc quarter rows, dev quarter
   columns, cumulative paid), ATA factors as a line beneath.
3. **SIU referral queue** — table sorted by referral_score with score-family
   bars (rules/graph/anomaly/notes), drill-through to claim detail incl.
   adjuster note text; alert bookmark: top-20 queue.
4. **Ring explorer** — shared-entity table (phone/shop degree), decomposition
   tree: paid by shop > phone > claim.
RLS: state-level roles on a dim_state.
