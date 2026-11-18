"""Unified historical game tables: one row per game with scores + market lines."""
import pandas as pd,numpy as np,json,sys,glob,re,difflib
S=sys.argv[1];D=S+'/dl/';W=S+'/work/'
# ---------- NBA
tb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D+'hoop/nba_tb_*.parquet'))])
def games_from_box(tb):
    tb=tb.copy()
    for c in ['field_goals_attempted','free_throws_attempted','offensive_rebounds','total_turnovers','turnovers','three_point_field_goals_attempted','three_point_field_goals_made','field_goals_made','free_throws_made','defensive_rebounds','team_score']:
        tb[c]=pd.to_numeric(tb[c],errors='coerce')
    tb['tov']=tb.total_turnovers.where(tb.total_turnovers>0,tb.turnovers)
    tb['poss']=tb.field_goals_attempted-tb.offensive_rebounds+tb.tov+0.46*tb.free_throws_attempted
    k=['game_id','season','season_type','game_date','game_date_time','team_id','team_display_name','team_location','team_name','team_score','poss','field_goals_attempted','field_goals_made','three_point_field_goals_attempted','three_point_field_goals_made','free_throws_attempted','free_throws_made','offensive_rebounds','defensive_rebounds','tov']
    h=tb[tb.team_home_away=='home'][k];a=tb[tb.team_home_away=='away'][k[:1]+k[5:]]
    r={c:c for c in k[:5]}
    short=dict(team_id='id',team_display_name='name',team_location='loc',team_name='nick',team_score='pts',poss='poss',field_goals_attempted='fga',field_goals_made='fgm',three_point_field_goals_attempted='tpa',three_point_field_goals_made='tpm',free_throws_attempted='fta',free_throws_made='ftm',offensive_rebounds='orb',defensive_rebounds='drb',tov='tov')
    h=h.rename(columns={c:'h_'+v for c,v in short.items()});a=a.rename(columns={c:'a_'+v for c,v in short.items()})
    g=h.merge(a,on='game_id').drop_duplicates('game_id');g['game_id']=g.game_id.astype(int)
    g['date']=pd.to_datetime(g.game_date);return g
nba=games_from_box(tb)
sch=pd.concat([pd.read_parquet(f,columns=['game_id','neutral_site','status_period']) for f in sorted(glob.glob(D+'hoop/nba_sch_*.parquet'))]);sch['game_id']=sch.game_id.astype(int)
nba=nba.merge(sch.drop_duplicates('game_id'),on='game_id',how='left')
nba=nba[nba.h_id.astype(int)<=30][lambda d:d.a_id.astype(int)<=30]
sb=pd.DataFrame(json.load(open(D+'hoop/sbr_nba.json')));sb=sb[pd.to_numeric(sb.home_final,errors='coerce').notna()]
sb['date']=pd.to_datetime(sb.date.astype(int).astype(str),errors='coerce');sb['h_pts']=sb.home_final.astype(float);sb['a_pts']=sb.away_final.astype(float)
sb=sb.rename(columns=dict(home_open_spread='o_spread',home_close_spread='c_spread',open_over_under='o_total',close_over_under='c_total',home_close_ml='c_hml',away_close_ml='c_aml'))
sb=sb[['date','h_pts','a_pts','o_spread','c_spread','o_total','c_total','c_hml','c_aml']].drop_duplicates(['date','h_pts','a_pts'],keep=False)
nba=nba.merge(sb,on=['date','h_pts','a_pts'],how='left')
# SBR spread signs are occasionally flipped: trust the moneyline favourite
for c in['o_spread','c_spread']:
    nba.loc[nba[c].abs()>30,c]=np.nan
m=nba.c_hml.notna()&(nba.c_spread.abs()>=1.5)&(np.sign(nba.c_spread)!=np.sign(nba.c_hml-nba.c_aml))
print('nba flipped',m.sum()); nba.loc[m,'c_spread']*=-1
m=(nba.o_spread-nba.c_spread).abs()>(nba.o_spread+nba.c_spread).abs()+4; print('open flips',m.sum()); nba.loc[m,'o_spread']=np.nan
for c in['o_total','c_total']: nba.loc[~nba[c].between(160,270),c]=np.nan
cl=pd.read_parquet(D+'hoop/nba_close.parquet');cl['game_id']=cl.game_id.astype(int)
nba=nba.merge(cl[['game_id','home_point','over_under']],on='game_id',how='left')
nba['c_spread']=nba.c_spread.fillna(nba.home_point);nba['c_total']=nba.c_total.fillna(nba.over_under)
nba=nba.sort_values(['date','game_id']);nba.to_parquet(W+'nba.parquet')
print(nba.groupby('season')[['game_id','o_spread','c_spread','c_total','c_hml']].count())
# ---------- NCAAB
mb=games_from_box(pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D+'hoop/mbb_tb_*.parquet'))]))
sch=pd.concat([pd.read_parquet(f,columns=['game_id','neutral_site','conference_competition','status_period']) for f in sorted(glob.glob(D+'hoop/mbb_sch_*.parquet'))]);sch['game_id']=sch.game_id.astype(int)
mb=mb.merge(sch.drop_duplicates('game_id'),on='game_id',how='left')
z=pd.read_csv(D+'zachmayer_point_spreads/r/data/spreads_and_totals.csv');z['date']=pd.to_datetime(z.game_date)
z['spread']=pd.to_numeric(z.spread,errors='coerce');z['total']=pd.to_numeric(z.total,errors='coerce');z=z[z.spread.notna()]
# learn covers-name -> espn id by schedule co-occurrence voting
from collections import Counter,defaultdict
byd=defaultdict(set)
for r in mb.itertuples(): byd[r.date].update([r.h_id,r.a_id])
cnt=defaultdict(Counter);n=Counter()
for r in z.itertuples():
    for t in (r.home_team,r.away_team):
        n[t]+=1;cnt[t].update(byd.get(r.date,()))
tot=Counter()
for d,s in byd.items(): tot.update(s)
nm=dict(zip(mb.h_id,mb.h_loc));nm.update(dict(zip(mb.a_id,mb.a_loc)))
mp={}
for t in cnt:
    c=cnt[t].most_common(6)
    if not c: continue
    best=max(c,key=lambda x:x[1]/n[t]+0.25*difflib.SequenceMatcher(None,t.lower(),str(nm[x[0]]).lower()).ratio())
    if best[1]/n[t]>0.6: mp[t]=best[0]
z['h']=z.home_team.map(mp);z['a']=z.away_team.map(mp);z=z.dropna(subset=['h','a'])
k=z[['date','h','a','spread','total']].drop_duplicates(['date','h','a'])
m1=mb.merge(k.rename(columns=dict(h='h_id',a='a_id',spread='c_spread',total='c_total')),on=['date','h_id','a_id'],how='left')
k2=k.assign(spread=-k.spread).rename(columns=dict(h='a_id',a='h_id',spread='s2',total='t2'))
m1=m1.merge(k2,on=['date','h_id','a_id'],how='left');m1['c_spread']=m1.c_spread.fillna(m1.s2);m1['c_total']=m1.c_total.fillna(m1.t2)
m1.loc[~m1.c_total.between(95,200),'c_total']=np.nan
m1=m1.sort_values(['date','game_id']);m1.to_parquet(W+'ncaab.parquet')
print(m1.groupby('season')[['game_id','c_spread','c_total']].count())
x=m1.dropna(subset=['c_spread']);print('ncaab resid sd',(x.h_pts-x.a_pts+x.c_spread).std(),(x.h_pts-x.a_pts+x.c_spread).mean())
x=nba.dropna(subset=['c_spread']);print((x.h_pts-x.a_pts+x.c_spread).groupby(x.season).agg(['mean','std']).round(2).T)
print((x.h_pts+x.a_pts-x.c_total).groupby(x.season).agg(['mean','std']).round(2).T)
