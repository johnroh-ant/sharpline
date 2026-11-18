import pandas as pd,numpy as np,sys
S=sys.argv[1];D=S+'/dl/';W=S+'/work/'
F=D+'ethanbell528-cmd_fda-project-1/r/data/'
m=pd.read_csv(F+'mlb.csv',low_memory=False);m=m[m.season>=2015]
st=pd.read_csv(F+'mlb_starters.csv')
m=m.merge(st,on=['game_id','team'],how='left')
# neutral rows: take first as away
m['ha']=m.home_away
for gid,g in m[m.home_away=='N'].groupby('game_id'):
    m.loc[g.index[0],'ha']='A';m.loc[g.index[1],'ha']='H'
h=m[m.ha=='H'][['game_id','date','season','game_type','team','opponent','score_for','score_against','decided_in','starter_name','starter_id','total_line','moneyline','home_away']].rename(columns=dict(team='h',opponent='a',score_for='h_pts',score_against='a_pts',starter_name='h_sp',starter_id='h_spid',total_line='sbr_total',moneyline='sbr_hml'))
a=m[m.ha=='A'][['game_id','starter_name','starter_id','moneyline']].rename(columns=dict(starter_name='a_sp',starter_id='a_spid',moneyline='sbr_aml'))
g=h.merge(a,on='game_id');g['date']=pd.to_datetime(g.date);g['neutral']=g.home_away=='N'
o=pd.read_csv(D+'tanhungle_mlb-reference-data/r/odds_flat.csv');o['date']=pd.to_datetime(o.date)
print(sorted(set(o.home)-set(g.h)),sorted(set(g.h)-set(o.home)))
mp={'AZ':'ARI','CHW':'CWS','KCR':'KC','SDP':'SD','SFG':'SF','TBR':'TB','WSN':'WSH','WAS':'WSH','OAK':'ATH','CWS':'CWS'}
inv={}
for t in set(o.home)-set(g.h): inv[t]=mp.get(t,t)
print(inv)
o['h']=o.home.replace(inv);o['a']=o.away.replace(inv)
o=o.rename(columns=dict(home_score='h_pts',away_score='a_pts'))
key=['date','h','a','h_pts','a_pts']
o=o.drop_duplicates(key+['market','sportsbook'],keep=False)
o=o.merge(g[key+['game_id']].drop_duplicates(key,keep=False),on=key,how='inner')
print('matched games',o.game_id.nunique())
o.to_parquet(W+'mlb_odds.parquet');g.sort_values(['date','game_id']).to_parquet(W+'mlb.parquet')
print(o[o.market=='pointspread'].head(3).T); print(o[o.market=='totals'].head(2).T)
# NFL
n=pd.read_csv(D+'nfl/games.csv');L=pd.read_csv(D+'greerreNFL_nfelomarket_data/r/Data/lines.csv')
print(L.groupby('season')[['home_spread_open','home_spread_open_price','home_ml_open','total_line_open']].count().tail(20).T)
print(L.groupby('season').home_spread_open_source.agg(lambda s:s.value_counts().to_dict()).tail(20).to_string())
print((L.home_spread_open!=L.home_spread_last).groupby(L.season).mean().round(2).to_dict())
