import sys,os,sqlite3,datetime as dt,json
S=sys.argv[1]; which=sys.argv[2]  # opening|closing for MLB
sys.path.insert(0,S+'/dl/old/repo/sports-betting')
os.environ['SPORTSBOOK_DB']=':memory:';os.environ['SPORTSBOOK_DATA_DIR']=S+'/work/olddata'
import pandas as pd,numpy as np
from sportsbook import db,betting,mathutils
from sportsbook.models import MODELS
conn=db.connect()
W=S+'/work/'
nn=lambda v: None if pd.isna(v) else v
ii=lambda v: None if pd.isna(v) else int(v)
G=[]
m=pd.read_parquet(W+'mlb.parquet');m=m[(m.season>=2020)&(m.date<'2025-08-17')]
o=pd.read_parquet(W+'mlb_odds.parquet');o=o[o.sportsbook=='fanduel'].set_index(['game_id','market'])
p=which+'_'
for r in m.itertuples():
    L=None
    try:
        a,b,c=o.loc[(r.game_id,'moneyline')],o.loc[(r.game_id,'pointspread')],o.loc[(r.game_id,'totals')]
        L=dict(home_spread=nn(b[p+'home_spread']),home_spread_price=ii(b[p+'home_odds']),away_spread_price=ii(b[p+'away_odds']),total=nn(c[p+'total']),over_price=ii(c[p+'over_odds']),under_price=ii(c[p+'under_odds']),home_ml=ii(a[p+'home_odds']),away_ml=ii(a[p+'away_odds']))
        if any(v is None or v==0 for v in L.values()) or abs(L['home_spread'])!=1.5 or max(abs(L['home_ml']),abs(L['away_ml']))>500: L=None
    except KeyError: pass
    G.append(dict(sport='MLB',date=r.date,h=r.h,a=r.a,hs=int(r.h_pts),as_=int(r.a_pts),neutral=r.neutral,hp=nn(r.h_sp),ap=nn(r.a_sp),L=L if r.season>=2021 else None))
for sp,f in(('NBA','nba'),('NCAAB','ncaab')):
    d=pd.read_parquet(W+f+'.parquet');d=d[(d.season>=2020)]
    for r in d.itertuples():
        L=None
        if r.season>=2021 and not pd.isna(r.c_spread):
            L=dict(home_spread=r.c_spread,home_spread_price=-110,away_spread_price=-110,total=nn(r.c_total),over_price=-110,under_price=-110,home_ml=ii(getattr(r,'c_hml',None)) if sp=='NBA' else None,away_ml=ii(getattr(r,'c_aml',None)) if sp=='NBA' else None)
            if L['home_ml']==0 or L['away_ml']==0: L['home_ml']=L['away_ml']=None
        G.append(dict(sport=sp,date=r.date,h=str(r.h_id),a=str(r.a_id),hs=int(r.h_pts),as_=int(r.a_pts),neutral=bool(r.neutral_site),hp=None,ap=None,L=L))
n=pd.read_csv(S+'/dl/nfl/games.csv');n=n[(n.season>=2019)&n.result.notna()]
for r in n.itertuples():
    L=dict(home_spread=-r.spread_line,home_spread_price=ii(r.home_spread_odds),away_spread_price=ii(r.away_spread_odds),total=r.total_line,over_price=ii(r.over_odds),under_price=ii(r.under_odds),home_ml=ii(r.home_moneyline),away_ml=ii(r.away_moneyline)) if r.season>=2021 else None
    G.append(dict(sport='NFL',date=pd.Timestamp(r.gameday),h=r.home_team,a=r.away_team,hs=int(r.home_score),as_=int(r.away_score),neutral=r.location=='Neutral',hp=None,ap=None,L=L))
G.sort(key=lambda g:g['date'])
models={s:MODELS[s](conn) for s in MODELS}
last={};out=[];trace=[]
from itertools import groupby
for day,gs in groupby(G,key=lambda g:g['date']):
    gs=list(gs);ts=day.strftime('%Y-%m-%dT20:00:00Z');c=[];preds=[]
    for i,g in enumerate(gs):
        M=models[g['sport']];game=dict(game_id=i,home_team=g['h'],away_team=g['a'],commence_time=ts,neutral_site=g['neutral'])
        rest=lambda t: None if (g['sport'],t) not in last else max(0,min((day-last[(g['sport'],t)]).days-1,30))
        pr=None
        if g['L']:
            ctx=dict(home_rest_days=rest(g['h']),away_rest_days=rest(g['a']),home_pitcher=g['hp'],away_pitcher=g['ap'])
            pred=M.predict(game,ctx);game['pred']=pred;M._pmf_cache.clear()
            s=betting.side_candidates(M,game,g['L']);t=betting.total_candidates(M,game,g['L'])
            for x in (s,t):
                if x: x['g']=g;c.append(x)
            bm=s['blended_margin'] if s else pred['margin']
            pr=dict(features=dict(x=pred['features'],ctx=dict(home_pitcher=g['hp'],away_pitcher=g['ap']),raw=dict(margin=pred['margin'],total=pred['total'],wp=M.win_prob_raw(bm))),pred_home_margin=bm,pred_total=pred['total'],pred_home_wp=.5,market_home_spread=g['L']['home_spread'],market_total=g['L']['total'],market_home_ml=g['L']['home_ml'],market_away_ml=g['L']['away_ml'])
            trace.append(dict(date=day,sport=g['sport'],a=M.alpha_margin,at=M.alpha_total,cal_a=M.cal_a,cal_b=M.cal_b,raw=pred['margin'],mkt=M.market_implied_margin(g['L']),res=g['hs']-g['as_'],rawt=pred['total'],mktt=g['L']['total'],rest_=g['hs']+g['as_']))
        preds.append(pr)
    card=set(map(id,betting.pick_card(c,10)))
    for x in c:
        g=x['g'];mg=g['hs']-g['as_'];tt=g['hs']+g['as_'];home=x['selection']==g['h']
        if x['market']=='spread': v=(mg if home else -mg)+x['line']
        elif x['market']=='total': v=(tt-x['line'])*(1 if x['selection']=='Over' else -1)
        else: v=mg if home else -mg
        d=mathutils.american_to_decimal(x['price'])
        out.append(dict(date=day,sport=g['sport'],market=x['market'],sel='Over' if x['selection']=='Over' else 'Under' if x['selection']=='Under' else 'home' if home else 'away',price=x['price'],p=x['win_prob'],edge=x['edge'],conf=x['confidence'],card=id(x) in card,res=np.sign(v),unit=(d-1) if v>0 else (0 if v==0 else -1)))
    for g,pr in zip(gs,preds):
        models[g['sport']].learn(dict(home_team=g['h'],away_team=g['a'],home_score=g['hs'],away_score=g['as_'],commence_time=ts,neutral_site=g['neutral'],home_pitcher=g['hp'],away_pitcher=g['ap']),pr)
        last[(g['sport'],g['h'])]=day;last[(g['sport'],g['a'])]=day
o=pd.DataFrame(out);o.to_parquet(W+f'old_{which}.parquet');pd.DataFrame(trace).to_parquet(W+f'oldtrace_{which}.parquet')
b=o[o.card].copy();b['pnl']=b.unit*b.conf
def s(x): return pd.Series(dict(n=len(x),win=(x.res>0).sum()/max(1,(x.res!=0).sum()),claimed_ev=x.edge.mean(),avg_p=x.p.mean(),staked=x.conf.sum(),pnl=x.pnl.sum(),roi=x.pnl.sum()/x.conf.sum()))
print(s(b));print(b.groupby('sport').apply(s).round(3));print(b.groupby(['sport','market','sel']).apply(s).round(3));print(b.groupby(b.date.dt.year).apply(s).round(3));print(b.groupby(pd.cut(b.conf,[0,25,50,75,99,100])).apply(s).round(3))
t=pd.DataFrame(trace);print(t.groupby(['sport',t.date.dt.year])[['a','at','cal_a','cal_b']].mean().round(2))
for sp,x in t.groupby('sport'):
    x=x.dropna();print(sp,'optimal weight margin',np.polyfit(x.raw-x.mkt,x.res-x.mkt,1)[0].round(3),'total',np.polyfit(x.rawt-x.mktt,x.rest_-x.mktt,1)[0].round(3))
