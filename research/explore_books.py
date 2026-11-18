import pandas as pd,numpy as np,sys,json
from scipy import stats
import statsmodels.api as sm
S=sys.argv[1];W=S+'/work/'
D=json.load(open(W+'dist.json'))
lg=lambda p:np.log(p/(1-p));sg=lambda z:1/(1+np.exp(-z))
class Dist:
    def __init__(s,k):
        d=D[k];s.m=k.endswith('margin');nfl=k.startswith('NFL')
        s.grid=np.arange(-80,81) if s.m else np.arange(0,341)
        K=len(d['mult']);mult=np.array(d['mult'])
        keys=np.minimum(abs(s.grid),K-1) if s.m else (np.where((s.grid>=20)&(s.grid<=66),s.grid-19,48) if nfl else 0*s.grid)
        s.mus=np.arange(-45,45.01,.05) if s.m else (np.arange(25,70,.05) if nfl else np.arange(100,280,.05))
        p=stats.t.pdf((s.grid[None]-s.mus[:,None])/d['scale'],d['df'])*mult[keys][None];s.P=p/p.sum(1,keepdims=True);s.c={}
    def curve(s,x):
        if x not in s.c:
            w=s.P[:,s.grid>x].sum(1);l=s.P[:,s.grid<x].sum(1);s.c[x]=(w,l,w/(w+l))
        return s.c[x]
    def inv(s,x,p): return np.interp(p,s.curve(x)[2],s.mus)
    def pr(s,x,mu): w,l,_=s.curve(x);return np.interp(mu,s.mus,w),np.interp(mu,s.mus,l)
def wide(d):
    d=d.copy();d['side']=np.where(d.market_key=='totals',np.where(d.outcome_name=='Over','A','B'),np.where(d.outcome_name==d.home_team,'A','B'))
    k=['sport','game_id','commence_time','home_team','away_team','fetch_timestamp','bookmaker_key','market_key']
    d['pt']=d.point.fillna(0)*np.where((d.market_key=='spreads')&(d.side=='B'),-1,1)
    w=d.pivot_table(index=k+['pt'],columns='side',values='price_decimal',aggfunc='first').dropna().reset_index()
    w['ov']=1/w.A+1/w.B;w=w[w.ov.between(.985,1.12)];w['p']=(1/w.A)/w.ov
    w['bal']=(w.p-.5).abs();w=w.sort_values('bal').drop_duplicates(k)  # main line only
    return w
d=pd.read_parquet(W+'abet.parquet');w=wide(d[d.sport.isin(['nba','nfl'])])
w['t']=pd.to_datetime(w.fetch_timestamp,format='ISO8601');w['c']=pd.to_datetime(w.commence_time);w=w[w.t<w.c]
DD={(s,m):Dist(s.upper()+'_'+m) for s in('nba','nfl') for m in('margin','total')}
def mu(r):
    if r.market_key=='h2h': return lg(r.p)
    if r.market_key=='spreads': return DD[r.sport,'margin'].inv(-r.pt,r.p)
    return DD[r.sport,'total'].inv(r.pt,r.p)
w['mu']=[mu(r) for r in w.itertuples()]
w.to_parquet(W+'abet_wide.parquet')
w['loc_t']=w.t.dt.tz_convert('America/Los_Angeles');w['gd']=w.c.dt.tz_convert('America/Los_Angeles').dt.date
mor=w[(w.loc_t.dt.date==w.gd)&(w.loc_t.dt.hour.between(7,11))].copy()
mor['gap']=(mor.loc_t.dt.hour*60+mor.loc_t.dt.minute-540).abs();ts=mor.groupby('game_id').gap.transform('min');mor=mor[mor.gap==ts]
last=w[(w.c-w.t)<pd.Timedelta('150min')];last=last[last.t==last.groupby('game_id').t.transform('max')]
M=mor.pivot_table(index=['sport','game_id','market_key'],columns='bookmaker_key',values='mu')
C=last.pivot_table(index=['sport','game_id','market_key'],columns='bookmaker_key',values='mu')
sharp=['pinnacle','novig','prophetx','kalshi','lowvig']
C['close']=C[['pinnacle','novig','prophetx','kalshi','lowvig','draftkings','fanduel','betmgm']].median(axis=1)
X=M.join(C[['close','pinnacle']].rename(columns={'pinnacle':'pc'}),how='inner')
gap=(mor.groupby('game_id').c.first()-mor.groupby('game_id').t.first()).dt.total_seconds()/3600
for (sp,mk),x in X.groupby(level=[0,2]):
    print('==',sp,mk,len(x),x[['fanduel','pinnacle','novig','prophetx','kalshi','lowvig','draftkings','betmgm','betopenly','polymarket_us']].count().to_dict())
    for tgt in('close','pc'):
        print(' rmse to',tgt,{b:round(float(np.sqrt(((x[b]-x[tgt])**2).mean())),3) for b in ['fanduel','pinnacle','novig','prophetx','kalshi','lowvig','draftkings','betmgm','betopenly','polymarket_us']})
    bs=['pinnacle','novig','kalshi','prophetx','lowvig','draftkings','betmgm']
    y=x.dropna(subset=['fanduel','pc']+bs)
    r=sm.OLS(y.close-y.fanduel,pd.DataFrame({b:y[b]-y.fanduel for b in bs})).fit();print(' n',len(y),r.params.round(2).to_dict(),r.bse.round(2).to_dict())
    y=x.dropna(subset=['fanduel','close']).copy();y['oth']=y[bs].mean(axis=1);y=y.dropna(subset=['oth'])
    r=sm.OLS(y.close-y.fanduel,(y.oth-y.fanduel)).fit();print(' mean-of-others coef',r.params.round(2).tolist(),r.bse.round(2).tolist(),'sd dev',(y.oth-y.fanduel).std().round(3),'sd move',(y.close-y.oth).std().round(3))
X.to_parquet(W+'abet_X.parquet');mor.to_parquet(W+'abet_mor.parquet')
