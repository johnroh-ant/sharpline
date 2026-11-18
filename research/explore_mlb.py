import pandas as pd,numpy as np,sys
import statsmodels.api as sm
S=sys.argv[1];W=S+'/work/'
use=['game.key','game.date','team.alignment','team.key','person.key','season.phase','B_PA','B_AB','B_R','B_H','B_2B','B_3B','B_HR','B_BB','B_IBB','B_HP','B_SO','B_SF','P_G','P_GS','P_OUT','P_TBF','P_H','P_HR','P_R','P_ER','P_BB','P_HP','P_SO','P_PITCH']
p=pd.concat([pd.read_csv(f'{S}/dl/rs/playing-{y}.csv',usecols=use,low_memory=False) for y in range(2017,2026)]);p.columns=[c.replace('.','_') for c in p.columns]
p['date']=pd.to_datetime(p.game_date);p['yr']=p.date.dt.year;p=p[p.season_phase!='A'].fillna(0)
p['woba_n']=.69*(p.B_BB-p.B_IBB)+.72*p.B_HP+.89*(p.B_H-p.B_2B-p.B_3B-p.B_HR)+1.27*p.B_2B+1.62*p.B_3B+2.1*p.B_HR;p['woba_d']=p.B_AB+p.B_BB-p.B_IBB+p.B_SF+p.B_HP
def cum(d,key,cols,prev_w):
    """season-to-date before game day + prev_w * previous full season"""
    t=d.groupby([key,'yr','date'])[cols].sum().reset_index().sort_values('date')
    c=t.groupby([key,'yr'])[cols].cumsum()-t[cols];t[cols]=c
    s=d.groupby([key,'yr'])[cols].sum().reset_index();s['yr']+=1
    t=t.merge(s,on=[key,'yr'],how='left',suffixes=('','_p')).fillna(0)
    for k in cols:t[k]=t[k]+prev_w*t[k+'_p']
    return t[[key,'date']+cols]
pc=['P_TBF','P_SO','P_BB','P_HP','P_HR','P_R','P_OUT','P_GS','P_G','P_H']
P=p[p.P_G>0];sp=cum(P,'person_key',pc,.6);bp=cum(P[P.P_GS==0],'team_key',pc,.3);of=cum(p[p.B_PA>0],'team_key',['woba_n','woba_d','B_R','B_PA','B_SO'],.35)
lg=p.groupby('yr')[pc+['woba_n','woba_d','B_R','B_PA']].sum();lg.index+=1;lg.loc[2017]=lg.loc[2018]
def pit(t,K):
    y=t.date.dt.year;r=lambda c,k:(t[c]+k*(lg[c]/lg.P_TBF).reindex(y).values)/(t.P_TBF+k)-(lg[c]/lg.P_TBF).reindex(y).values
    t['fip']=38*(-.28*r('P_SO',70)+.33*(r('P_BB',170)+r('P_HP',600))+1.4*r('P_HR',1300))*1.0   # runs/9 above avg, skill-only
    t['ra']=38*r('P_R',K);t['ipgs']=((t.P_OUT+16*5)/(t.P_G+5)).clip(6,20);return t
sp=pit(sp,500);bp=pit(bp,1500)
y=of.date.dt.year;of['woba']=((of.woba_n+800*(lg.woba_n/lg.woba_d).reindex(y).values)/(of.woba_d+800)-(lg.woba_n/lg.woba_d).reindex(y).values)/1.22*38;of['rs']=((of.B_R+1200*(lg.B_R/lg.B_PA).reindex(y).values)/(of.B_PA+1200)-(lg.B_R/lg.B_PA).reindex(y).values)*38
# bullpen pitches last 2 days
u=P[P.P_GS==0].groupby(['team_key','date']).P_TBF.sum()
G=P[P.P_GS>0][['game_key','date','team_alignment','team_key','person_key']].drop_duplicates(['game_key','team_alignment'])
G=G.merge(sp[['person_key','date','fip','ra','ipgs']],on=['person_key','date']).merge(bp[['team_key','date','fip','ra']].rename(columns=dict(fip='bfip',ra='bra')),on=['team_key','date'],how='left').merge(of[['team_key','date','woba','rs']],on=['team_key','date'],how='left').fillna(0)
G['tired']=[u.get((t,d-pd.Timedelta(days=1)),0)+.5*u.get((t,d-pd.Timedelta(days=2)),0) for t,d in zip(G.team_key,G.date)]
G['sh']=G.ipgs/27;G['pf']=G.sh*G.fip+(1-G.sh)*G.bfip;G['pr']=G.sh*G.ra+(1-G.sh)*G.bra
H=G[G.team_alignment==1].set_index('game_key').add_prefix('h_').join(G[G.team_alignment==0].set_index('game_key').add_prefix('a_'),how='inner')
g=pd.read_parquet(W+'mlb.parquet').set_index('game_id').join(H,how='inner')
o=pd.read_parquet(W+'mlb_odds.parquet');ip=lambda a:np.where(a>0,100/(a+100),-a/(-a+100))
for w in('opening','closing'):
    d=o[o.market=='moneyline'];d=d[(d[w+'_home_odds'].abs().between(100,450))&(d[w+'_away_odds'].abs().between(100,450))];g['z_'+w]=pd.Series(np.log(ip(d[w+'_home_odds'])/ip(d[w+'_away_odds'])),index=d.game_id.values).groupby(level=0).median()
    d=o[o.market=='totals'];d=d[d[w+'_total'].between(5,15)&d[w+'_over_odds'].abs().between(100,200)&d[w+'_under_odds'].abs().between(100,200)]
    g['T_'+w]=pd.Series(d[w+'_total'].values+2.2*np.log(ip(d[w+'_over_odds'])/ip(d[w+'_under_odds'])),index=d.game_id.values).groupby(level=0).median()
s=g.sbr_hml.abs().between(100,450)&g.sbr_aml.abs().between(100,450);g.loc[s&g.z_closing.isna(),'z_closing']=np.log(ip(g.sbr_hml)/ip(g.sbr_aml))[s&g.z_closing.isna()];g['T_closing']=g.T_closing.fillna(g.sbr_total)
g['sk']=(g.h_woba-g.a_woba)-(g.h_pf-g.a_pf);g['res']=(g.h_rs-g.a_rs)-(g.h_pr-g.a_pr);g['off']=g.h_woba-g.a_woba;g['spf']=-(g.h_sh*g.h_fip-g.a_sh*g.a_fip);g['bpf']=-((1-g.h_sh)*g.h_bfip-(1-g.a_sh)*g.a_bfip);g['spr']=-(g.h_sh*g.h_ra-g.a_sh*g.a_ra);g['rsd']=g.h_rs-g.a_rs;g['bpr']=-((1-g.h_sh)*g.h_bra-(1-g.a_sh)*g.a_bra);g['tired']=(g.h_tired-g.a_tired)/10
g['tsk']=g.h_woba+g.a_woba+g.h_pf+g.a_pf;g['tres']=g.h_rs+g.a_rs+g.h_pr+g.a_pr;g['ttired']=(g.h_tired+g.a_tired)/10;g['y']=(g.h_pts>g.a_pts)*1;g['tt']=g.h_pts+g.a_pts;g=g[g.season>=2018];g['mo']=g.date.dt.month
g.to_parquet(W+'mlb_f.parquet')
for w in('opening','closing'):
    for f in(['sk','res'],['off','rsd','spf','spr','bpf','bpr','tired']):
        x=g.dropna(subset=['z_'+w]);r=sm.Logit(x.y,sm.add_constant(x[['z_'+w]+f])).fit(disp=0);print(w,len(x),'  '.join(f'{k}={r.params[k]:.3f}({r.tvalues[k]:.1f})' for k in r.params.index))
    x=g.dropna(subset=['T_'+w]);x=x.assign(T=x['T_'+w]-8.7,apr=(x.mo<=4)*1.,sep=(x.mo>=9)*1.);r=sm.OLS((x.tt-x['T_'+w]).clip(-6,7),sm.add_constant(x[['T','tsk','tres','ttired','apr','sep']])).fit();print(w,'tot',len(x),'  '.join(f'{k}={r.params[k]:.3f}({r.tvalues[k]:.1f})' for k in r.params.index))
x=g.dropna(subset=['z_opening','z_closing']);r=sm.OLS(x.z_closing-x.z_opening,sm.add_constant(x[['z_opening','off','rsd','spf','spr','bpf','bpr','tired']])).fit();print('move',(x.z_closing-x.z_opening).std(),r.rsquared,'  '.join(f'{k}={r.params[k]:.3f}({r.tvalues[k]:.1f})' for k in r.params.index))
for w in('opening','closing'):
  for s in range(2022,2026):
    f=['z_'+w,'sk','res'];tr=g[g.season<s].dropna(subset=f);te=g[g.season==s].dropna(subset=f);r=sm.Logit(tr.y,sm.add_constant(tr[f])).fit(disp=0);pp=r.predict(sm.add_constant(te[f]));p0=1/(1+np.exp(-te[f[0]]))
    ll=lambda q:-(te.y*np.log(q)+(1-te.y)*np.log(1-q)).mean();print(w,s,round(ll(p0),5),round(ll(pp),5),'sd',(pp-p0).std().round(4))
