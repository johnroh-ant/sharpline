import pandas as pd,glob,sys
S=sys.argv[1]
fr=[]
for f in sorted(glob.glob(S+'/dl/hf/abet/*.gz')):
    d=pd.read_json(f,lines=True,compression='gzip',dtype={'commence_time':str,'fetch_timestamp':str},convert_dates=False)
    d=d[(d.is_live_game==0)&d.market_key.isin(['h2h','spreads','totals'])&~d.bookmaker_key.str.startswith('_')]
    fr.append(d[['sport','game_id','commence_time','home_team','away_team','fetch_timestamp','bookmaker_key','market_key','outcome_name','price_american','price_decimal','point','bet_limit']])
d=pd.concat(fr);d.to_parquet(S+'/work/abet.parquet')
print(d.shape);print(d.groupby(['sport','bookmaker_key']).size().unstack(0).fillna(0).astype(int).to_string())
print(d.groupby('sport').game_id.nunique())
x=d[d.sport=='nba'].fetch_timestamp.drop_duplicates().str[:16].sort_values();print(len(x));print(x[x.str[:10]=='2026-11-10'].tolist())
