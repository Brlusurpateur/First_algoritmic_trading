from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
import ssl #Importer le module ssl pour gérer la vérification du certificat SSL
import urllib.request
import sklearn
import math

# Supprimer les variables créées
# del ssl_context, urllib, symbols_list, start_date, end_date, sp500

#Ignorer la vérification du certificat SSL en définissant une politique de contexte SSL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

warnings.filterwarnings('ignore')

#Récupérer la table des actions du S&P 500
#Ici on récupère la liste de DataFrame de la page HTML et avec "[0]" on précise qu'on récupère le premier DataFrame. 
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

#Remplace dans la colonne Symbol les point par des tirés.
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

#Création et affectation des valeurs de la colonne Symbol du DataFrame sp500.
#tolist convertit les valeurs en une liste Python.
symbols_list = sp500['Symbol'].unique().tolist()

#Téléchargement des prix journalier de l'ensemble des indices du S&P500 d'il y a 8 ans à hier.
today = dt.date.today()
yesterday = today - dt.timedelta(days=1)
#end_date = yesterday.strftime('%Y-%m-%d')
end_date = '2023-09-27'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)
df = pd.DataFrame(yf.download(tickers=symbols_list, start=start_date, end=end_date))
#Stack est utilisé pour empiler les niveau de colonnes d'un DataFrame.
df= df.stack()

#lignes pour changer la syntaxe des noms de colonnes. On met tout en minuscule pour simplifier leur utilisation par la suite
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

#print(df)




######################## 2. Calculate features and technical indicators for each stock ########################

#Garman-Klass Volatility: Méthode de calcul de la volatilité.
#RSI: L'indice de force relative mesure la vitesse et l'ampleur des changements de prix récents d'un titre.
#Bollinger Bands: Outil d'analyse technique qui permet d'analyser la volatilité et la force de la tendance.
#ATR: Outil utilisé pour mesurer la volatilité. 
#MACD: Moving Average Convergence Divergence est un indicateur technique qui représente la différence entre deux moyennes mobiles exponentielles de périodes différentes (12 et 26 jours)
#Dollar Volume



#Garman-Klass Volatility

df['garman_klass_vol'] = (((np.log(df['high'])-np.log(df['low']))**2)/2)-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

#RSI: Relative Strength Index
#Ici on utilise "groupby" pour regrouper l'ensemble des données que nous avons par indice boursier d'où l'utilisation de "level=1" qui équivaut à "ticker". 
#"transform..." applique une transformation à chaque groupe de données. La fonction lambda calcule le RSI pour chaque groupe (indice) avec une longueur de fenêtre de 20jours.
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20)) 
#df.xs('KO', level=1)['rsi'].plot()
#plt.show()

#Bollinger Bands
# "pandas_ta..." calcule les bandes à partir des données de clôtures, qui sont transmises sous forme de logarithme naturel + 1 de 'x' avec un longueur de 20 jours.
# "transform" applique la fonction lambda à chaque groupe de données, la fonction lambda permet d'appliquer la fonction "pandas_ta.bbands" à chaque groupeby.
# ".iloc..." sélectionne la première colonne du DataFrame, qui correspond aux valeurs des bandes inférieur, middle et supérieur.
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2]) 

#ATR: Average True Rage
#"def compute_atr" définie la fonction compute_atr qui prend en arguments stock_data
#"atr = ..." utilise la fonction atr dans pandas_ta pour calculer l'atr à partir de 3 types de données stocké dans stock_data
#"return.." retourne: (atr - moyenne(atr)) / écart-type atr
def compute_atr(stock_data):
    atr = pandas_ta.atr(high = stock_data['high'],
                        low = stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

#MACD: Moving Average Convergence Divergence
#"group_keys=False" est utlisé pour éviter d'inclure les clés de groupe supplémentaires dans le résultat

def compute_macd(close):
    macd = pandas_ta.macd(close=close,length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd) 

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6 
#print(df)



######################## 3. Aggregate to monthly level and filter top 150 most liquid stocks for each month. ##########################

#"df.columns.unique(0)": This gets the unique values along the specified axis 0 (corresponds to the rows). It returns an Index object containing the unique row labels
#For each label'c', it checks if the labels is not present in the list ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']. If it's not present it includes it in the list 'last_cols'
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
#print(last_cols)

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'), 
          df.unstack()[last_cols].resample('M').last().stack('ticker')], axis=1)).dropna()

#print(data)

#Calculate 5-year rolling average ("Moyenne Mobile") of dollar volume for each stocks before filtering.
#".stack()" This operation reverses the effect of '.unstack()', pivoting the innermost level of the row index back into the column index.
#So we pass to ticker columns to ticker rows when I use '.unstack' and '.stack'

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

#Here I compute the dollar volume rank.
#'ascending = False' mean that the ranks are assigned in descending order, meaning higher valuers get lower ranks.
#So for each date group I select the higher dollar volumes receive lower ranks.
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

#print(data)


######################## 4.Calculate Monthly Returns for different time horizons as features. ########################

#To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method .pct_change(lag), that is, 
#returns over various monthly periods as identified by lags.
def calculate_returns(df):
    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                        upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()


#print(data)

######################## 5.Download Fama-French Factors and Calculate Rolling Factor Betas. ########################

#We will introduce the Fama-French data to estimate the exposure of assets to common risk factors using linear regression.
#The five Fama-French factors, namely market risk, size, value, operating profitability, 
#and investmenthave been shown empirically to explain asset returns and are commonly used to assess the risk/return profile of portfolios.
#hence, it is natural to include past factor exposures as financial features in models.
#We can access the historical factor returns using the panda-datareader and estimate historical exposures using the RollingOLS rolling linear regression.

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
               'famafrench',
               start='2010')[0].drop('RF', axis=1) #I only want monthly factors so I use [0]

factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name ='date'
factor_data = factor_data.join(data['return_1m']).sort_index()
#print(factor_data.xs('AAPL', level=1).head())
#print(factor_data.xs('MSFT', level=1).head())
factor_data.groupby(level=1).size()

# Filter out stocks with less than 10 months of data.
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]
#print(factor_data)

#Calculate Rolling Factor Betas

betas = (factor_data.groupby(level=1,
                        group_keys=False)
        .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                    exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                    window=min(24, x.shape[0]),
                                    min_nobs=len(x.columns)+1)
        .fit(params_only=True)
        .params
        .drop('const', axis=1)))

#fixing issue of missing values 
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas.groupby('ticker').shift()))
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()
#print(data.info())
#print(data)
#At this point we have to decide on what ML model and approach to use for predictions etc



######################## 6. For each month fit a K-Means Clustering Algorithm to group similar assets based on their features. ########################

## K-Means Clustering:

# You may want to initialize predefined centroids for each cluster based on your research.
# For visualization purpose of this tutorial we will initially rely on the 'k-means+' initialization.
# Then we will pre-define our centroids for each cluster.

from sklearn.cluster import KMeans
# Apply predefined centroids.

target_rsi_values = [30, 45, 55, 70]
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 6] = target_rsi_values

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
#print(data)

def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6], color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,0], cluster_1.iloc[:,6], color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,0], cluster_2.iloc[:,6], color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,0], cluster_3.iloc[:,6], color = 'black', label='cluster 3')

    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

# for i in data.index.get_level_values('date').unique().tolist():
#     g = data.xs(i, level=0)
#     plt.title(f'Date {i}')
#     plot_clusters(g)



######################## 7.For each month select assets based on the cluster and form a portfolio based on efficient frontier max sharpe ratio optimization. ########################

# First we will filter only stocks corresponding to the cluster we choose based on our hypothesis.
# Momentum is persistent and my idea would be that stocks clustered around RSI 70 centroid should continue to outperform in the following month - thus I would select stocks corresponding to cluster 3.

filtered_df = data[data['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()


## Define portfolio optimization function 

# We will define a function which optimizas portfolio weights using PyPortfolioOpt package and EfficientFrontier optimiazer to maximize the sharpe ratio.
# To optimize the weights of a given portfolio we would need to supply last 1 year prices to the function.
# Apply signle stock weight bounds constraint for diversification (minimum half of equaly weight and maximum 10% of portfolio).

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):

    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()

    return ef.clean_weights()

# Download Fresh Daily Prices Data Only for short listed stocks.

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = pd.DataFrame(yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1]))

#print(new_df)

# Calculate daily returns for each stock which could land up in our portfolio.
# Then loop over each month start, select the stocks for the month and calculate their weights for the next month.
# If the maximum sharpe ratio optimization fails for a given month, apply equally-weighted weights.
# Calculated each day portfolio return.
#new_df.columns = new_df.columns.str.lower()
returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    
    end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')

#optimize_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
optimization_df = new_df['Adj Close']['2016-11-01':'2017-10-30'][cols]

weights = optimize_weights(prices=optimization_df,
                           lower_bound=0)

print(len(optimization_df.columns))

## !!!!!!!!! problem with the function optimize_weights