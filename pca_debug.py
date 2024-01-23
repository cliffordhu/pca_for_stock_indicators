# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:37:48 2024

@author: kuifenhu
"""

import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

from arch import arch_model
import yfinance as yf
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import talib as ta
import datetime

tk='tsla'
start_year=2007
#indicators=['d5','ema','rsi','cci','atr','obv','roc','bb','kdj','macd','rocp','will']
indicators=['rsi','cci','atr','obv','roc','rocp']

start_date = datetime.datetime(start_year, 1, 1)
end_date = datetime.date.today()  # Or use datetime.datetime.now() for the exact current time
# Download QQQ stock data
ts = yf.download(tk, start=start_date, end=end_date)  # Download maximum available historical data
df = pd.DataFrame(ts) # Save to dataframe df. 
# Create a pandas DataFrame df1 for traing. The rows are the  sample observations. the columns are the genes
# currently we think the price change over past three days, distance precentage to EMA 20 50 and 200, RSI , BB_Bollinger bnad, 
# dftmp is used to save some temp varibles. 
tmp=df['Adj Close'].diff(periods=30).shift(-30)/df['Adj Close']*100
df1tmp=pd.DataFrame({'return30':tmp})
df1=df1tmp 
rng=df1tmp.std()['return30']

if 'd5' in indicators: 
    df1['d1'] =df['Adj Close'].diff()/df['Adj Close']*100
    df1['d2']=df1['d1'].shift(1)
    df1['d2'].iloc[0]=np.nan
    df1['d3']=df1['d2'].shift(1)
    df1['d3'].iloc[0]=np.nan
    df1['d4']=df1['d3'].shift(1)
    df1['d4'].iloc[0]=np.nan
    df1['d5']=df1['d4'].shift(1)
    df1['d5'].iloc[0]=np.nan


if 'ema' in indicators: 
    df1['EMA_5']=(df['Close']-ta.EMA(df['Close'],timeperiod=5))/df['Close']*100
    df1['EMA_20']=(df['Close']-ta.EMA(df['Close'],timeperiod=20))/df['Close']*100
    df1['EMA_50']=(df['Close']-ta.EMA(df['Close'],timeperiod=50))/df['Close']*100
    df1['EMA_100']=(df['Close']-ta.EMA(df['Close'],timeperiod=100))/df['Close']*100
    df1['EMA_200']=(df['Close']-ta.EMA(df['Close'],timeperiod=200))/df['Close']*100

if 'rsi' in indicators: 
    df1['RSI5']=ta.RSI(df['Close'],timeperiod=5)
    df1['RSI14']=ta.RSI(df['Close'],timeperiod=14)
    df1['RSI28']=ta.RSI(df['Close'],timeperiod=28)

if 'cci' in indicators: 
    df1['CCI5']=ta.CCI(df['High'],df['Low'],df['Close'],timeperiod=5)
    df1['CCI14']=ta.CCI(df['High'],df['Low'],df['Close'],timeperiod=14)
    df1['CCI28']=ta.CCI(df['High'],df['Low'],df['Close'],timeperiod=28)

if 'atr' in indicators: 
    df1['ATR5']=ta.ATR(df['High'],df['Low'],df['Close'],timeperiod=5)
    df1['ATR14']=ta.ATR(df['High'],df['Low'],df['Close'],timeperiod=14)

if 'obv' in indicators: 
    df1['OBV5']=ta.OBV(df['Close'],df['Volume']/10000)
    df1['OBV14']=ta.OBV(df['Close'],df['Volume']/10000)

if 'roc' in indicators: 
    df1['ROC5']=ta.ROC(df['Close'],timeperiod=5)
    df1['ROC14']=ta.ROC(df['Close'],timeperiod=14)
    df1['ROC28']=ta.ROC(df['Close'],timeperiod=28)


if 'bb' in indicators: 
    df1tmp['UpperBand'], df1tmp['MiddleBand'], df1tmp['LowerBand'] = ta.BBANDS(df['Close'], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)
    df1['BB_percent14']=(df['Close']-df1tmp['LowerBand'])/(df1tmp['UpperBand']-df1tmp['LowerBand'])*100
    df1['BB14']=df1tmp['UpperBand']-df1tmp['LowerBand']


    df1tmp['UpperBand'], df1tmp['MiddleBand'], df1tmp['LowerBand'] = ta.BBANDS(df['Close'], timeperiod=28, nbdevup=2, nbdevdn=2, matype=0)
    df1['BB_percent28']=(df['Close']-df1tmp['LowerBand'])/(df1tmp['UpperBand']-df1tmp['LowerBand'])*100
    df1['BB28']=df1tmp['UpperBand']-df1tmp['LowerBand']

if 'kdj' in indicators: 
    df1tmp['k'],df1tmp['d'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df1['j'] = 3 * df1tmp['d'] - 2 * df1tmp['k']  # Calculate J line
    df1['d'] = df1tmp['d'] 
    df1['k']= df1tmp['k']  # Calculate J line


if 'rocp' in indicators: 
    df1['rocp10']=ta.ROCP(df['Close'],timeperiod=10)*100
    df1['rocp20']=ta.ROCP(df['Close'],timeperiod=20)*100
    df1['rocp30']=ta.ROCP(df['Close'],timeperiod=30)*100

if 'will' in indicators: 
    df1['willr'] = ta.WILLR(df['High'],df['Low'],df['Close'], timeperiod=14)
    df1['willr28'] = ta.WILLR(df['High'],df['Low'],df['Close'], timeperiod=28)

if 'macd' in indicators: 
    df1['macd'], df1['signal'], df1['macd_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

df1['gain']=df1['return30']*0

# #check the imported data return of 30 days out. 
# fig=plt.figure(figsize=(10,4))
# ax = fig.add_subplot(111)
# plt.plot(df1tmp['return30'])
# plt.xlabel('Date')
# plt.xlabel('returns')
# plt.title(tk+ ' returns %', fontsize=20)

#Transpose to make it ready for PCA training
df2=df1.T
N=100
k=0
j=0
marker=[]
colors=[]
sizes=[]
data=pd.DataFrame(index=df2.index)
# tag the sample points into high low gains group before creating the training data. 
# EMA use 200, avoid first 300 points with nan. The observation has 30 days out, avoid last 31 days for nan. 
#create the train data set called data as dataframe
for i in df2.columns[300:]:
   if  df1tmp['return30'][i]>=2*rng: 
       df2.loc['gain',i]=0.5
       data=data.copy()
       data['up'+str(k)+' '+i.strftime("%Y-%m-%d")]=df2[i]
       marker.append('up'+str(k)+' '+i.strftime("%Y-%m-%d"))
       colors.append('darkgreen')
       sizes.append(40)
       k=k+1
   if  df1tmp['return30'][i]<2*rng and df1tmp['return30'][i]>=rng: 
       df2.loc['gain',i]=0.8
       data=data.copy()
       data['up'+str(k)+' '+i.strftime("%Y-%m-%d")]=df2[i]
       marker.append('up'+str(k)+' '+i.strftime("%Y-%m-%d"))
       colors.append('lime')
       sizes.append(30)
       k=k+1
   if df1tmp['return30'][i]<rng and df1tmp['return30'][i]>=0: 
       df2.loc['gain',i]=1
       data=data.copy()
       data['up'+str(k)+' '+i.strftime("%Y-%m-%d")]=df2[i]
       marker.append('up'+str(k)+' '+i.strftime("%Y-%m-%d"))
       colors.append('limegreen')
       sizes.append(20)
       k=k+1
       
   if df1tmp['return30'][i]<0 and df1tmp['return30'][i]>=-rng: 
        df2.loc['gain',i]=-1
        data=data.copy()
        data['dn'+str(j)+' '+i.strftime("%Y-%m-%d")]=df2[i]
        marker.append('dn'+str(j)+' '+i.strftime("%Y-%m-%d"))
        colors.append('lightcoral')
        sizes.append(20)
        j=j+1
      
    
   if df1tmp['return30'][i]<-rng and df1tmp['return30'][i]>=-2*rng:
        df2.loc['gain',i]=-0.8
        data=data.copy()
        data['dn'+str(j)+' '+i.strftime("%Y-%m-%d")]=df2[i]
        marker.append('dn'+str(j)+' '+i.strftime("%Y-%m-%d"))
        colors.append('brown')
        sizes.append(30)
        j=j+1
          
   if df1tmp['return30'][i]<-2*rng: 
        df2.loc['gain',i]=-0.5
        data=data.copy()
        data['dn'+str(j)+' '+i.strftime("%Y-%m-%d")]=df2[i]
        marker.append('dn'+str(j)+' '+i.strftime("%Y-%m-%d"))
        colors.append('maroon')
        sizes.append(40)
        j=j+1
      
data1=data
data=data.drop('return30')
data=data.drop('gain')

# scale and prepare the data into array    
scaled_data=preprocessing.scale(data.T)
# pca fit
pca=PCA()
pca.fit(scaled_data)
#pca transpose the training data into new dimension
pca_data=pca.transform(scaled_data)
#pca covariant of each principle axis
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
#Create the PCx lables
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
# save the training samples new cooridinate to df holder pca_df
pca_df=pd.DataFrame(pca_data,index=marker,columns=labels)


# plot the PCx covirance 
fig=plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('percentage of expalned variance')
plt.xlabel('Priciple components')
plt.title(tk+ ' PCA Screen plot')
plt.show()

# plot the transposed sample points on the first three PCA axis
fig=plt.figure(figsize=(10,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df.PC1, pca_df.PC2,pca_df.PC3,c=colors,s=sizes)
plt.title(tk+'PCA Analysis')
plt.xlabel('PC1-{0}%'.format(per_var[0]))
plt.ylabel('PC2-{0}%'.format(per_var[1]))
ax.set_zlabel('PC3-{0}%'.format(per_var[2]))
ax.scatter(pca_df.PC1, pca_df.PC2,pca_df.PC3,c=colors,s=sizes)
plt.show()

# Predict with models
# prepare the new observation with all new value assigned to genes



sample='Current'

#a=data.T
#gene=a.loc['up0'].tolist()
gene=df2.iloc[:, -1].drop('return30')
gene=gene.drop('gain')
gene_p=preprocessing.scale(gene)
observation = pca.transform([gene_p])
obs_df=pd.DataFrame(observation[0],index=labels,columns=['obs1']).T
ax.scatter(obs_df.PC1, obs_df.PC2,obs_df.PC3,c='r',s=300,marker='*')
plt.annotate(sample,xy=(obs_df.PC1[0],obs_df.PC2[0]))

# Post data analysis: Evaluate the 
r_df=pd.concat([pca_df, obs_df],axis=0)

for i in r_df.columns:
    r_df[i]=r_df[i].apply(lambda p:(p-r_df[i].iloc[-1])**2)
distance_df=r_df.iloc[:,:3].sum(axis=1).apply(lambda p:(np.sqrt(p))).to_frame()
distance_df.columns=['distance']
gain=data1.loc['gain'].to_frame()
gain.loc['obs1']={'gain':0}

Rp_df=pd.DataFrame(columns=['posDistance'])
Rn_df=pd.DataFrame(columns=['negDistance'])
for i in gain.index:
    if gain['gain'].loc[i]>0:
      Rp_df.loc[i]=distance_df['distance'].loc[i]*gain.loc[i,'gain'] 
    elif gain['gain'].loc[i]<0:
      Rn_df.loc[i]=distance_df['distance'].loc[i]*gain.loc[i,'gain'] 

summary_df=pd.DataFrame(columns=['posDistance','negDistance'])
for i in range(10,200,20): 
      summary_df.loc['Points'+ str(i),'posDistance']=float(Rp_df.sort_values(by='posDistance').head(i).sum()[0])
      summary_df.loc['Points'+ str(i),'negDistance']= float(Rn_df.sort_values(by='negDistance',ascending=False).head(i).sum()[0])


# verify the cloest points up1062
RpselectedRow=Rp_df.sort_values(by='posDistance').head(20).index
RnselectedRow=Rn_df.sort_values(by='negDistance',ascending=False).head(20).index
ax.scatter(pca_df.loc[RpselectedRow,'PC1'],pca_df.loc[RpselectedRow,'PC2'],pca_df.loc[RpselectedRow,'PC3'],c='g',s=200,marker='2')
ax.scatter(pca_df.loc[RnselectedRow,'PC1'],pca_df.loc[RnselectedRow,'PC2'],pca_df.loc[RnselectedRow,'PC3'],c='r',s=200,marker='1')

xs=pca_df.loc[RpselectedRow,'PC1'].tolist()
ys=pca_df.loc[RpselectedRow,'PC2'].tolist()
zs=pca_df.loc[RpselectedRow,'PC3'].tolist()
for x,y,z,index in zip(xs,ys,zs,RpselectedRow):  
    ax.text(x, y, z, index, (1,1,1),color='g')
xs=pca_df.loc[RnselectedRow,'PC1'].tolist()
ys=pca_df.loc[RnselectedRow,'PC2'].tolist()
zs=pca_df.loc[RnselectedRow,'PC3'].tolist()
for x,y,z,index in zip(xs,ys,zs,RnselectedRow):  
    ax.text(x, y, z, index, (1,1,0),color='r')


# Generate tos script 
k=0
tos=""
for i in RpselectedRow:
    label=i[-10:].replace('-','')
    tos=tos+f"AddChartBubble(GetYYYYMMDD() == {label}  , low, \"Rp{str(k)}\" , Color.GREEN, no);\n"
    k=k+1 
k=0
for i in RnselectedRow:
    label=i[-10:].replace('-','')
    tos=tos+f"AddChartBubble(GetYYYYMMDD() == {label}  , low,  \"Rn{str(k)}\"  , Color.RED, no);\n"
    k=k+1





summary_df[tk]=(1-summary_df['posDistance']/(summary_df['posDistance']-summary_df['negDistance'])) *100  

print(summary_df)
print(tos)



#return summary_df, df1,data