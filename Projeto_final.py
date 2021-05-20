import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
#from dotenv import load_dotenv
import os
import tweepy as tw
from textblob import TextBlob
from wordcloud import WordCloud
import re
from googletrans import Translator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Importing Data
tickers=pd.read_csv('tickers_BOVESPA.csv')
tickers['Name Option']=tickers['Acao']+' / '+tickers['Codigo']
#tickers_brasil=tickers.loc[tickers['Exchange']=='SAO',['Ticker','Name']]
#tickers_brasil['ticker_name']=tickers_brasil['Ticker']+' / '+tickers_brasil['Name']
st.title('Projeto Final')


#Selectbox for tickers

option = st.selectbox(
     'Which ticker would you like to see?',
    list(tickers['Name Option']))

chose_option=st.checkbox('Check if you want to search for Segment')
if chose_option:
  option_setor=st.selectbox('Setor', list(tickers['Setor'].unique()))
  option=st.selectbox('Ticker', list(tickers.loc[tickers['Setor']==option_setor]['Name Option']))

#option = st.sidebar.selectbox(
#     'Which ticker would you like to analyse?',
#    list(tickers_brasil['ticker_name']))

option_selected=tickers.loc[tickers['Name Option']==option]['Codigo']+'.SA'
st.write('You selected:', option)



#Load tha data
data=yfinance.download(str(option_selected.values[0]), start='2017-01-01', end=datetime.datetime.today())


data_check=st.checkbox('Check if you want to see the data')
if data_check:
  st.write(data)

st.markdown('<img src=%s>' % yfinance.Ticker(str(option_selected.values[0])).info['logo_url'])
st.header('**%s**' % yfinance.Ticker(str(option_selected.values[0])).info['longName'])
st.info(yfinance.Ticker(str(option_selected.values[0])).info['longBusinessSummary'])

#MACD

st.title('1.MACD Signal')

data_macd=data[-40:-1]
#calculatge the short term exponential moving average(EMA)
shortEMA=data_macd.Close.ewm(span=12, adjust=False).mean()
#calculate the long term exponential moving average(EMA)
longEMA=data_macd.Close.ewm(span=26, adjust=False).mean()
#calculate the MACD Line
MACD=shortEMA-longEMA
#calculate the signal line
signal=MACD.ewm(span=9, adjust=False).mean()

data_macd['MACD']=MACD
data_macd['Signal Line']=signal

def buy_sell(signal):
  buy=[]
  sell=[]
  flag=-1

  for i in range(0, len(signal)):
    if signal['MACD'][i] > signal['Signal Line'][i]:
      sell.append(np.nan)
      if flag != 1:
        buy.append(signal['Close'][i])
        flag = 1
      else:
        buy.append(np.nan)

    elif signal['MACD'][i] < signal['Signal Line'][i]:
      buy.append(np.nan)
      if flag != 0:
        sell.append(signal['Close'][i])
        flag = 0
      else:
        sell.append(np.nan)
    else:
      buy.append(np.nan)
      sell.append(np.nan)
  return (buy,sell)

a= buy_sell(data_macd)
data_macd['Buy_signal_price']=a[0]
data_macd['Sell_signal_price']=a[1]



fig=make_subplots(vertical_spacing=0, rows=2, cols=1, row_heights=[4,3], shared_xaxes=True)
fig.add_trace(go.Scatter(x=data_macd.index, y=data_macd.Close, name='Price'))
fig.add_trace(go.Scatter(x=data_macd.index, y=data_macd['Buy_signal_price'], mode='markers', marker_color='green', marker=dict(size=8,symbol=5), name='Buy'))
fig.add_trace(go.Scatter(x=data_macd.index, y=data_macd['Sell_signal_price'], mode='markers', marker_color='red', marker=dict(size=8,symbol=6), name='Sell'))
fig.add_trace(go.Scatter(x=data_macd.index, y=MACD, name='MACD', line=dict(color='brown')), row=2, col=1 )
fig.add_trace(go.Scatter(x=data_macd.index, y=signal, name='Signal', line=dict(color='yellow')), row=2, col=1 )

fig.update_layout(xaxis_rangeslider_visible=False, xaxis=dict(zerolinecolor='black', showticklabels=False), xaxis2=dict(showticklabels=True), yaxis2=dict(showticklabels=False))
fig.update_layout(width=1000)

fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
st.plotly_chart(fig)



val_buy=0
val_sell=0
last_price_buy=0
last_price_sell=0
for i in reversed(range(1,data_macd.shape[0])):
  if np.isnan(data_macd['Sell_signal_price'][i])==False:
    last_price_sell=data_macd['Sell_signal_price'][i]
    val_sell=i
  if np.isnan(data_macd['Buy_signal_price'][i])==False:
    last_price_buy=data_macd['Buy_signal_price'][i]
    val_buy=i
  if val_sell!=0 or val_buy!=0:
    break

if val_buy < val_sell:
  val=val_sell
  st.write(f'Last signal was SELL, on {data_macd.index[val].month}/{data_macd.index[val].day} ({data_macd.Sell_signal_price[val]:.2f} reais)')
elif val_buy > val_sell:
  val=val_buy
  st.write(f'Last signal was SELL, on {data_macd.index[val].month}/{data_macd.index[val].day} ({data_macd.Buy_signal_price[val]:.2f} reais)')

#FBProphet

st.title('2.Predicting with Machine Learning (30 days)')

periods_input=30



period_checkbox=st.checkbox('If you want to change the period check it')

if period_checkbox:
  periods_input = st.number_input('How many periods would you like to forecast into the future? (1~360 Days)',
                                  min_value=1, max_value=365)


data['y']=data['Close']
data['ds']=data.index
prophet=Prophet(daily_seasonality=True)
prophet.fit(data)
future=prophet.make_future_dataframe(periods=periods_input)
forecast=prophet.predict(future)
fig=plot_plotly(prophet, forecast)
st.plotly_chart(fig)



#Twitter

st.title('3.Twitter Sentiment Analysis')

#load_dotenv('C://Users//youngbae//Documents//twitter_token.env')
#consumer_key=os.getenv('consumer_key')
#consumer_secret=os.getenv('consumer_secret')
#access_token=os.getenv('access_token')
#access_token_secret=os.getenv('access_token_secret')

st.write('consumer_key:', st.secrets['consumer_key'])


#consumer_key=st.secrets['consumer_key']
#consumer_secret=st.secrets['consumer_secret']
#access_token=st.secrets['acess_token']
#access_token_secret=st.secrets['access_token_secret']


authenticate=tw.OAuthHandler(consumer_key, consumer_secret)
authenticate.set_access_token(access_token, access_token_secret)
api=tw.API(authenticate, wait_on_rate_limit=True)

option_tweet=str(option_selected.values[0]).replace('.SA','')

search_words=option_tweet + '-filter:retweets'
tweets=tw.Cursor(api.search, q=search_words, lang='pt').items(2000)
df=pd.DataFrame([[tweet.created_at,tweet.text] for tweet in tweets], columns=['Date','Tweets'])

def cleanTxt(text):
  text=re.sub('@[A-Za-z0-9:_]+','',text) #remove @mentions
  text=re.sub('RT[\s]+','',text) # remove RT
  text=re.sub('http\S+','',text)
  text=re.sub('\n','',text)
  return text
df['Tweets']=df['Tweets'].apply(cleanTxt)

translator = Translator()
df['Translated']=pd.DataFrame([translator.translate(df['Tweets'][i], src='pt', dest='en').text for i in range(0, df.shape[0])])

def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
  return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['Translated'].apply(getSubjectivity)
df['Polarity']=df['Translated'].apply(getPolarity)

word_cloud=st.checkbox('See the Word Cloud')

allWords=' '.join( [twts for twts in df['Tweets']] )
wordCloud=WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allWords)

if word_cloud:
  plt.figure(figsize=(20, 10))
  plt.imshow(wordCloud, interpolation='bilinear')
  plt.axis('off')
  st.pyplot(fig=plt)

#create a function to compute the negative, neutral and positive analysis
def getAnalysis(score):
  if score<0:
    return 'Negative'
  elif score==0:
    return 'Neutral'
  else:
    return 'Positive'

df.index=df['Date']
df.index=df.index.date

checkbox_twitter=st.checkbox('See tweets')
df['Analysis']=df['Polarity'].apply(getAnalysis)
if checkbox_twitter:
  st.write(df[['Tweets']])

#show the value counts

checkbox_counts=st.checkbox('See polarity counts')
if checkbox_counts:
  fig1=plt.figure(figsize=(20,10))
  ax= sns.countplot(df['Analysis'], palette="Set3")
  plt.title('Sentiment Analysis')
  plt.xlabel('Sentiment')
  plt.ylabel('Count')
  num_total=len(df['Analysis'])
  for p in ax.patches:
      ax.annotate(f'\n{p.get_height()}\n({((p.get_height())/num_total)*100:.1f}%)', (p.get_x()+0.4, p.get_height()), ha='center', va='top', size=18)
  st.pyplot(fig=fig1)

df_polarity=df.groupby(by=df.index).mean()[['Polarity']]
stock=data[['Close']]
df_polarity['Date']=df_polarity.index
df_polarity=df_polarity.reset_index()[['Date','Polarity']]
stock=stock.reset_index()
df_polarity['Date'] = df_polarity['Date'].astype('datetime64[ns]')
compare=df_polarity.merge(stock, how='left')

for i in range(compare.shape[0]):
  if np.isnan(compare['Close'][i])==True:
    compare['Close'][i]=compare['Close'][i-1]
compare.index=compare['Date']

#compare=df_polarity.merge(stock, left_index=True, right_index=True)[['Polarity','Close']]

#compare=df_polarity.merge(stock)
fig, ax1=plt.subplots(figsize=(16,6))
ax2=ax1.twinx()
ax1.plot(compare['Close'], color='blue', label='Price')
ax2.plot(compare['Polarity'], color='red', label='Polarity', linestyle=':')
ax1.set_xlabel('Date', fontsize=15)
ax1.set_ylabel('Price', fontsize=15)
ax2.set_ylabel('Polarity', fontsize=15)
ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.93), fontsize=12)
st.pyplot(fig=fig)

