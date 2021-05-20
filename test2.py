import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from dotenv import load_dotenv

import os
import tweepy as tw
from textblob import TextBlob
from wordcloud import WordCloud
import re
from googletrans import Translator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.wrote('oi')