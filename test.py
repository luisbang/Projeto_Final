from textblob import TextBlob
import streamlit as st
from googletrans import Translator
st.write('a')

st.write(TextBlob('Bom dia').translate(from_lang='pt', to='en'))
translator = Translator()
st.write(translator.translate('Bom dia', src='pt', dest='en').text)