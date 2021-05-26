   <p align="center">
<img src=https://user-images.githubusercontent.com/79090589/119140306-5e5cfb00-ba1a-11eb-977b-14d97c56c9b8.png width='400' > <img src=https://user-images.githubusercontent.com/79090589/114392695-816cd300-9b6f-11eb-8b13-16c9465707fb.png width='200' >  

# Projeto SENO°
  - link da plataforma : https://share.streamlit.io/luisbang/projeto_final/main/Projeto_final.py

  
Desde muito tempo o homem tem curiosidade sobre o futuro,
Hoje em dia temos muitos referências como filmes, seriados e etc

<img src="https://cdn.meutimao.com.br/_upload/forumtopico/2020/12/03/DeLorean_Arrival.gif" width="250" height="150" /> <img src="https://super.abril.com.br/wp-content/uploads/2018/03/visoes.gif" width="250" height="150">

Apesar de estarmos no século 21 ainda não temos super poderes e nem bola de cristal. Mas somos privilegiados pela quantidade de dados que temos, na verdade a diferença está em o que podemos fazer com elas.  

Se eu te dizer que vc é só um dado!

Muitas pessoas não gostam desse termo, mas é porque talvez não veem o valor que um dado tem.
Vocês assistiram esse filme? se chama divertidamente, e o diretor desse filme conseguiu tornar as emoções que são abstratas em algo visível, aqui vocês veem o controle das emoções de uma adolescente, dependendo da emoção que o controla, ela tem certos comportamentos e influencia os que estão em volta dela.

<img src="https://i.makeagif.com/media/7-18-2017/EpkdfL.gif">

Estou dizendo isso porque suas emoções também são dados, assim como o comportamento. 
Agora imaginem se a gente tivesse o humor e comportamento de todas as pessoas e que isso pode sim mostrar o nosso futuro.

Talvez vocês estejam pensando, o que esse cara está falando?

<img src="https://cms.hostelworld.com/hwblog/wp-content/uploads/sites/2/2017/11/giphy-64.gif">

O Meu projeto é a consolidação de dados do passado, presente e comportamento e sentimento das pessoas, assim apresento a vocês SENO, que tem como objetivo em auxiliar na tomada de decisões dentro do mercado de ações de valores.
  
A plataforma mostra 3 sinais que podem ajudar a analisar.


## 1. MACD
  
  O Primeiro sinal é através da análise de MACD em portugues é Média Móvel Convergente e Divergente. é um método que realmente está sendo utilizado no mercado financeiro. basicamente diferença entre média móvel de 26 dias e 12 dias. E linha de sinal é calculado média móvel de 9 dias de MACD. a linha pontilhada é linha de sinal e linha reta é linha de MACD. Então quando a linha de sinal pontilhada cruza a linha de MACD indo para cima, significa o sinal de Compra e ao contrário quando linha de sinal cruza a linha de MACD descendo significa um sinal de venda. 

![image](https://user-images.githubusercontent.com/79090589/119143667-438c8580-ba1e-11eb-8bcc-1622f2d41cc4.png)
  
  Legal! temos o primeiro sinal para analisar.
  
## 2. Machine Learning
  
  O Segundo sinal é através de dados de passados, para isso utilizei 3 modelos diferentes para comparar dentro de machine learning.
Por se tratar de série temporal utilizei LSTM que é um tipo de rede neural quando se trata de prazo longo. então o resultado foi isso. 
  
<img src="https://user-images.githubusercontent.com/79090589/119144695-4cca2200-ba1f-11eb-9873-1080b8f6575f.png" width="300">
  
  Não parece um modelo legal e parece que o modelo preditivo está seguindo o valor real só que atrasado. 
  
  E o segundo método foi através de Pycaret, primeiramente obteve melhor modelo com a ajuda do Pycaret e com esse modelo obteve seguinte gráfico. 

  <img src="https://user-images.githubusercontent.com/79090589/119145438-045f3400-ba20-11eb-9daf-81cf5f2a0acd.png" witdth="100" height="250">
  
  E overfitou, ele literalmente copiou os dados reais. Isso que eu separei os dados de treino e teste.
  
  Ultimo método que utilizei é Prophet do facebook. que é utilizado para prever o forecast de dados de série temporal como ação. e o resultado foi isso.
 
  <img src="https://user-images.githubusercontent.com/79090589/119145946-86e7f380-ba20-11eb-9ba5-ef5b8728361d.png" witdth="250" height="300">

  ![image](https://user-images.githubusercontent.com/79090589/119146309-e0502280-ba20-11eb-9855-8c8b411847e2.png)

  ele segue uma tendência e mostra uma variânça também e o resultado de score foi muito bom também como a imagem mostrada. Legal! temos o segundo sinal para analisar se a tendência é subir ou descer.

  
  ## 3. Análise de Sentimentos (Twitter)
  
  O último sinal, como no começo tinha citado, é através da análise de sentimentos. Para isso usei Twitter. com Api do twitter trouxe todos tweets relacionados a aquela ação selecionada e transformei cada tweets para um valor de polaridade de -1 a 1. O número negativo significa tweets negativos, 0 neutro e número positivo significa tweets positivo. Assim, comparei o gráfico de polaridade com gráfico de preço de ação para analisar o comportamento dessas duas variáveis. se uma ação está se comportando igual a polaridade e recentemente os tweets estão positivos, é um sinal positivo para compra.
LEGAL! Agora temos último sinal para analisar. 


Assim, juntei 3 análises atrvés de Streamlit para o usuário analisar e tomar a decisão.
