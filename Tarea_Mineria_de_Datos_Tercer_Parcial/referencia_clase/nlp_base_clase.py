

!pip install nltk

sentence = "The cat is running. The mouse was eating."

from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt_tab')

document = sent_tokenize(sentence)

type(document)

for sentence in document:
  print(sentence)

from nltk.tokenize import word_tokenize

words = word_tokenize(sentence)
type(words)

for word in words:
  print(word)

from nltk.stem import PorterStemmer

steaming = PorterStemmer()

for word in words:
  print(steaming.stem(word))

steaming.stem("congratulations")

steaming.stem("sitting")

from nltk.stem import RegexpStemmer

regex_steammer = RegexpStemmer('ing$|s$|e$|able$', min=4)

regex_steammer.stem("disable")

from nltk.stem import SnowballStemmer

snow = SnowballStemmer(language='spanish')

snow.stem("sonriendo")

"""TF
IDF
Transormaciones que se le hacen a las palabras

"""

from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')

lemma = WordNetLemmatizer()

lemma.lemmatize("was", pos='v')

from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words

paragraph = "The cat is running. The mouse was eaten."

sentences = nltk.sent_tokenize(paragraph)
sentences

words = []

for i in range(len(sentences)):
    words_list = nltk.word_tokenize(sentences[i])
    for j in range(len(words_list)):
       words.append(words_list[j])
words

stopwords_list = []

for i in range (len(words)):
  if words[i] in stop_words:
    stopwords_list.append(words[i])
stopwords_list

from nltk.stem  import PorterStemmer
stemmer = PorterStemmer()

lemmanizelist = []
for i in range(len(stopwords_list)):
  lemmanizelist.append(lemma.lemmatize(stopwords_list[i], pos='v'))
lemmanizelist

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

tagginlist = []
tagginlist = nltk.pos_tag(lemmanizelist)
tagginlist

"""- The food is good
- The food is bad

Si utilzas one hot encoder vas a tener 4 palabras, genera 5 vecotores linealmente independiente. La matriz resultante sería de 5 * 4, 5 por el vocabulario y 4 por la cantidad de palabras.
Al hablar de un vector linealmente independiente se habla de lo siguiente

$v_1 = [1, 0, 0, 0, 0]$

$v_2 = [0, 1, 0, 0, 0]$

Por esto one hot no es la mejor idea porque genera varios problemas, entre los que destacan
* Matriz dispersa: Es una matriz que todos sus vectores son ortogonales.
* Genera overfitting
* Fuera del vocabulario
* Semantica
* Longitud fija: Es que al modelo siempre le llegue la misma matriz

Ahora trabajaremos con las siguientes oraciones:
Ya después de aplicar todos los metodos quería algo así

* He is a good Boy --- [good, boy]
* She is a good girl -- [good, girl]
* Boy and Girl are good -- [boy, girl, good]

Ahora por frecuencia tenemos:
F W
3 good
2 boy
2 girl
Suponiendo que tenemos el vocabulario aplicado así tenemos

B G G

[1 0 1 ]

[0 1 1 ]

[1 1 1 ]

A este algoritmo se le conoce algorithm back of works, ya completo sería

B G G

[2 0 3 ]

[0 2 3 ]

[2 2 3 ]

Desventajas:
* Puede tener vectores identicos con oraciones identicas
* Problemas de la redaccione sematica
* Es poco utlizado
* Problemas de falta de vocabulario
* No capura mucho el orden de las palabras
"""

import pandas as pd

message = pd.read_csv('/content/SMSSpamCollection.txt', sep='\t', names=['label', 'message'])
message.head()

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

for i in range(0, len(message)):
  temp = re.sub('[^a-zA-Z]', ' ', message['message'][i])
  temp = temp.lower()
  temp = temp.split()
  temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
  temp = ' '.join(temp)
  corpus.append(temp)

corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100,binary=True)
cv

X = cv.fit_transform(corpus).toarray()
X

#Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(corpus).toarray()
tfidf.vocabulary_

pip install gensim
import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
wordtvec = api.load('word2vec-google-news-300')
wordtvec['man']
wordtvec['man'].shape
wordtvec.most_similar('happy')
