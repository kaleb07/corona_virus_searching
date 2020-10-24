import re
import math
import string
import xml.dom.minidom as minidom
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def parse_xml():
    news_collection = minidom.parse("data/news.xml")

    news_id = news_collection.getElementsByTagName('ID')
    news_source = news_collection.getElementsByTagName('SOURCE')
    news_link = news_collection.getElementsByTagName('LINK')
    news_title = news_collection.getElementsByTagName('TITLE')
    news_author = news_collection.getElementsByTagName('AUTHOR')
    news_datetime = news_collection.getElementsByTagName('DATETIME')
    news_paragraph = news_collection.getElementsByTagName('PARAGRAPH')
    N_news = len(news_id)
    
    id_in_news =[]
    sentence_in_source = []
    sentence_in_link = []
    sentence_in_title = []
    sentence_in_author = []
    sentence_in_datetime = []
    sentence_in_news = []

    for i in range(N_news):
        ids=news_id[i].firstChild.data
        id_in_news.append(ids)

    for i in range(N_news):
        sentences = news_source[i].firstChild.data
        sentence_in_source.append(sentences)

    for i in range(N_news):
        sentences = news_link[i].firstChild.data
        sentence_in_link.append(sentences)

    for i in range(N_news):
        sentences = news_title[i].firstChild.data
        sentence_in_title.append(sentences)

    for i in range(N_news):
        sentences = news_author[i].firstChild.data
        sentence_in_author.append(sentences)

    for i in range(N_news):
        sentences = news_datetime[i].firstChild.data
        sentence_in_datetime.append(sentences)

    for i in range(N_news):
        sentences = news_paragraph[i].firstChild.data
        sentence_in_news.append(sentences)
        
    return ({'id_in_news': id_in_news, 'sentence_in_source' : sentence_in_source, 'sentence_in_link' : sentence_in_link, 'sentence_in_title' : sentence_in_title,
            'sentence_in_author' : sentence_in_author, 'sentence_in_datetime' : sentence_in_datetime,
            'sentence_in_news': sentence_in_news})

def removePunctuation(textList):
    for i in range(len(textList)):
        for punct in string.punctuation:
            textList[i] = textList[i].replace(punct, " ")
        textList[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', textList[i], flags=re.MULTILINE)
        textList[i] = re.sub(r'“', '', textList[i])
        textList[i] = re.sub(r'”', '', textList[i])
    return textList

def token(sentence):
    token = []
    for word in CountVectorizer().build_tokenizer()(sentence):
        token.append(word)
    return token

def tokenize(textList):
    tokens = []
    for i in range(len(textList)):
        tokens.append(token(textList[i]))
    return tokens

def caseFolding(textList):
    text = []
    for i in range(len(textList)):
        text.append(textList[i].lower())
    return text

def get_token():
    file = parse_xml()
    content = removePunctuation(file['sentence_in_news'])
    title = removePunctuation(file['sentence_in_title'])
    contents = caseFolding(content)
    titles = caseFolding(title)
    token_contents = tokenize(contents)
    token_titles = tokenize(titles)

    token = []
    for i in token_titles:
        token.append(i)
    for j in token_contents:
        token.append(j)
    return token

def checkStopword(sentence, stop_words):
    sentence = [w for w in sentence if not w in stop_words]
    return sentence

def stopwordRemove():
    token = get_token()
    with open("data/id.stopwords.02.01.2016.txt", "r") as fd:
        stopwords = fd.read().splitlines()
    stop_words = set(stopwords)
    text = []
    for i in range(len(token)):
        text.append(checkStopword(token[i], stop_words))
    return text

def numberRemove():
    tokenize = stopwordRemove()
    text = []
    for i in range(len(tokenize)):
        text.append([w for w in tokenize[i] if not any(j.isdigit() for j in w)])
    return text

def stemming():
    tokenize = numberRemove()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = tokenize
    for i in range(len(tokenize)):
        for j in range(len(tokenize[i])):
            text[i][j] = stemmer.stem(text[i][j])
    return text

def getAllTerms():
    term = stemming()
    terms = []
    for i in range(len(term)):
        for j in range(len(term[i])):
            terms.append(term[i][j])
    return sorted(set(terms))

def createIndex():
    file = parse_xml()
    textList = stemming()
    terms = getAllTerms()
    proximity = {}
    for term in terms:
        position = {}
        for n in range(len(textList)):
            if(term in textList[n]):
                position[(file['id_in_news']*2)[n]] = []
                for i in range(len(textList[n])):
                    if(term == textList[n][i]):
                        position[(file['id_in_news']*2)[n]].append(i)
        proximity[term] = position
    return proximity

def queryPreprocessing(query):
    terms=[]
    translator=str.maketrans('','',string.punctuation)

    for i in range(len(query)):
        query[i]=query[i].translate(translator)
        query[i]=''.join([i for i in query[i] if not i.isdigit()])
        query[i]=re.sub(r'^https?:\/\/.*[\r\n]*','', query[i], flags=re.MULTILINE)
        terms.append(word_tokenize(query[i]))
    terms
    terms = stopwordRemove()
    terms = stemming()
    return terms

def queryInIndex(query, index):
    result = []
    for word in query:
        if word in index:
            result.append(word)
    return result

def df(query, index):
    docFreq = {}
    for word in query:
        if word in index:
            docFreq[word] = len(index[word])
    return docFreq

def idf(df, N):
    inv = {}
    for word in df:
        inv[word] = math.log10(N/df[word])
    return inv

def tf(query, index):
    termFreq = {}
    for word in query:
        freq = {}
        if word in index:
            for i in index[word]:
                freq[i] = len(index[word][i])
        termFreq[word] = freq
    return termFreq

def tfidf(tf, idf):
    w = {}
    for word in tf:
        wtd = {}
        for doc in tf[word]:
            wtd[doc] = (1+(math.log10(tf[word][doc])))*idf[word]
        w[word] = wtd
    return w
    
def score(TFIDF):
    res = {}
    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = 0
    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = res[j]+TFIDF[i][j]
    sorted_dict = sorted(res, key=res.get, reverse=True)
    return ({'sorted_dict': sorted_dict, 'res': res})

def results(query):
    file = parse_xml()
    index = createIndex()
    terms = queryPreprocessing(query)
    query = terms[0]
    query = queryInIndex(query, index)

    N               = N_news
    tfidf_list      = []

    docFrequency    = df(query, index)
    invDocFrequency = idf(docFrequency, N)
    termFrequency   = tf(query, index)
    TFIDF           = tfidf(termFrequency, invDocFrequency)
    sc              = score(TFIDF)
    
    relevanceDocNumber = []
    count = 0
    result = []
    for i in range(len(sc['sorted_dict'])):
        relevanceDocNumber.append(int(sc['sorted_dict'][i]))
        a = file['id_in_news'].index(sc['sorted_dict'][i])
        rank = i+1
        doc_id = sc['sorted_dict'][i]
        doc_source = sc['res'][sc['sorted_dict'][i]]
        doc_link = file['sentence_in_link'][a][:]
        doc_title = file['sentence_in_title'][a][:]
        doc_author = file['sentence_in_author'][a][:]
        doc_datetime = file['sentence_in_datetime'][a][:]
        doc_contents = file['sentence_in_news'][a][0:400]+'..........'
        result.append({'doc_id' : doc_id, 'doc_source': doc_source, 'doc_link': doc_link, 'doc_author': doc_author,
                 'doc_datetime': doc_datetime, 'doc_contents': doc_contents})
        count = count + 1
        if(count>=5):
            break
    return result