import numpy as np
from collections import defaultdict
from gensim.models import FastText
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases
import gensim
import fasttext

from utils.config import config as cf


def analyse_oov(documents, pretrained_model, condition_function):
    exist_count = 0
    distinct_exist_count = 0
    total = 0
    word_set = set()
    oov_words = set()
    for line in documents:
        for token in line :
            total +=1
            word_set.add(token)
            if condition_function(pretrained_model, token):
                exist_count +=1

    for token in word_set:
        if condition_function(pretrained_model, token):
            distinct_exist_count +=1
        else:
            oov_words.add(token)
    
    distinct_oov = round(1 - distinct_exist_count/len(word_set), 4)
    total_oov = round(1 - exist_count/total, 4)
    print('Distinct OOV words : {:.2f}%'.format(distinct_oov*100))
    print('Words represented by a vector of zeros : {:.2f}%'.format(total_oov*100))
    return distinct_oov, total_oov, oov_words

    
def createW2V(sentences, size=100, window=5, min_count=2, iter_num=100):
    w2v_model = Word2Vec(sentences, size=size, window=window, min_count=min_count, iter=iter_num)
    w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.vectors)}
    return w2v
        
        
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        super().__init__()
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        result = list()
        for words in X:
            temp = []
            if len(words) > 0 :
                for token in words:
                    if  token in self.word2vec:
                        temp.append(self.word2vec[token])
            if len(temp) == 0:
                temp.append(np.zeros(self.dim))
                
            result.append(np.mean(temp, axis=0))
        return result


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # If the word have been never seen, 
        # it must be at least as infrequent as any of the known words
        # So, the default idf is the max of known idf
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class W2VBigramVectorizer(BaseEstimator, TransformerMixin):
    """
    Build bigram using gensim.Phraser
    Example code:
    
    >>  bigram = BigramBuilder(w2v)
    >>  bigram.fit(list(df['processed']))
    >>  temp = bigram.transform(list(df['processed']))
    >>  for topic in temp:
    >>      print(topic)
    
    """
    def __init__(self, min_count=5, threshold=10, delimiter=b'_'):
        super().__init__()
        self.min_count = min_count
        self.threshold = threshold
        self.delimiter = delimiter
            
    def fit(self, x, y=None):
        self.phraser = Phraser(Phrases(x, min_count=self.min_count, 
                        threshold=self.threshold, delimiter=self.delimiter))
        self.bigram = self.phraser[x]
        self.word2vec = createW2V(self.bigram)
        if len(self.word2vec) > 0:
            self.dim = len(self.word2vec[next(iter(self.word2vec))])
        else:
            self.dim = 0
            
        return self

    def transform(self, x):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in x
        ])
    
class FastTextMeanEmbeddingVectorizer(object):
    def __init__(self, size=100, window=5, min_count=2, workers=4, epochs=100):
        super().__init__()
        self.model = FastText(size=size, window=window, min_count=min_count, workers=workers, sg=1)
        self.size = size
        self.epochs = epochs

    def fit(self, X, y=None):
        self.model.build_vocab(sentences=X)
        self.model.train(sentences=X, total_examples=len(X), epochs=self.epochs)  # train
        return self

    def transform(self, X):
        result = []
        for words in X:
            if len(words) >0 :
                mean = np.mean([self.model.wv[w] for w in words], axis=0)
            else:
                mean = [np.zeros(self.size)]
            result.append(mean)
        return np.array(result)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        

class W2VVectorizer(object):
    def __init__(self, w2v_model, model_bin_file = cf.PRETRAINED_W2V_BIN):
        super().__init__()
        if w2v_model == None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_bin_file, binary=True) 
        else:
            self.model = w2v_model

    def fit(self, X, y):
        return self

    def transform(self, X):
        result =[]
        for words in X:
            temp = []
            if len(words) > 0 :
                for token in words:
                    if  token in self.model.vocab:
                        temp.append(self.model[token])
                    elif token.capitalize() in self.model.vocab:
                        temp.append(self.model[token.capitalize()])
                    elif token.upper() in self.model.vocab:
                        temp.append(self.model[token.upper()])
                   
            if len(temp) == 0:
                temp.append(np.zeros(self.model.vector_size))
            result.append(np.mean(temp, axis=0))
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
class FastTextVectorizer(object):
    def __init__(self, fastText_model = None, model_bin_file = cf.PRETRAINED_FASTTEXT_BIN):
        """
        Initialize FastText Vectorizer from pre-trained model

        :param fastText_model: pre-trained FastText model.
        :param model_bin_file: path to pre-trained model binary file. Ignored when fastText_model is passed
        """
        super().__init__()
        if fastText_model is None:
            self.model = fasttext.FastText.load_model(model_bin_file) 
        else:
            self.model = fastText_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for words in X:
            sentence = ' '.join(words)
            result.append(self.model.get_sentence_vector(sentence))
        return np.array(result)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)



class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if x[self.key].dtype == object:
            return x[self.key]
        else:
            return x[self.key].values.reshape(-1,1)


class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.transpose(np.matrix(data))
        