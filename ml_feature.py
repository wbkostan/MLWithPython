__author__ = 'wbk3zd'

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array

class MLFeatureSet:
    def __init__(self, raw_data, raw_targets, fe=TfidfVectorizer, fe_params={'min_df':2, 'max_df':10, 'max_features':50}):
        self.data = None
        self.target = None
        self.vocabulary = {}
        extractor = fe(**fe_params)
        self.data = extractor.fit_transform(raw_data).toarray()
        self.__codify_targets__(raw_targets)

    def __codify_targets__(self, raw_targets):
        for index, cat in enumerate(set(raw_targets)):
            self.vocabulary[cat] = index
            self.vocabulary[index] = cat
        for index, target in enumerate(raw_targets):
            raw_targets[index] = self.vocabulary[target]
        self.target = array(raw_targets)
