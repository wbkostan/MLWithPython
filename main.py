__author__ = 'Liam Kostan'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from ml_input import MLInputFile
from ml_feature import MLFeatureSet

def main():
    ifile = MLInputFile('DataSet.txt')

    feature1 = MLFeatureSet(raw_data=ifile.raw_data['bookTitle'], raw_targets=ifile.raw_data['categoryLabel'],
                            fe=TfidfVectorizer, fe_params={'min_df':2, 'max_df':10, 'max_features':50})
    feature2 = MLFeatureSet(raw_data=ifile.raw_data['bookAuthor'], raw_targets=ifile.raw_data['categoryLabel'],
                            fe=TfidfVectorizer, fe_params={'min_df':2, 'max_df':10, 'max_features':50})

    X = feature1.data
    y = feature1.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    kfold = cross_validation.StratifiedKFold(y, 3)

    score = cross_validation.cross_val_score(estimator=lsvc, X=X, y=y, cv=kfold)
    print(score.mean(), score.std())

    X = feature2.data
    y = feature2.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    kfold = cross_validation.StratifiedKFold(y, 3)

    score = cross_validation.cross_val_score(estimator=lsvc, X=X, y=y, cv=kfold)
    print(score.mean(), score.std())

if __name__ == "__main__":
    main()