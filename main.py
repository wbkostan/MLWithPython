__author__ = 'Liam Kostan'

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from ml_input import MLInputFile
from ml_feature import MLFeatureSet

def main():
    ifile = MLInputFile('DataSet.txt')
    percent_split = 0.4
    train_data,test_data = ifile.split_data_set(percent_split)

    #data_pool_1 = ifile.raw_data['bookTitle']
    #data_pool_2 = merge_data([ifile.raw_data['bookTitle'], ifile.raw_data['bookAuthor']])
    data_pool_1 = merge_data([ifile.raw_data['bookTitle']])
    data_pool_2 = consolidate_data(ifile.raw_data)
    #train_pool_1 = consolidate_data(train_data)
    #train_pool_2 = consolidate_data(train_data)
    #test_pool_1 = consolidate_data(test_data)
    #test_pool_2 = consolidate_data(test_data)

    feature1 = MLFeatureSet(raw_data=data_pool_1, raw_targets=ifile.raw_data['categoryLabel'],
                            fe=TfidfVectorizer, fe_params={'min_df':2, 'max_df':10, 'max_features':50})
    feature2 = MLFeatureSet(raw_data=data_pool_2, raw_targets=ifile.raw_data['categoryLabel'],
                            fe=CountVectorizer, fe_params={'analyzer':'char_wb','ngram_range':(3,4),'min_df':3, 'max_features':150})

    X = feature1.data
    y = feature1.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    dtreec = DecisionTreeClassifier()
    kfold = cross_validation.StratifiedKFold(y, n_folds=3)


    score = cross_validation.cross_val_score(estimator=lsvc, X=X, y=y, cv=kfold)
    print("Linear SVC (FS1):", score.mean(), score.std())

    score = cross_validation.cross_val_score(estimator=dtreec, X=X, y=y, cv=kfold)
    print("Decision Tree Classifier (FS1):", score.mean(), score.std())

    X = feature2.data
    y = feature2.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    dtreec = DecisionTreeClassifier()
    kfold = cross_validation.StratifiedKFold(y, n_folds=3)

    score = cross_validation.cross_val_score(estimator=lsvc, X=X, y=y, cv=kfold)
    print("Linear SVC (FS2):", score.mean(), score.std())

    score = cross_validation.cross_val_score(estimator=dtreec, X=X, y=y, cv=kfold)
    print("Decision Tree Classifier (FS2):", score.mean(), score.std())

def merge_data(data_lists):
    if(data_lists == []):
        raise ValueError
    new_list = data_lists[0]
    for measurement in data_lists[1:]:
        assert(len(new_list) == len(measurement))
        for index, entry in enumerate(measurement):
            new_list[index] = new_list[index].strip() + ' ' + entry.strip()
    return new_list

def consolidate_data(data_dict):
    keys = list(data_dict.keys())
    keys.remove('categoryLabel')
    data = data_dict[keys[0]]
    for key in keys[1:]:
        data = merge_data([data, data_dict[key]])
    return data


if __name__ == "__main__":
    main()