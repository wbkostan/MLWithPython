__author__ = 'Liam Kostan'

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from ml_input import MLInputFile
from ml_feature import MLFeatureSet
from math import ceil

def main():
    MIN_RECORDS_PER_FOLD = 100

    ifile = MLInputFile(get_input_file())

    if('bookTitle' in ifile.raw_data):
        data_pool_1 = merge_data([ifile.raw_data['bookTitle']])
    else:
        data_pool_1 = merge_data([ifile.raw_data[ifile.get_key_by_length_rank(1)]])
    data_pool_2 = consolidate_data(ifile.raw_data, ifile.category_key)

    number_of_records = ifile.number_of_records
    number_of_folds = ceil(number_of_records/float(MIN_RECORDS_PER_FOLD))
    if number_of_folds < 2:
        number_of_folds = 2
    if number_of_folds > 10:
        number_of_folds = 10

    feature1 = MLFeatureSet(raw_data=data_pool_1, raw_targets=ifile.raw_data[ifile.category_key],
                            fe=TfidfVectorizer, fe_params={'min_df':2, 'max_df':number_of_records/10, 'max_features':50})
    feature2 = MLFeatureSet(raw_data=data_pool_2, raw_targets=ifile.raw_data[ifile.category_key],
                            fe=CountVectorizer, fe_params={'analyzer':'char_wb','ngram_range':(3,5),'min_df':3, 'max_features':150})

    X = feature1.data
    y = feature1.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    dtreec = DecisionTreeClassifier()
    kfold = cross_validation.StratifiedKFold(y, n_folds=number_of_folds)


    score = cross_validation.cross_val_score(estimator=lsvc, X=X, y=y, cv=kfold)
    print("Linear SVC (FS1):", score.mean(), score.std())

    score = cross_validation.cross_val_score(estimator=dtreec, X=X, y=y, cv=kfold)
    print("Decision Tree Classifier (FS1):", score.mean(), score.std())

    X = feature2.data
    y = feature2.target

    #Estimator and Validator components
    lsvc = LinearSVC(multi_class='ovr')
    dtreec = DecisionTreeClassifier()
    kfold = cross_validation.StratifiedKFold(y, n_folds=number_of_folds)

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

def consolidate_data(data_dict, category_key):
    keys = list(data_dict.keys())
    keys.remove(category_key)
    data = data_dict[keys[0]]
    for key in keys[1:]:
        data = merge_data([data, data_dict[key]])
    return data

def get_input_file():
    default = "DataSet.txt"
    print("Default: "+default)
    file_name = input("Relative path to input file (leave blank to use default): ")
    if not file_name:
        file_name = default
    print()
    return file_name

if __name__ == "__main__":
    main()