__author__ = 'Liam Kostan'

from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from ml_input import MLInputFile

def main():
    iris = load_iris()
    ifile = MLInputFile('DataSet.txt')
    vectorizer = TfidfVectorizer(min_df=2)
    X = vectorizer.fit_transform(ifile.data['bookTitle'])
    y = ifile.data['categoryLabel']
    svc_learner = LinearSVC(multi_class='ovr')
    svc_learner.fit(X,y)

    X_new = vectorizer.transform(['Cengage Advantage Books: American Pageant, Volume 2: Since 1865','Programming Logic and Design, Comprehensive'])

    result = svc_learner.predict(X_new)
    print(result)

if __name__ == "__main__":
    main()