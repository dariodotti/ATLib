import pickle
import random
import argparse
from collections import Counter

import ATLib_classifiers


def load_matrix_pickle(filename):

    with open(filename, 'rU') as handle:
        file = pickle.load(handle)
    return file



def main():
    # ##INPUT: path of the txt file with all the recorded days
    parser = argparse.ArgumentParser(description='path to txt file')
    parser.add_argument('path_toData')
    #
    args = parser.parse_args()
    path_todata = args.path_toData

    normalized_data_HOT = load_matrix_pickle(path_todata)

    cluster_pred = ATLib_classifiers.cluster_fit_predict_kmeans(normalized_data_HOT,3)
    print Counter(cluster_pred).most_common()


    ## doctors choose what are the clustering that represent the abnormal behavior and a classifier is trained accordingly
    abnormal_HOT_classes = [ Counter(cluster_pred).most_common()[0][0]]


    HOT_normal, HOT_abnormal = ATLib_classifiers.normal_abnormal_class_division(cluster_pred,normalized_data_HOT,abnormal_HOT_classes)


    hot_accuracy = []

    for ten_fold in range(0,10):
        print 'test set '+str(int(len(HOT_normal)*0.1))
        test_index = random.sample(range(0,len(HOT_normal)),int(len(HOT_normal)*0.1))

        hot_accuracy.append(ATLib_classifiers.classification_clustering_accuracy(HOT_normal,HOT_abnormal,test_index,classifier='logistic'))




    print sum(hot_accuracy)/len(hot_accuracy)



if __name__ == '__main__':
    main()
