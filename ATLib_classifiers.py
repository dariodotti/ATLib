import numpy as np
from sklearn.cluster import KMeans,MeanShift
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import svm


__logistic = LogisticRegression()
__my_svm = svm.SVC()




def logistic_regression_train(data,labels):
    __logistic.fit(data, labels)


def logistic_regression_predict(test_data):
    return __logistic.predict(test_data)


def svm_train(data,labels):
    __my_svm.fit(data,labels)


def svm_predict(test_data):
    return __my_svm.predict(test_data)



def cluster_fit_predict_meanShift(data,seeds):

    if len(seeds)>0:
        my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1,seeds=seeds)
    else:
        my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1)

    #trained_cluster = my_meanShift.fit(data)
    #cluster_prediction = np.array(trained_cluster.predict(data))
    my_meanShift.fit(data)
    #cluster_prediction = np.array(my_meanShift.fit_predict(data))
    # with open('C:/Users/dario.dotti/Documents/LOST_dataset/22_9_2014-1_10_2013_camera017/CHOT_clusterModel.gz', 'wb') as handle:
    #     cPickle.dump(my_meanShift,handle,protocol=2)

    cluster_prediction = np.array(my_meanShift.predict(data))
    return cluster_prediction


def cluster_fit_predict_kmeans(data,k):
    ##KMeans
    my_kmean = KMeans(n_clusters=k,init='k-means++',n_jobs=-1)
    cluster_prediction = np.array(my_kmean.fit_predict(data))
    return cluster_prediction


def normal_abnormal_class_division(cluster_pred,data,abnormal_class):
    freq_counter = Counter(cluster_pred)

    normal_pedestrian_class=[]
    abnormal_pedestrian_class=[]

    for traj in freq_counter.keys():
        if traj in abnormal_class:
            index = np.where(cluster_pred == traj)[0]
            for i in index:
                #print i
                abnormal_pedestrian_class.append(data[i])

        else:
            index = np.where(cluster_pred == traj)[0]
            for i in index:
                normal_pedestrian_class.append(data[i])
                # show_normal_abnormal_traj(pedestrian_cluster[i],scene)


    return normal_pedestrian_class,abnormal_pedestrian_class


def classification_clustering_accuracy(normal_data_pedestrian,abnormal_data_pedestrian,test_index,classifier):
    ####SUPERVISED CLASSIFIER

    all_data= normal_data_pedestrian+abnormal_data_pedestrian
    all_data_labels=np.vstack((np.ones((len(normal_data_pedestrian),1)),np.zeros((len(abnormal_data_pedestrian),1))))

    training_samples = []
    training_labels=[]
    test_samples = []
    test_labels=[]

    for i in range(0,len(all_data)):
        if i in test_index:
            test_samples.append(all_data[i])
            test_labels.append(all_data_labels[i])

        else:
            training_samples.append(all_data[i])
            training_labels.append(all_data_labels[i])


    if classifier == 'logistic':
        #train Logistic regression classifier
        logistic_regression_train(training_samples,np.ravel(training_labels))

        #test
        pred = logistic_regression_predict(test_samples)

    elif classifier == 'svm':
        svm_train(training_samples,np.ravel(training_labels))

        #test
        pred = svm_predict(test_samples)
    else:
        print 'classifier not recognized'

    counter = 0

    for i in range(0,len(pred)):
        if pred[i]==test_labels[i]:
            counter+=1

    return float(counter)/len(test_labels)
