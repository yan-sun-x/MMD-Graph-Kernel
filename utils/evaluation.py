
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_acc(pred_y, y):
    return (pred_y == y).sum() / len(y)


def cal_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def eva_clustering(K, y_true):
    '''
    Uses the Spectral Clustering to perform clustering
    '''
    y_pred = SpectralClustering(n_clusters = len(np.unique(y_true)), 
                                    random_state = 0,
                                    affinity = 'precomputed').fit_predict(K)

    acc_score = cal_acc(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.cluster.adjusted_rand_score(y_true, y_pred)

    return {'train_clu_acc': acc_score, 'train_clu_nmi': nmi, 'train_clu_ari': ari}


def train_test_svc(K_train, y_train, K_test=None, y_test=None):
    results = {}
    
    clf = SVC(kernel="precomputed", tol=1e-6, probability = True)
    clf.fit(K_train, y_train)
    train_score = clf.score(K_train, y_train)
    results['train_svc_acc'] = train_score

    if K_test is not None:
        test_score = clf.score(K_test, y_test)
        results['test_svc_acc'] = test_score

    return results


def eva_svc(K, y_true):
    '''
    Uses the SVM classifier to perform classification
    '''
    clf = SVC(kernel="precomputed", tol=1e-6, probability=True)
    acc_score = cross_val_score(clf, K, y_true, cv=5)
    return {'cv_svc_mean': np.mean(acc_score), 'cv_svc_std': np.std(acc_score)}
