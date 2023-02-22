import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def demographic_parity(pred, sensitive, protected_value, unprotected_value):
    assert len(pred) == len(sensitive)

    protected_idx = np.where(sensitive == protected_value)[0]
    unprotected_idx = np.where(sensitive == unprotected_value)[0]

    dp = 1 - np.abs(pred[protected_idx].mean() - pred[unprotected_idx].mean())

    return dp

def del_DP(pred, sensitive, protected_value, unprotected_value):
    assert len(pred) == len(sensitive)

    protected_idx = np.where(sensitive == protected_value)[0]
    unprotected_idx = np.where(sensitive == unprotected_value)[0]

    del_dp = np.abs(pred[protected_idx].mean() - pred[unprotected_idx].mean())

    return del_dp

def equality_odds_0(pred, actual, sensitive, protected_value, unprotected_value):
    protected_idx = np.where(sensitive == protected_value)[0]
    unprotected_idx = np.where(sensitive == unprotected_value)[0]

    up_conf = confusion_matrix(actual[unprotected_idx], pred[unprotected_idx], labels=[0, 1])
    p_conf = confusion_matrix(actual[protected_idx], pred[protected_idx], labels=[0, 1])

    up_tn, up_fp, up_fn, up_tp = up_conf.ravel()
    p_tn, p_fp, p_fn, p_tp = p_conf.ravel()

    eq_odds_0 = 1 - np.abs(up_fp/(up_fp + up_tn + 1e-10) - p_fp/(p_fp + p_tn + 1e-10))
    return eq_odds_0

def equality_odds_1(pred, actual, sensitive, protected_value, unprotected_value):
    protected_idx = np.where(sensitive == protected_value)[0]
    unprotected_idx = np.where(sensitive == unprotected_value)[0]

    up_conf = confusion_matrix(actual[unprotected_idx], pred[unprotected_idx], labels=[0, 1])
    p_conf = confusion_matrix(actual[protected_idx], pred[protected_idx], labels=[0, 1])

    up_tn, up_fp, up_fn, up_tp = up_conf.ravel()
    p_tn, p_fp, p_fn, p_tp = p_conf.ravel()

    eq_odds_1 = 1 - np.abs(up_tp/(up_tp + up_fn + 1e-10) - p_tp/(p_tp + p_fn + 1e-10))

    return eq_odds_1

def equality_odds(pred, actual, sensitive, protected_value, unprotected_value):
    eq_odds_0 = equality_odds_0(pred, actual, sensitive, protected_value, unprotected_value)
    eq_odds_1 = equality_odds_1(pred, actual, sensitive, protected_value, unprotected_value)

    return 0.5 * (eq_odds_0 + eq_odds_1)

def del_EO(pred, actual, sensitive, protected_value, unprotected_value):
    protected_idx = np.where(sensitive == protected_value)[0]
    unprotected_idx = np.where(sensitive == unprotected_value)[0]

    up_conf = confusion_matrix(actual[unprotected_idx], pred[unprotected_idx], labels=[0, 1])
    p_conf = confusion_matrix(actual[protected_idx], pred[protected_idx], labels=[0, 1])

    up_tn, up_fp, up_fn, up_tp = up_conf.ravel()
    p_tn, p_fp, p_fn, p_tp = p_conf.ravel()

    del_eo = np.abs(up_fp/(up_fp + up_tn + 1e-10) - p_fp/(p_fp + p_tn + 1e-10))  \
        + np.abs(up_fn/(up_fn + up_tp + 1e-10) - p_fn/(p_fn + p_tp + 1e-10))
    
    return del_eo

def unprotected_accuracy(pred, actual, sensitive, unprotected_value):
    assert len(pred) == len(actual)
    assert len(pred) == len(sensitive)

    unprotected_idx = np.where(sensitive == unprotected_value)[0]
    up_acc = accuracy_score(actual[unprotected_idx], pred[unprotected_idx])
    return up_acc

def protected_accuracy(pred, actual, sensitive, protected_value):
    assert len(pred) == len(actual)
    assert len(pred) == len(sensitive)

    protected_idx = np.where(sensitive == protected_value)[0]
    p_acc = accuracy_score(actual[protected_idx], pred[protected_idx])
    return p_acc

def accuracy(pred, actual, sensitive, protected_value, unprotected_value):
    up_acc = unprotected_accuracy(pred, actual, sensitive, unprotected_value)
    p_acc = protected_accuracy(pred, actual, sensitive, protected_value)

    return 0.5 * (up_acc + p_acc)
