"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""

import numpy as np

def cal_nauc(fpr, tpr):
    fpr_loc = np.where(fpr<=0.001)
    fpr_new = fpr[fpr_loc]
    tpr_new = tpr[fpr_loc]
    
    if fpr_new[-1]<0.001:
        fpr_new = np.concatenate((fpr_new,np.ones(1)*0.001), axis=0)
        tpr_new = np.concatenate((tpr_new,np.ones(1)*tpr_new[-1]), axis=0)
        
    from numpy import trapz
    auc = trapz(tpr_new, fpr_new, dx=0.001)*1000
    
    return auc
