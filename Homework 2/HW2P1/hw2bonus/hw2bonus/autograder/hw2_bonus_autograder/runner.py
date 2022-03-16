import json
import sys
import traceback
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from test import *

sys.path.append('mytorch')
from dropout2d import *
from batchnorm2d import *


base_dir = 'autograder/hw2_bonus_autograder/'

############################################################################################
###############################   Section 5 - Dropout2d  ###################################
############################################################################################

def dropout2d_correctness():
    scores_dict = [0, 0]
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################

    dropout_p = 0.4
    test_forward_data = np.random.random((5, 3, 4, 4))
    test_backward_data = np.random.random((5, 3, 4, 4))

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_model = Dropout2d(dropout_p)

    #############################################################################################
    ##########################    Load the correct results from file   ##########################
    #############################################################################################
    sol_data = np.load(base_dir + "sol_dropout2d_handout.npz")
    sol_forward = sol_data["forward"]
    sol_backward = sol_data["backward"]

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        test_y = test_model(test_forward_data, False)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_y, sol_forward, check, 'train forward'): return scores_dict

    inference_y = test_model(test_forward_data, True)

    for check in ('type', 'shape', 'closeness'):
        if not assertions(inference_y, test_forward_data, check, 'eval forward'): return scores_dict

    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        test_xp = test_model.backward(test_backward_data)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_xp, sol_backward, check, 'backward'): return scores_dict

    scores_dict[1] = 1

    return scores_dict

def batchnorm2d_correctness():
    scores_dict = [0, 0]
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################

    alpha = 0.4
    num_channels = 3
    test_forward_data = np.random.random((5, num_channels, 4, 4))
    test_backward_data = np.random.random((5, num_channels, 4, 4))

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_model = BatchNorm2d(num_channels, alpha)

    #############################################################################################
    ##########################    Load the correct results from file   ##########################
    #############################################################################################
    sol_data = np.load(base_dir + "sol_bn2d_handout.npz")

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        test_y = test_model(test_forward_data, False)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_y, sol_data["fw_train"], check, 'train forward'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.running_M, sol_data["running_M"], check, 'train running mean'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.running_V, sol_data["running_V"], check, 'train running var'): return scores_dict

    inference_y = test_model(test_forward_data, True)

    for check in ('type', 'shape', 'closeness'):
        if not assertions(inference_y, sol_data["fw_eval"], check, 'eval forward'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.running_M, sol_data["running_M"], check, 'eval running mean'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.running_V, sol_data["running_V"], check, 'eval running var'): return scores_dict

    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        test_dx = test_model.backward(test_backward_data)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_dx, sol_data["bw"], check, 'dx'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.dLdBb, sol_data["dLdBb"], check, 'dLdBb'): return scores_dict

    for check in ('type', 'shape', 'closeness'):
        if not assertions(test_model.dLdBW, sol_data["dLdBW"], check, 'dLdBW'): return scores_dict
    scores_dict[1] = 1

    return scores_dict

def test_dropout2d():
    np.random.seed(11785)
    a, b = dropout2d_correctness()
    if a != 1:
        if __name__ == '__main__':
            print('Failed Dropout2d Forward Test')
        return False
    elif b != 1:
        if __name__ == '__main__':
            print('Failed Dropout2d Backward Test')
        return False
    else:
        if __name__ == '__main__':
            print('Passed Dropout2d Test')
    return True

def test_bn2d():
    np.random.seed(11785)
    a, b = batchnorm2d_correctness()
    if a != 1:
        if __name__ == '__main__':
            print('Failed BatchNorm2d Forward Test')
        return False
    elif b != 1:
        if __name__ == '__main__':
            print('Failed BatchNorm2d Backward Test')
        return False
    else:
        if __name__ == '__main__':
            print('Passed BatchNorm2d Test')
    return True


############################################################################################
#################################### DO NOT EDIT ###########################################
############################################################################################

if __name__ == '__main__':
    
    tests = [
        {
            'name': 'Section 3 - Dropout2d',
            'autolab': 'Dropout2d',
            'handler': test_dropout2d,
            'value': 5,
        },
        {
            'name': 'Section 4 - BatchNorm2d',
            'autolab': 'BatchNorm2d',
            'handler': test_bn2d,
            'value': 5,
        }
    ]

    scores = {}

    for t in tests:
        print_name(t['name'])
        res = t['handler']()
        print_outcome(t['autolab'], res)
        scores[t['autolab']] = t['value'] if res else 0

    print(json.dumps({'scores': scores}))
