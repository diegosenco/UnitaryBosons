'''
program: perform_test.py
created: 2016-06-02 -- 15 CEST
author: tc
'''

import random

from hcorr_integrals_cy import HCorr as HCorr_cy

def random_vector():
    return [round(random.uniform(-2.0, 2.0), 5) for i in [0, 1, 2]]


ntests = 20

for itest in range(ntests):
    A = random_vector()
    B = random_vector()
    E = random_vector()
    k = random_vector()
    beta_star = round(random.uniform(0.4, 2.5), 5)
    HC = HCorr_cy(A, B, E, beta_star)
    HC.full_check(k, ID='test_%02i' % itest)
