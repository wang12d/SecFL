# from ABY.ABY import main
import ctypes
import numpy as np

from numpy import number

secureLR = ctypes.CDLL("ABY/build/lib/libsecureLR.so")
# secureLR.loss_second_communicate.restype = ctypes.c_double
# secureLR.loss_mu_computation.restype = ctypes.c_double
secureLR.loss_first_computation.restype = ctypes.c_double

number = 3



for i in range(1000):
    input = (ctypes.c_double * number)(*[1.0, 2.0, 3.0])
    # w = (ctypes.c_double * number)(*[])
    # secureLR.grad_communicate(0, number, input, input)
    # w = np.array(w, dtype=np.float32)
    # if(ret != 13.0):
    #     print("Error: " + str(ret))
    #     assert(False)
    secureLR.loss_first_computation(0, number, input)
