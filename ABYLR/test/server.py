# from ABY.ABY import main
import ctypes
import numpy as np

secureLR = ctypes.CDLL("ABY/build/lib/libsecureLR.so")
# secureLR.loss_second_communicate.restype = ctypes.c_double
# secureLR.loss_mu_computation.restype = ctypes.c_double
secureLR.loss_first_computation.restype = ctypes.c_double


number = 3

for i in range(1000):
    input = (ctypes.c_double * number)(*[1.0, 2.0, 3.0])
    # w = (ctypes.c_double * number)(*[])
    # secureLR.grad_communicate(1, number, input, w)
    # w = np.array(w, dtype=np.float32)
    # print(f"{i}:{w}")
    print(secureLR.loss_first_computation(1, number, input))
