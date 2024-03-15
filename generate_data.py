import json
import numpy as np
import random
from code_utils import code_initialization, decoder

def data_file_gen(d,JSON_PATH):
    # logical_err_list = np.zeros(Niter)

    stab_matrices, s_mat, logicals = code_initialization(d)
    #  matrix to calculate the commutation relation in stabilizer formalism
    comm_mat = np.kron([[0,1],[1,0]],np.eye(d**2))
    #####################################################

    ### simulation
    pauli = [0,1,2,3] # I X Z Y
    # JSON_PATH = f"datasets/test_d_{d}_p_{p_err:.2f}.json"
    qlist = {}
    with open(JSON_PATH, 'w') as json_file:

        for iter in range(Niter):
            err_instance = np.array(random.choices(pauli, [1-p_err,p_err/3,p_err/3,p_err/3], k=d**2))
            x_err = np.argwhere(err_instance==1)[:,0]
            z_err = np.argwhere(err_instance==2)[:,0]
            y_err = np.argwhere(err_instance==3)[:,0]
            err_dict = {}
            err_dict["x"] = x_err.tolist()
            err_dict["z"] = z_err.tolist()
            err_dict["y"] = y_err.tolist()
            # err_list.append(err_dict)

            err_vec = np.zeros(2*d**2)
            err_vec[x_err] = 1
            err_vec[d**2 + z_err] = 1
            err_vec[y_err] = 1
            err_vec[d**2 + y_err] = 1

            syndrome = (s_mat@ (comm_mat @err_vec)) % 2
            active_syndrome_idx = np.argwhere(syndrome>0)[:,0]
            # syndrome_list.append(active_syndrome_idx.tolist())

            recovery_x, recovery_z = decoder(d, stab_matrices, active_syndrome_idx)

            err_rec = np.copy(err_vec)
            err_rec[recovery_z] += 1
            err_rec[d**2 + recovery_x] += 1
            err_rec %= 2

        # checking if there is a logical error
            logical_err_list = (logicals@ (comm_mat @ err_rec) % 2)
        
            qlist['errors'] = err_dict 
            qlist['input'] = syndrome.tolist() 
            qlist['target'] = logical_err_list.tolist()

            # Serializing json
            json_file.write(json.dumps(qlist) + '\n')
            # print("Done!")

d_list = [3] # code distance, must be an odd number 
Niter = 1000 # number of random iterations for error
p_err_list = np.arange(0.01,0.31,0.01)
for p_err in p_err_list:
    # p_err = 0.15 # error probability (depolarizing channel)
    for d in d_list:
        # # qubit indices as defined in the paper
        # q_func = lambda t,r,c,i: int(((d-1)/2+r*(i+1)+1)*(t*d+(1-t)) -1 + 2*c*(d*(1-t)-t))
        # # ancilla indices as defined in the paper
        # a_func = lambda t,r,c,i: int((d**2-1)/4*(1+2*t) + ((r-1)/2+r*i)*(d+1)/2 +c )
        # fname = f"datasets/train_d_{d}_p_{p_err:.2f}.json"
        fname = f"datasets/test_d_{d}_p_{p_err:.2f}.json"
        data_file_gen(d,fname)