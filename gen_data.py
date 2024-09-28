import json
import numpy as np
import random
import time
from code_utils import code_initialization, decoder, spiral_coord

def data_file_gen(d,JSON_PATH):
    # logical_err_list = np.zeros(Niter)

    stab_matrices, s_mat, logicals = code_initialization(d)
    #  matrix to calculate the commutation relation in stabilizer formalism
    comm_mat = np.kron([[0,1],[1,0]],np.eye(d**2))
    #####################################################

    perm_mat = spiral_coord(d, stab_matrices)
    # shift by one
    perm_mat = (np.array(perm_mat)+1).astype(int).tolist()
    ### simulation
    pauli = [0,1,2,3] # I X Z Y
    # JSON_PATH = f"datasets/test_d_{d}_p_{p_err:.2f}.json"
    qlist = {}
    with open(JSON_PATH, 'w') as json_file:

        for _ in range(Niter):
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
            syndrome[(d**2-1)//2:] = 2*syndrome[(d**2-1)//2:]
            syndrome_train = np.zeros((d+1)**2+2 )
            syndrome_train[perm_mat] = syndrome
            syndrome_train[(d+1)**2+1:] = 3
            syndrome_train[0] = 3

            # print(syndrome_train)
            recovery_x, recovery_z = decoder(d, stab_matrices, active_syndrome_idx)

            err_rec = np.copy(err_vec)
            err_rec[recovery_z] += 1
            err_rec[d**2 + recovery_x] += 1
            err_rec %= 2

        # checking if there is a logical error
            logical_err_list = (logicals@ (comm_mat @ err_rec) % 2)
        
            qlist['dist'] = d 
            qlist['errors'] = err_dict 
            qlist['input'] = syndrome_train.tolist() 
            qlist['target'] = logical_err_list.tolist()
            # print("syndrome_train after: ", syndrome_train[perm_mat[ids[:num_mask]]])
            
            # Serializing json
            json_file.write(json.dumps(qlist) + '\n')
            # print("Done!")

### training data
tic = time.time()
d_list = [5,7,9] # code distance, must be an odd number 
Niter = 200000 # number of random iterations for error
p_err_list =  [0.05] #np.arange(0.01,0.31,0.01) # 
for p_err in p_err_list:
    for d in d_list:
        fname = f"datasets/train_spiral_eos_d_{d}_p_{p_err:.2f}.json"
        data_file_gen(d,fname)
toc = time.time()
print(f"training data generation is complete, {toc-tic} sec.")

### validation data
# tic = time.time()
# # d_list = [5,7,9,11] # code distance, must be an odd number 
# Niter = 10000 # number of random iterations for error
# p_err_list =  [0.15] #np.arange(0.01,0.31,0.01) # 
# for p_err in p_err_list:
#     for d in d_list:
#         fname = f"datasets/val_spiral_eos_d_{d}_p_{p_err:.2f}.json"
#         data_file_gen(d,fname)
# toc = time.time()
# print(f"val data generation is complete, {toc-tic} sec.")

# ### test data
# tic = time.time()
# d_list = [5,7,9,11] # code distance, must be an odd number 
# Niter = 100000 # number of random iterations for error
# p_err_list =  np.arange(0.008,0.21,0.008) # 
# for p_err in p_err_list:
#     for d in d_list:
#         fname = f"datasets/test_spiral_eos_d_{d}_p_{p_err:.3f}.json"
#         data_file_gen(d,fname)
# toc = time.time()
# print(f"test data generation is complete, {toc-tic} sec.")
