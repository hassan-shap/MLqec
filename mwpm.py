import numpy as np
import pymatching
from code_utils import code_initialization, idx_to_coord
from datasets import load_dataset
import json
import time

p_err = 0.1
d_list = [11]
p_err_list = np.arange(0.01,0.12,0.01)
logical_err = np.zeros((p_err_list.shape[0],len(d_list)))
for i_d, d in enumerate(d_list):
    print(f"code dist: {d}")
    tic = time.time()
    data_list = {}
    JSON_PATH = f"results/mwpm_sweep_d_{d}.json"
    data_list["p_err"] = p_err_list.tolist()
    

    q_list, sx_list, sz_list = idx_to_coord(d)
    stab_matrices, s_mat, logicals = code_initialization(d)
    x_stab_to_c,z_stab_to_c,sx_list,sz_list = stab_matrices
    weights = np.ones(d**2) * np.log((1-p_err)/p_err)
    ind = (d**2-1)//2
    matching_x = pymatching.Matching(s_mat[:ind,:d**2],spacelike_weights=weights)
    matching_z = pymatching.Matching(s_mat[ind:,d**2:],spacelike_weights=weights)

    for i_p, p_err in enumerate(p_err_list):
        dataset = load_dataset("json", data_files={
                # 'test' : f"datasets/test_d_{d}_p_{p_err:.2f}.json"
                'test' : f"datasets/test_spiral_eos_d_{d}_p_{p_err:.2f}.json",
            })
        err_list = dataset["test"]["errors"]
        # print(err_list)
        for i in range(len(err_list)):
            x_err = err_list[i]['x']
            z_err = err_list[i]['y']
            y_err = err_list[i]['z']
            # print(z_err)
            # print(err_instance)
            err_vec = np.zeros(2*d**2)
            if len(x_err)> 0:
                err_vec[x_err] = 1
            if len(z_err)> 0:
                err_vec[d**2 + np.array(z_err)] = 1
            if len(y_err)> 0:
                err_vec[y_err] = 1
                err_vec[d**2 + np.array(y_err)] = 1

            comm_mat = np.kron([[0,1],[1,0]],np.eye(d**2))
            syndrome = (s_mat@ (comm_mat @err_vec)) % 2
            active_syndrome_idx = np.argwhere(syndrome>0)[:,0]

            syndrome_x = syndrome[:ind]
            syndrome_z = syndrome[ind:]

            recovery_x = matching_x.decode(syndrome_x)
            recovery_z = matching_z.decode(syndrome_z)

            idx_recovery_z = np.argwhere(recovery_z>0)[:,0]
            idx_recovery_x = np.argwhere(recovery_x>0)[:,0]

            err_rec = np.copy(err_vec)
            err_rec[idx_recovery_z] += 1
            err_rec[d**2 + idx_recovery_x] += 1
            err_rec %= 2

            logical_err[i_p,i_d] += np.sum(logicals@ (comm_mat @ err_rec) % 2)>0

        logical_err[i_p,i_d] /= len(err_list)
    data_list["logical_err"] = logical_err[:,i_d].tolist()
    with open(JSON_PATH, 'w') as json_file:
        json_file.write(json.dumps(data_list) + '\n')
    toc = time.time()
    print(f"Finished in {toc-tic} sec.")

# print(logical_err)