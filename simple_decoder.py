import numpy as np
import random


def idx_to_coord(d):
    """
    This is to keep track of index assignment to data and ancilla qubits
    c.f. Figure 9 of arXiv:2202.05741
    """
    q_list = {}
    sz_list = {}
    c_sz = (d**2-1)-1
    for row in range(d):
        for col in range(d):
            x, y = col-(d-1)/2, row-(d-1)/2
            q_list[f"{d**2-d+col-d*row}"] = (x,y)
            if row %2 ==1 and col % 2 == 0:
                x_sz, y_sz = x+0.5, y+ 0.5
                sz_list[f"{c_sz}"] = (x_sz, y_sz)
                c_sz -= 1
            if row %2 ==0 and col % 2 == 0 and row < d-1:
                x_sz, y_sz = x-0.5, y+ 0.5
                sz_list[f"{c_sz}"] = (x_sz, y_sz)
                c_sz -= 1

    c_q = 0
    sx_list = {}
    c_sx = 0# (d**2-1)-1
    for col in range(d):
        for row in range(d):
            x, y = col-(d-1)/2, (d-1)/2-row
            if row %2 ==0 and col % 2 == 1:
                x_s, y_s = x+0.5, y- 0.5
                sx_list[f"{c_sx}"] = (x_s, y_s)
                c_sx += 1
            if row %2 ==0 and col % 2 == 0 and col < d-1:
                x_s, y_s = x+0.5, y+ 0.5
                sx_list[f"{c_sx}"] = (x_s, y_s)
                c_sx += 1
    return q_list, sx_list, sz_list


def decoder(d, stab_matrices, syndrome):
    """
        Simple decoder as defined in the paper.
    """
    x_stab_to_c,z_stab_to_c,sx_list,sz_list= stab_matrices 
    syndrome_x = syndrome[np.argwhere(syndrome< (d**2-1)//2)[:,0]]
    syndrome_z = syndrome[np.argwhere(syndrome>= (d**2-1)//2)[:,0]]
    
    t = 0
    recovery_x = np.zeros(d**2)
    trc_list = []
    for s in syndrome_x:
        c = x_stab_to_c[s]
        r = -1 if s< (d**2-1)//4 else 1
        trc_list.append((t,r,c))

    trc_list = list(set(trc_list))
    for t,r,c in trc_list:
        i = 0
        recovery_x[q_func(t,r,c,i)] += (a_func(t,r,c,i) in syndrome_x)
        for i in range(1,(d-1)//2 ):
            recovery_x[q_func(t,r,c,i)] += (a_func(t,r,c,i) in syndrome_x) + recovery_x[q_func(t,r,c,i-1)]
    recovery_x %= 2 

    t = 1
    recovery_z = np.zeros(d**2)
    trc_list = []
    for s in syndrome_z:
        c = z_stab_to_c[s-(d**2-1)//2]
        # r = -1 if s< (d**2-1)//4 else 1
        r = -1 if sz_list[f"{s}"][1]> 0 else 1
        trc_list.append((t,r,c))

    trc_list = list(set(trc_list))
    for t,r,c in trc_list:
        i = 0
        recovery_z[q_func(t,r,c,i)] += (a_func(t,r,c,i) in syndrome_z)
        for i in range(1,(d-1)//2 ):
            recovery_z[q_func(t,r,c,i)] += (a_func(t,r,c,i) in syndrome_z) + recovery_z[q_func(t,r,c,i-1)]
    recovery_z %= 2 


    return np.argwhere(recovery_x>0)[:,0], np.argwhere(recovery_z>0)[:,0]

def code_initialization(d):
    """
        Calculate necessary mappings from qubit indices to qubit location,
        stabilizer matrix
        logical operator matrix
        input:
            d :int  surface code distance
        output:
            s_mat : X, Z stabilizer matrix
            logicals : X, Z logical operator matrix
    """
    c_list = np.arange((d+1)//2) # c as defined in the paper

    x_stab_to_c = np.zeros((d**2-1)//2,dtype=int)
    t = 0
    for r in [-1,1]:
        for c in c_list:
            for i in range((d-1)//2 ):
                x_stab_to_c[a_func(t,r,c,i)] = c

    z_stab_to_c = np.zeros((d**2-1)//2,dtype=int)
    t = 1
    for r in [-1,1]:
        for c in c_list:
            for i in range((d-1)//2 ):
                z_stab_to_c[a_func(t,r,c,i)-(d**2-1)//2] = c

    q_list, sx_list, sz_list = idx_to_coord(d)

    # define stabilizers 
    s_mat = np.zeros(((d**2-1), 2*d**2))
    for sx in sx_list:
        r_sx = sx_list[sx]
        for q in q_list:
            r_q = q_list[q]
            if (r_q[0]-r_sx[0])**2+(r_q[1]-r_sx[1])**2 < 1:
                # print(sx,q)
                s_mat[int(sx),int(q)] = 1

    for sz in sz_list:
        r_sz = sz_list[sz]
        for q in q_list:
            r_q = q_list[q]
            if (r_q[0]-r_sz[0])**2+(r_q[1]-r_sz[1])**2 < 1:
                # print(sx,q)
                s_mat[int(sz),d**2+ int(q)] = 1

    # define logical operators
    logicals = np.zeros((2,2*d**2))
    logicals[0,np.arange(0,d**2,d)]= 1 # x logical
    logicals[1,d**2+np.arange(d**2-d, d**2)]= 1 # z logical

    stab_matrices = (x_stab_to_c,z_stab_to_c,sx_list,sz_list)
    return stab_matrices, s_mat, logicals
#################################################
#################################################

d = 7 # code distance, must be an odd number 
Niter = 100 # number of random iterations for error
p_err = 0.2 # error probability (depolarizing channel)
logical_err_list = np.zeros((Niter, 2))
# qubit indices as defined in the paper
q_func = lambda t,r,c,i: int(((d-1)/2+r*(i+1)+1)*(t*d+(1-t)) -1 + 2*c*(d*(1-t)-t))
# ancilla indices as defined in the paper
a_func = lambda t,r,c,i: int((d**2-1)/4*(1+2*t) + ((r-1)/2+r*i)*(d+1)/2 +c )

stab_matrices, s_mat, logicals = code_initialization(d)
#  matrix to calculate the commutation relation in stabilizer formalism
comm_mat = np.kron([[0,1],[1,0]],np.eye(d**2))
#####################################################

### simulation
pauli = [0,1,2,3] # I X Z Y
for iter in range(Niter):
    err_instance = np.array(random.choices(pauli, [1-p_err,p_err/3,p_err/3,p_err/3], k=d**2))
    x_err = np.argwhere(err_instance==1)[:,0]
    z_err = np.argwhere(err_instance==2)[:,0]
    y_err = np.argwhere(err_instance==3)[:,0]

    err_vec = np.zeros(2*d**2)
    err_vec[x_err] = 1
    err_vec[d**2 + z_err] = 1
    err_vec[y_err] = 1
    err_vec[d**2 + y_err] = 1

    syndrome = (s_mat@ (comm_mat @err_vec)) % 2
    active_syndrome_idx = np.argwhere(syndrome>0)[:,0]

    recovery_x, recovery_z = decoder(d, stab_matrices, active_syndrome_idx)

    err_rec = np.copy(err_vec)
    err_rec[recovery_z] += 1
    err_rec[d**2 + recovery_x] += 1
    err_rec %= 2

# checking if there is a logical error
    logical_err_list[iter,:] = logicals@ (comm_mat @ err_rec) % 2

print(logical_err_list)