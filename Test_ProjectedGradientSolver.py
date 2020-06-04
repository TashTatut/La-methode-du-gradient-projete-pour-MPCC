from ProjectedGradientSolver import *
import numpy as np


############$$$$$$$$Problem ex9.1.2.mod###############################
#set I := 1..4;

#var x >= 0;
#var y binary;
#var s{I} >= 0;
#var l{I} >= 0;

# ... Outer Objective function

#minimize objf: - x -3*y;

#subject to 
   # ... Inner Problem    Constraints
#   c2:  - x +   y + s[1] = 3;
#   c3:    x + 2*y + s[2] = 12;
#   c4:  4*x -   y + s[3] = 12;
#   c5:       -  y + s[4] = 0;

   # ... KKT conditions for the inner problem optimum
#   kt1:  l[1] + 2*l[2] - l[3] - l[4] = -1;
 
#  compl {i in I}: 0 <= l[i]   complements   s[i] >= 0;
######################################################################################

def probex921(x):
    fobj = -x[0] - 3 * x[1]
    h = [-x[0] + x[1] + x[2] - 3,#eta[0]
         x[0] + 2 * x[1] + x[3] - 12,#eta[1]
         4 * x[0] - x[1] + x[4] - 12,#eta[2]
         -x[1] + x[5],#etq[3]
         x[6] + 2 * x[7] - x[8] - x[9] + 1,#eta[4]
         (x[2] - rp) * x[6] + x[10],#eta[5]
         (x[3] - rp) * x[7] + x[11],#eta[6]
         (x[4] - rp) * x[8] + x[12],#eta[7]
         (x[5] - rp) * x[9] + x[13]#eta[8]
         ]

    h = np.array(h)
    f = fobj + h.dot(eta) + (rho / 2) * (h**2).sum()

    g = [-1 - eta[0] +eta[1] + 4*eta[2] + rho*(-h[0] + h[1] + 4*h[2]),
         -3 + eta[0] + 2*eta[1] - eta[2] -eta[3] + rho*(h[0] + 2*h[1] - h[2] - h[3]),
         eta[0] + eta[5]*x[6] + rho*(h[0] + x[6]*h[5]),
         eta[1] + eta[6]*x[7] + rho*(h[1] + x[7]*h[6]),
         eta[2] + eta[7]*x[8] + rho*(h[2] + x[8]*h[7]),
         eta[3] + eta[8]*x[9] + rho*(h[3] + x[9]*h[8]),
         eta[4] + eta[5]*(x[2] - rp) + rho*(h[4] + (x[2] - rp)*h[5]),
         2*eta[4] + eta[6] * (x[3] - rp) + rho * (2*h[4] + (x[3] - rp) * h[6]),
         -eta[4] + eta[7] * (x[4] - rp) + rho * (-h[4] + (x[4] - rp) * h[7]),
         -eta[4] + eta[8] * (x[5] - rp) + rho * (-h[4] + (x[5] - rp) * h[8]),
         eta[5] + rho*h[5],
         eta[6] + rho*h[6],
         eta[7] + rho* h[7],
         eta[8] + rho*h[8]
         ]
    g = np.array(g)
    return [f, g]
    
    
    #lb = np.array([-np.inf, -np.inf, 0, 0, 0, 0, -rp, -rp, -rp, -rp, 0, 0, 0, 0])
    def construct_lb(rp, n, ni, nc):
    lst = []
    for i in range(n+ni+3*nc):
        if i < n:
            lst.append(-np.inf)
        elif (i >= n) and (i < n + ni + nc):
            lst.append(0.0)
        elif (i >= n + ni + nc) and (i < n + ni + 2*nc):
            lst.append(-rp)
        else:
            lst.append(0.0)
    return np.array(lst)



def get_working_set(current_x, n, ni, nc, epsilon=1e-6):
    A = np.zeros((ni+3*nc, n))
    B = np.eye(ni+3*nc)
    mat_all = np.block([A, B])
    lb = construct_lb(rp, n, ni, nc)
    index_ws =[]
    count = 0
    for row in mat_all:
        if abs( row.dot(current_x) - lb[count] ) <= epsilon:
            index_ws.append(count)
        count += 1

    mat_ws = np.delete(mat_all, index_ws, axis=0)
    return [index_ws, mat_ws]



rp = 10
rho = 1
eta = np.zeros((9,))


def my_test0():
    global eta
    global rho
    global rp
    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 0
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.maxIter = 1000
    spg_options.verbose = 1

    x_init = np.array([-1, -3, 1, -1, 2, -4, 1, -2, - 6, 9, 3, -8, 2, 7])
    lb = construct_lb(rp, 2, 0, 4)
    funObj = lambda x: probex921(x)

    funProj = lambda x: project_bound(x, lb)

    eqConstr = lambda x: np.array([-x[0] + x[1] + x[2] - 3,#eta[0]
         x[0] + 2 * x[1] + x[3] - 12,#eta[1]
         4 * x[0] - x[1] + x[4] - 12,#eta[2]
         -x[1] + x[5],#etq[3]
         x[6] + 2 * x[7] - x[8] - x[9] + 1,#eta[4]
         (x[2] - rp) * x[6] + x[10],#eta[5]
         (x[3] - rp) * x[7] + x[11],#eta[6]
         (x[4] - rp) * x[8] + x[12],#eta[7]
         (x[5] - rp) * x[9] + x[13]#eta[8]
         ])
         
    x_plus = x_init
    flist =[]
    xlist =[]
    etalist = []
    for i in range(20):

        x_prev = x_plus
        x_plus, fvalue = ProjectedGradientSolver(funObj, funProj, x_prev, spg_options)

        eta = eta + rho * eqConstr(x_plus)
        if np.amax(np.absolute(eqConstr(x_plus))) > 0.95 * np.amax(np.absolute(eqConstr(x_prev))):
            rho = 7 * rho
        rp = 0.1 * rp
        flist.append(fvalue)
        xlist.append(x_plus)
        etalist.append(eta)
    return xlist, flist, etalist


if __name__ == '__main__':

    n = 2
    ni = 0
    nc = 4
    xlist, flist , etalist = my_test0()

    g = probex921(xlist[-1])[1]
    index_ws, mat_ws = get_working_set(xlist[-1], n, ni, nc, epsilon=1e-6)
   
    print('final point :{}'.format(xlist[-1]))
    print("final objective : {}".format(flist[-1]))
    print('final working set : {}'.format(index_ws))
    print('final gradient subset estimate:{}' .format(mat_ws.dot(g)))
    print('final lagrange multiplier estimate:{}'.format(etalist[-1]))

############################  OUTPUT   ###################################

#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#First-Order Optimality Conditions Below optTol
#final point :[ 1.99999924e+00  5.00000060e+00  0.00000000e+00  0.00000000e+00
#  9.00000388e+00  5.00000228e+00 -2.00004119e-01 -4.00007803e-01
# -4.43028915e-06 -1.66032429e-05  0.00000000e+00  0.00000000e+00
#  3.94659537e-05  8.33507574e-05]
#final objective : -16.999999998129557
#final working set : [8, 9]
#final gradient subset estimate:[ 2.95326832e-01  1.18142974e+00 -9.35082995e-06  5.39109317e-06
#  5.30826215e-06  1.06165243e-05 -1.12814233e-06  1.14576874e-06
#  4.64478657e-07  1.29084357e-06]
#final lagrange multiplier estimate:[ 3.33304426e-01  1.33334494e+00 -9.58306214e-06  3.70637772e-06
#  3.99918577e-06  1.89890859e-01  3.79781669e-01  8.71144470e-07
#  9.56338576e-07]

#Process finished with exit code 0
#final gradient subset estimate:[ 2.95378985e-01  1.18140006e+00  9.73937811e-06 -1.74916210e-05
#  3.65604490e-06  7.31208981e-06 -2.16953686e-06 -2.07927216e-05
#  1.65167561e-07 -3.42733535e-06]
#final lagrange multiplier estimate:[ 3.33359905e-01  1.33331561e+00  1.03430766e-05 -1.68009466e-05
#  3.00633891e-06  1.89890860e-01  3.79781670e-01  4.47519612e-07
# -5.28092228e-07]
