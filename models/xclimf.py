from math import exp, log
import numpy as np

def g(x):
    """sigmoid function"""
    return 1/(1+np.exp(-x))

def dg(x):
    """derivative of sigmoid function"""
    return np.exp(x)/(1+np.exp(x))**2

def precompute_f(data,U,V,m):
    """precompute f[j] = <U[m],V[j]>
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      m   : user of interest
    returns:
      dot products <U[m],V[j]> for all j in data[i]
    """
    k = data[m].indices
    v = np.dot(U[m], V[k].transpose())
    return (k, v)

def relevance_probability(r, maxi):
  """compute relevance probability as described xClimf paper
  params:
    r:   rating
    ma:  max rating
  """
  return (np.power(2,r)-1)/np.power(2,maxi)

def objective(data,U,V,lbda):
    """compute objective function F(U,V)
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      lbda: regularization constant lambda
    returns:
      current value of F(U,V)
    """
    maxi = data.max()
    obj = -0.5*lbda*(np.sum(U*U)+np.sum(V*V))

    for m in range(len(U)):
        (iks, fmi) = precompute_f(data, U, V, m)
        N = len(fmi)
        fmj = fmi.reshape(N,1)
        ymi = data[m, iks].toarray()
        rmi = relevance_probability(ymi, maxi)
        rmj = rmi.transpose()
        fmj_fmi = np.subtract(fmj, fmi)
        b1 = np.log(g(fmi))
        b2 = np.sum(np.log(1 - rmj * g(fmj_fmi)), axis=0)
        obj += np.dot(rmi, (b1 + b2))[0]

    return obj / len(U)

def update(data,Uo,Vo,lbda,gamma):
    """update user/item factors using stochastic gradient ascent
    params:
      data : scipy csr sparse matrix containing user->(item,count)
      Uo   : user factors
      Vo   : item factors
      lbda : regularization constant lambda
      gamma: learning rate
    """
    U = Uo.copy()
    V = Vo.copy()

    for m in range(len(U)):
        #Common variables used in both partial derivatives
        (iks, fmi) = precompute_f(data, U, V, m)
        N = len(fmi)
        fmk = fmi.reshape(N,1)
        fmi_fmk = np.subtract(fmi, fmk)
        fmk_fmi = np.subtract(fmk, fmi)
        ymi = data[m, iks].toarray()
        ymk = ymi.transpose()
        viks = V[iks]
        g_fmi = g(-1 * fmi)

        #Updating item vector
        div1 = 1/(1 - (ymk * g(fmk_fmi)))
        div2 = 1/(1 - (ymi * g(fmi_fmk)))
        brackets_i = g_fmi + np.sum(ymk * dg(fmi_fmk) * (div1 - div2), axis=0)
        dI = (ymi * brackets_i).transpose() * U[m] - lbda * viks
        Vo[iks] += gamma * dI

        #Updating user vector
        N2 = N*N
        brackets_ui = g_fmi.reshape(N, 1) * viks

        D = viks.shape[1]
        top = ymk * dg(fmk_fmi)
        bot = 1 - ymk * g(fmk_fmi)
        vis = np.tile(viks, (1, N)).reshape(N2, D)
        vks = np.tile(viks, (iks.shape[0], 1))
        sub = np.subtract(vis, vks)
        top_bot = (top / bot).transpose().reshape(N2, 1)
        brackets_uk = np.sum((top_bot * sub).reshape(N, N, D), axis=1)

        brackets_u = brackets_ui + brackets_uk
        dU = ymi.transpose() * brackets_u
        dU = np.sum(dU.transpose(), axis=1) - lbda * U[m]
        Uo[m] += (gamma * dU).transpose()


def compute_mrr(data,U,V,k=None):
    """compute average Mean Reciprocal Rank of data according to factors
    params:
      data      : scipy csr sparse matrix containing user->(item,count)
      U         : user factors
      V         : item factors
      the mean MRR over all users in data
    """
    mrr = []
    for m in range(data.shape[0]):
        if(len(data[m].indices) > 0):
            items = set(data[m].indices)
            predictions = np.sum(np.tile(U[m],(len(V),1))*V,axis=1)
            for rank,item in enumerate(np.argsort(predictions)[::-1]):
                if item in items:
                    mrr.append(1.0/(rank+1))
                    break
                elif k and k < rank+1:
                    mrr.append(0.0)
                    break
    return np.mean(mrr)

def gradient_ascent(train, test, params, foreach=None, eps=1e-4):
    D = params["dims"]
    lbda = params["lambda"]
    gamma = params["gamma"]
    iters = params.get("iters", 25)

    U = 0.01*np.random.random_sample((train.shape[0],D))
    V = 0.01*np.random.random_sample((train.shape[1],D))

    last_objective = float("-inf")
    note = None
    for i in range(iters):
        update(train, U, V, lbda, gamma)
        obj = objective(train, U, V, lbda)
        if foreach:
            __ = foreach(i, obj, U, V, train, test, params)
            if __ is not None:
                note = __
        if obj > last_objective:
            last_objective = obj
        elif obj < last_objective + eps:
            print("objective should be bigger or equal last objective...")
            break

    return (U, V), note
