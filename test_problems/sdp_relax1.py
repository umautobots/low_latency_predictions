import numpy as np

EPS = 0.001


def make_instance(n=5, eps=0.1, g_lb=(), k_ub=1, seed=0):
    np.random.seed(seed)
    vl_v = np.random.rand(n)*10 - 10
    if len(g_lb) == 0:
        lb_ = np.random.rand()*10 + eps
        g_lb = (lb_, lb_ + np.random.rand()*5)
    g_star = np.random.rand() * (g_lb[1] - g_lb[0]) + g_lb[0]
    g = np.sort(np.random.rand(n))*20 + eps
    # x = [kv, kg, kg*g, g]
    A = np.array([
        vl_v, g, 0*g - 1, 0*g
    ]).T
    A = np.vstack((A, np.array([0, 0, 0, 1.])))
    kv, kg = np.random.rand(2) * (k_ub - eps)
    u = kg * g_star
    x_true = np.array([kv, kg, u, g_star])
    b = A.dot(x_true)

    def f(x):
        return 0.5 * np.linalg.norm(A.dot(x) - b)**2

    return A, b, x_true, f, g_lb, k_ub


def solve(A, b, g_lb, k_ub):
    """
    :param A:
    :param b: 
    :param lb: 
    :param ub: 
    :return: 
    """
    import cvxopt as cvx
    import picos as pic
    # from pprint import pprint

    # x = [kv, kg1, u1 g]
    # kgi*g - ui = 0
    m = A.shape[1]
    # print(np.linalg.cond(A))
    print(A)
    B1 = np.zeros((m, m))
    B1[1, 3] = B1[3, 1] = 1
    q1 = np.array([0, 0, -1, 0])
    A, b, B1, q1 = [cvx.matrix(arr.copy()) for arr in [A, b, B1, q1]]

    prob = pic.Problem()
    x = prob.add_variable('x', m)
    X = prob.add_variable('X', (m, m), vtype='symmetric')
    prob.add_constraint(x >= EPS)
    prob.add_constraint(x[:2] <= k_ub)
    prob.add_constraint(x[3] <= g_lb[1])
    prob.add_constraint(x[3] >= g_lb[0])
    # [[1, x']; [x, X]] > 0
    prob.add_constraint(((1 & x.T) // (x & X)) >> 0)
    prob.add_constraint(0.5 * (X | B1) + q1.T * x == 0)

    prob.set_objective('min', (X | A.T * A) - 2 * b.T * A * x)
    # from pprint import pprint
    # print(prob)
    prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False)  # , feastol=EPS/2)
    x_hat = np.array(x.value).T[0]
    assert (x_hat >= EPS).all() and (x_hat[:2] <= k_ub).all()
    return x_hat


def main():
    from time import time
    seed = np.random.randint(0, 1000)
    # seed = 0
    print('seed: {}'.format(seed))
    A, b, x_true, f, lb, ub = make_instance(seed=seed)
    print(x_true)
    t0 = time()
    x_hat = solve(A, b, lb, ub)
    print('time elapsed: {:0.5f}'.format(time() - t0))
    print(np.round(x_hat, 2))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))


if __name__ == '__main__':
    main()
