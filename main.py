import numpy as np
from copy import deepcopy
from time import time
from scipy import optimize
from time import time


def rosenbrock(a, b, f0, n):
   return lambda x: sum(a*(x[i]**2 - x[i+1])**2 + b*(x[i] - 1)**2 for i in range(n-1))+f0


def gradient_descent(f, x0, eps_1=1e-4, eps_2=1e-4, eps_grad=1e-4, n_limit=1e3):
    k = 0
    x = x0
    end = False
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)

    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k


        x_old = deepcopy(x)
        # left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        # a = half(f_directed(x, -grad_f), left, right)
        a = np.max(optimize.minimize(f_directed(x, -grad_f), 1).x)
        # print(a)
        x -=  a * grad_f

        if np.linalg.norm(x-x_old) < eps_1 and np.abs(f(x)-f(x_old)) < eps_2:
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False



def outer_penalty(f, cond_eq, cond_ineq ,x0, r0=1., C = 4., eps=1e-3):
    x = deepcopy(x0)
    r = deepcopy(r0)
    k = 1
    while True:
        penalty_func = lambda r: lambda x: f(x) + sum([max(cond(x), 0) for cond in cond_ineq])
        x, _ = gradient_descent(penalty_func(r), x)
        # print(x)

        loss = penalty_func(r)(x) - f(x)

        if np.abs(loss) <= eps:
            return x, k
        else:
            r = C*r
            k += 1


def inner_penalty(f, cond_eq, cond_ineq ,x0, r0 = 1, C = 4, eps=1e-3):
    x = deepcopy(x0)
    r = deepcopy(r0)
    k = 0
    while True:
        # print(cond_ineq[0](x))
        penalty_func = lambda r: lambda x: f(x) - r*sum([1/cond(x) for cond in cond_ineq])
        x, _ = gradient_descent(penalty_func(r), x)

        loss = penalty_func(r)(x) - f(x)

        if np.abs(loss) <= eps:
            return x, k
        else:
            r = r/C
            k += 1



def combined_penalty(f, cond_eq, cond_ineq ,x0, r0=1., C = 4., eps=1e-3):
    x = deepcopy(x0)
    r_o, r_i = deepcopy(r0), deepcopy(r0)
    k = 0
    while True:
        penalty_func = lambda r_o, r_i: lambda x: f(x) + (r_o/2)*(np.sum([abs(cond(x)) for cond in cond_eq]) +
                                                         np.sum([max(cond(x), 0) for cond in cond_ineq])) \
                                           - r_i*np.sum([1./cond(x) for cond in cond_ineq])

        x, _ = gradient_descent(penalty_func(r_o, r_i), x)

        loss = penalty_func(r_o, r_i)(x) - f(x)

        if np.abs(loss) <= eps:
            return x, k
        else:
            r_o = C*r_o
            r_i = r_i/C
            k += 1


def lagrange_opt(f, cond_eq, cond_ineq ,x0, cond_eq_par, cond_ineq_par, r0=1., C = 4., eps=1e-5):
    x = deepcopy(x0)
    r = deepcopy(r0)
    k = 1
    while True:
        penalty_func = lambda x: f(x) + np.sum(np.abs(cond_eq_par*np.asarray([cond(x) for cond in cond_eq]))) \
                                + (r/2)*np.sum([cond(x)**2 for cond in cond_eq]) \
                                + (1./(2*r))*np.sum((np.asarray([max(par+r*cond(x), 0.) for cond, par in zip(cond_ineq, cond_ineq_par)]))**2-\
                                                    cond_ineq_par**2)
        x, _ = gradient_descent(penalty_func, x)

        loss = penalty_func(x) - f(x)

        if np.abs(loss) <= eps:
            return x, k
        else:
            r = C*r
            cond_eq_par += r*np.asarray([cond(x) for cond in cond_eq])
            cond_ineq_par += r**np.asarray([cond(x) for cond in cond_eq])
            cond_ineq_par = np.asarray([max(par, 0) for par in cond_ineq_par])
            k += 1

def main():
    f = rosenbrock(30, 2, 80, 4)
    c_ineq = [lambda x: -x[i] for i in range(4)] + [lambda x: x[0]**2 + x[1]**2 - 3*x[2] + x[3]**2]
    c_eq = []
    c_eq_1_par = np.array([0.])
    c_ineq_1_par = np.array([0., 0.])


    start = time()
    res = outer_penalty(f, c_eq, c_ineq, np.array([1.32, 1.2, 1.4, 1.5]))
    a = np.array(res[0])
    t = time() - start
    print(f'For outer penalty: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iterations')


    start = time()
    res = inner_penalty(f, c_eq, c_ineq, np.array([1.32, 1.2, 1.4, 1.5]))
    a = np.array(res[0])
    t = time() - start
    print(f'For inner penalty: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iterations')


    start = time()
    res = combined_penalty(f, c_eq, c_ineq, np.array([1.32, 1.2, 1.4, 1.5]))
    a = np.array(res[0])
    t = time() - start
    print(f'For combined penalty: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iterations')


    start = time()
    res = lagrange_opt(f, c_eq, c_ineq, np.array([1.32, 1.2, 1.4, 1.5]), c_eq_1_par, c_ineq_1_par)
    a = np.array(res[0])
    t = time() - start
    print(f'For Lagrange optimization: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iterations')


if __name__ == '__main__':
    main()
