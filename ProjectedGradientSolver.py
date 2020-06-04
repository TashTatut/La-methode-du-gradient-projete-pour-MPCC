import numpy as np
import sys

# Options:
# verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:debug)
# optTol: tolerance used to check for optimality
# progTol: tolerance used to check for lack of progress
# maxIter: maximum number of calls to funObj
# suffDec: sufficient decrease parameter in Armijo condition
# curvilinear: backtrack along projection arc
# memory: number of steps to look back in non-monotone Armijo condition
# bbType: type of Barzilai-Borwein step
# interp: 0=none, 2=cubic (for the most part.. see below)
# numDiff: compute derivatives numerically (0: use user-supplied derivatives (default), 1: use finite differences)

# June 04th 2020
class PGSolverOptions():
        pass

default_options = PGSolverOptions()
default_options.maxIter = 500
default_options.verbose = 2
default_options.suffDec = 1e-4
default_options.progTol = 1e-9
default_options.optTol = 1e-5
default_options.memory = 10
default_options.useSpectral = True
default_options.bbType = 1
default_options.interp = 0  # Halfing stepsize
default_options.numdiff = 0
default_options.testOpt = True


def log(options, level, msg):
    if options.verbose >= level:
        print(msg, file=sys.stderr)


def assertVector(v):
    assert len(v.shape) == 1


defProjectedGradientSolver(funObj0, funProj, x, options=default_options):
    x = funProj(x)
    i = 1  # iteration

    funEvalMultiplier = 1
    if options.numdiff == 1:
        funObj = lambda x: auto_grad(x, funObj0, options)
        funEvalMultiplier = len(x)+1
    else:
        funObj = funObj0

    f, g = funObj(x)
    projects = 1
    funEvals = 1

    if options.verbose >= 2:
        if options.testOpt:
            print('%10s %10s %10s %15s %15s %15s' % ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val', 'Opt Cond'))
        else:
            print('%10s %10s %10s %15s %15s' % ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val'))

    while funEvals <= options.maxIter:
        if i == 1 or not options.useSpectral:
            alpha = 1.0
        else:
            y = g - g_old
            s = x - x_old
            assertVector(y)
            assertVector(s)

            # type of BB step
            if options.bbType == 1:
                alpha = np.dot(s.T, s) / np.dot(s.T, y)
            else:
                alpha = np.dot(s.T, y) / np.dot(y.T, y)

            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1.0

        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        d = funProj(x + d) - x
        projects += 1

        gtd = np.dot(g, d)

        if gtd > -options.progTol:
            log(options, 1, 'Directional Derivative below progTol')
            break

        if i == 1:
            t = min([1, 1.0 / np.sum(np.absolute(g))])
        else:
            t = 1.0

        if options.memory == 1:
            funRef = f
        else:
            if i == 1:
                old_fvals = np.tile(-np.inf, (options.memory, 1))

            if i <= options.memory:
                old_fvals[i - 1] = f
            else:
                old_fvals = np.vstack([old_fvals[1:], f])

            funRef = np.max(old_fvals)

        x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals += 1
        lineSearchIters = 1
        while f_new > funRef + options.suffDec * np.dot(g.T, (x_new - x)) or not isLegal(f_new):
            temp = t
            # Halfing step size
            if options.interp == 0 or ~isLegal(f_new):
                log(options, 3, 'Halving Step Size')
                t /= 2.0
            # Check whether step has become too small
            if np.max(np.absolute(t * d)) < options.progTol or t == 0:
                log(options, 3, 'Line Search failed')
                t = 0.0
                f_new = f
                g_new = g
                break

            x_new = x + t * d

            f_new, g_new = funObj(x_new)
            funEvals += 1
            lineSearchIters += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if options.testOpt:
            optCond = np.max(np.absolute(funProj(x - g) - x))
            projects += 1

        # Output Log
        if options.verbose >= 2:
            if options.testOpt:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier,
                                                                               projects, t, f, optCond))
            else:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier, projects, t, f))

        # Check optimality
        if options.testOpt:
            if optCond < options.optTol:
                log(options, 1, 'First-Order Optimality Conditions Below optTol')
                break

        if np.max(np.absolute(t * d)) < options.progTol:
            log(options, 1, 'Step size below progTol')
            break

        if np.absolute(f - f_old) < options.progTol:
            log(options, 1, 'Function value changing by less than progTol')
            break

        if funEvals * funEvalMultiplier > options.maxIter:
            log(options, 1, 'Function Evaluations exceeds maxIter')
            break

        i += 1

    return x, f


def isLegal(v):
    no_complex = v.imag.any().sum() == 0
    no_nan = np.isnan(v).sum() == 0
    no_inf = np.isinf(v).sum() == 0
    return no_complex and no_nan and no_inf


def auto_grad(x, funObj, options):
    # notice the funObj should return a single value here - the objective (i.e., no gradient)
    p = len(x)
    f = funObj(x)
    if type(f) == type(()):
        f = f[0]

    mu = 2*np.sqrt(1e-12)*(1+np.linalg.norm(x))/np.linalg.norm(p)
    diff = np.zeros((p,))
    for j in xrange(p):
        e_j = np.zeros((p,))
        e_j[j] = 1
        # this is somewhat wrong, since we also need to project,
        # but practically (and locally) it doesn't seem to matter.
        v = funObj(x + mu*e_j)
        if type(v) == type(()):
            diff[j] = v[0]
        else:
            diff[j] = v

    g = (diff-f)/mu

    return f, g


# Computes projection of x onto constraints LB <= x <= UB
def project_bound(x, lb, ub=None):
    x[x < lb] = lb[x < lb]
    if ub is not None:
        x[x > ub] = ub[x > ub]
    return x
