# De-Wei Ye, YCC lab, 2018
# modified from CMRT program (in C) by Tseng Wei-Hsiang, YCC lab, 2015
from lib import *


def modified_Redfield_theory_calculation(hamiltonian, temperature, reorganization_energy, cutoff_freq):
    beta = get_beta_from_T(temperature)
    print('utilize modified Redfield theory to obtain rate constant matrix')
    print('bath: Over-damped Brownian Oscillators')
    size = hamiltonian.shape[0]
    rate = np.zeros((size, size))
    cm_to_fs = 5309.1
    w, v = np.linalg.eigh(hamiltonian)
    for a in range(size):
        for b in range(size):
            params = [
                beta, reorganization_energy, cutoff_freq,
                w[a], w[b],
                transCoeff(v, b, a, a, b),
                transCoeff(v, a, a, b, a),
                transCoeff(v, b, b, b, a),
                transCoeff(v, a, a, a, b),
                transCoeff(v, b, b, a, b),
                transCoeff(v, b, b, a, a),
                transCoeff(v, a, a, a, a),
                transCoeff(v, b, b, b, b),
            ]
            result, *_ = integrate.quad(mrt_kernal, 0, 0.2, args=params)
            rate[a, b] = result * 2 / cm_to_fs * 1000  # to ps

    # modified rate constant matrix:
    # ignore the negative terms
    rate[rate <= 0] = 0

    # diagonal term = - sum(offdiagonal) for normalization in column
    # (go out from some state)
    rate[np.diag_indices_from(rate)] = 0
    rate -= np.diag(np.sum(rate, axis=0))

    print('Rate constants:')
    print(rate)
    return rate


def transCoeff(_evec, _a, _b, _c, _d):
    _r = 0
    for i in range(_evec.shape[0]):
        _r += _evec[i, _a] * _evec[i, _b] * _evec[i, _c] * _evec[i, _d]
    return _r


def mrt_kernal(_tau, _params):
    _beta, _lambda, _gamma, _Ea, _Eb, _Cbaab, _Caaba, _Cbbba, _Caaab, _Cbbab, _Cbbaa, _Caaaa, _Cbbbb = _params
    _G, _H, _C = [biexp(_tau, _beta, _lambda, _gamma, option=_i) for _i in range(3)]
    _Re = _Cbaab * _C.real - (
            (_Caaba - _Cbbba) * (_Caaab - _Cbbab) * _H.real ** 2
            - (
                    ((_Caaba - _Cbbba) * _H.imag - 2 * _Cbbba * _lambda)
                    * ((_Caaab - _Cbbab) * _H.imag - 2 * _Cbbab * _lambda)
            )
    )
    _Im = _Cbaab * _C.imag - (
            (_Caaba - _Cbbba) * _H.real
            * ((_Caaab - _Cbbab) * _H.imag - 2 * _Cbbab * _lambda)
            + ((_Caaba - _Cbbba) * _H.imag - 2 * _Cbbba * _lambda)
            * (_Caaab - _Cbbab) * _H.real
    )
    _expx = np.exp((2 * _Cbbaa - _Caaaa - _Cbbbb) * _G.real)
    _theta = (2 * _Cbbaa - _Caaaa - _Cbbbb) * _G.imag + (2 * (_Cbbaa - _Cbbbb) * _lambda + _Eb - _Ea) * _tau
    _result = _expx * (_Re * np.cos(_theta) - _Im * np.sin(_theta))
    return _result


def G_biexp_aux(_tau, _c, _r):
    return _c * (np.exp(-_r * _tau) - 1) / _r ** 2 + _c * _tau / _r


def H_biexp_aux(_tau, _c, _r):
    return _c * (1 - np.exp(-_r * _tau)) / _r


def C_biexp_aux(_tau, _c, _r):
    return _c * np.exp(-_r * _tau)


def biexp(_tau, _beta, _lambda, _r1, option=0):
    """
    option 0: G
    option 1: H
    option 2: C
    """
    _r2 = 42 ** .5 / _beta
    _delta = _beta * _lambda * _r1 / 20
    _c1 = complex(
        2 * _lambda / _beta * (1 - 10 * (_delta / _lambda) ** 2 - 2.45 / ((_r2 / _r1) ** 2 - 1)),
        -_lambda * _r1
    )
    _c2 = 98 * _r2 * _delta / _beta ** 2 / (_r2 ** 2 - _r1 ** 2)
    if option == 0:
        return G_biexp_aux(_tau, _c1, _r1) + G_biexp_aux(_tau, _c2, _r2) + _delta * _tau
    if option == 1:
        return H_biexp_aux(_tau, _c1, _r1) + H_biexp_aux(_tau, _c2, _r2) + _delta
    if option == 2:
        return C_biexp_aux(_tau, _c1, _r1) + C_biexp_aux(_tau, _c2, _r2)
