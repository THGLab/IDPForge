"""Single-property maximum log-likelihood scoring (X-EISD)."""
import numpy as np
import pandas as pd


def calc_opt_params(beta, exp, exp_sig, sig):
    opt_params = np.zeros(beta.shape)
    exp_sig = np.asarray(exp_sig, dtype=float)
    valid = exp_sig > 0
    if np.any(valid):
        sig_sq = np.where(valid, sig ** 2.0, 0.0)
        esig_sq = np.where(valid, exp_sig ** 2.0, 1.0)
        ratio = sig_sq / esig_sq
        opt_params = np.where(valid, (ratio * (exp - beta)) / (1.0 + ratio), 0.0)
    return opt_params


def normal_loglike(x, mu, sig, gamma=1.0):
    x = np.asarray(x, dtype=float)
    sig = np.asarray(sig, dtype=float)
    logp = np.zeros(x.shape)
    if sig.ndim == 0:
        if sig > 0:
            exp_val = -gamma * ((x - mu) ** 2.0) / (2.0 * (sig ** 2.0))
            logp = np.log(1.0 / np.sqrt(2.0 * np.pi * (sig ** 2.0))) + exp_val
    else:
        valid = sig > 0
        if np.any(valid):
            mu_arr = np.broadcast_to(np.asarray(mu, dtype=float), x.shape)
            exp_val = -gamma * ((x[valid] - mu_arr[valid]) ** 2.0) / (2.0 * (sig[valid] ** 2.0))
            logp[valid] = np.log(1.0 / np.sqrt(2.0 * np.pi * (sig[valid] ** 2.0))) + exp_val
    return logp


def calc_score(beta, exp, exp_sig, sig, opt_params, gamma=1.0):
    f_q = normal_loglike(opt_params, 0, sig, gamma)
    f_err = normal_loglike(exp - opt_params - beta, 0, exp_sig, gamma)
    return f_q + f_err


def cs_score_ensemble(exp_data, bc_data, indices):
    exp_data = pd.read_csv(exp_data)
    exp_cs = exp_data['value'].values
    exp_sigma = exp_data['error'].values
    atom_types = exp_data['atomname'].values
    bc_cs = np.mean(bc_data.data[indices], axis=0)
    bc_sigma = np.array([bc_data.sigma[a] for a in atom_types])
    opt_params = calc_opt_params(bc_cs, exp_cs, exp_sigma, bc_sigma)
    f = calc_score(bc_cs, exp_cs, exp_sigma, bc_sigma, opt_params)
    error = (exp_cs - bc_cs) ** 2.0
    mae = np.mean(error ** 0.5)
    return mae, np.sum(f), error


def vect_calc_opt_params_jc(alpha1, alpha2, exp_j, exp_sig, mus, sigs):
    exp_sig = np.asarray(exp_sig, dtype=float)
    valid = exp_sig > 0
    safe = np.where(valid, exp_sig, 1.0)
    a = np.zeros((alpha1.shape[0], 3, 3))
    b = np.zeros((alpha1.shape[0], 3))
    a[:, 0, 0] = 1.0 / (sigs[0] ** 2.0) + ((alpha2 / safe) ** 2.0)
    a[:, 1, 1] = 1.0 / (sigs[1] ** 2.0) + ((alpha1 / safe) ** 2.0)
    a[:, 2, 2] = 1.0 / (sigs[2] ** 2.0) + 1.0 / (safe ** 2.0)
    a[:, 0, 1] = a[:, 1, 0] = alpha1 * alpha2 / (safe ** 2.0)
    a[:, 0, 2] = a[:, 2, 0] = alpha2 / (safe ** 2.0)
    a[:, 1, 2] = a[:, 2, 1] = alpha1 / (safe ** 2.0)
    b[:, 0] = mus[0] / (sigs[0] ** 2.0) + exp_j * alpha2 / (safe ** 2)
    b[:, 1] = mus[1] / (sigs[1] ** 2.0) + exp_j * alpha1 / (safe ** 2)
    b[:, 2] = mus[2] / (sigs[2] ** 2.0) + exp_j / (safe ** 2)
    opt_params = np.array([np.linalg.solve(a[i], b[i]) for i in range(a.shape[0])])
    if not np.all(valid):
        opt_params[~valid] = np.array(mus, dtype=float)
    return opt_params


def vect_calc_score_JC(alpha1, alpha2, exp_j, exp_sig, opt_params, mus, sigs):
    f_a = normal_loglike(opt_params[:, 0], mus[0], sigs[0])
    f_b = normal_loglike(opt_params[:, 1], mus[1], sigs[1])
    f_c = normal_loglike(opt_params[:, 2], mus[2], sigs[2])
    err = exp_j - opt_params[:, 0] * alpha2 - opt_params[:, 1] * alpha1 - opt_params[:, 2]
    f = f_a + f_b + f_c + normal_loglike(err, 0, exp_sig)
    invalid = np.asarray(exp_sig, dtype=float) <= 0
    if np.any(invalid):
        f[invalid] = 0.0
    return f


def jc_score_ensemble(exp_data, bc_data, indices):
    exp_data = pd.read_csv(exp_data)
    exp = exp_data['value'].values
    exp_sigma = exp_data['error'].values
    bc_alpha1 = np.mean(bc_data.data[indices], axis=0)
    bc_alpha2 = np.mean(np.square(bc_data.data[indices]), axis=0)
    bc_sigma = [bc_data.sigma[i] for i in ["A", "B", "C"]]
    bc_mu = [bc_data.mu[i] for i in ["A", "B", "C"]]
    opt_params = vect_calc_opt_params_jc(bc_alpha1, bc_alpha2, exp, exp_sigma, bc_mu, bc_sigma)
    f = vect_calc_score_JC(bc_alpha1, bc_alpha2, exp, exp_sigma, opt_params, bc_mu, bc_sigma)
    error = (opt_params[:, 0] * bc_alpha2 + opt_params[:, 1] * bc_alpha1 + opt_params[:, 2] - exp) ** 2.0
    mae = np.mean(error ** 0.5)
    return mae, np.sum(f)


def dist_score_ensemble(exp_data, bc_data, indices):
    exp_data = pd.read_csv(exp_data)
    exp_distance = exp_data['dist_value'].values
    upper_bound_value = exp_data['upper'].values
    lower_bound_value = exp_data['lower'].values
    exp_sigma = (upper_bound_value + lower_bound_value) / 2.0
    bc_ensemble = np.power(bc_data.data[indices], -6.0)
    avg_distance = np.power(np.mean(bc_ensemble, axis=0), (-1. / 6.))
    opt_params = calc_opt_params(avg_distance, exp_distance, exp_sigma, bc_data.sigma)
    f = calc_score(avg_distance, exp_distance, exp_sigma, bc_data.sigma, opt_params)
    trial_error = np.stack([exp_distance - avg_distance,
                            exp_distance + upper_bound_value - avg_distance,
                            exp_distance - lower_bound_value - avg_distance], axis=0)
    error = (np.min(abs(trial_error), axis=0)) ** 2
    mae = np.mean(error ** 0.5)
    return mae, np.sum(f)


def generic_score_ensemble(exp_data, bc_data, indices):
    exp_data = pd.read_csv(exp_data)
    exp = exp_data['value'].values
    exp_sigma = exp_data['error'].values
    bc = np.mean(bc_data.data[indices], axis=0)
    opt_params = calc_opt_params(bc, exp, exp_sigma, bc_data.sigma)
    f = calc_score(bc, exp, exp_sigma, bc_data.sigma, opt_params)
    mae = np.mean(((exp - bc) ** 2.0) ** 0.5)
    return mae, np.sum(f)


ENSEMBLE_Scorers = {
    "jc": jc_score_ensemble, "noe": dist_score_ensemble, "pre": dist_score_ensemble,
    "cs": cs_score_ensemble, "fret": generic_score_ensemble,
}
