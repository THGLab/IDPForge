"""X-EISD API: score and/or optimize the log-likelihood of a disordered ensemble."""
import numpy as np
import pandas as pd
from copy import deepcopy
from .scorer import ENSEMBLE_Scorers


def monte_carlo(beta, old_total_score, new_total_score):
    new_probability = np.exp(beta * new_total_score)
    old_probability = np.exp(beta * old_total_score)
    if not (np.isfinite(old_probability) and np.isfinite(new_probability)):
        # stable log-domain Metropolis when probabilities overflow
        log_ratio = beta * (new_total_score - old_total_score)
        if log_ratio >= 0:
            return True
        return np.random.random_sample() < np.exp(log_ratio)
    return np.random.random_sample() < min(1, new_probability / old_probability)


class XEISD:
    def __init__(self, exp_data, bc_data, pool_size=None, verbose=False, exclude=None, seed=42):
        np.random.seed(seed)
        self.exp_data = exp_data
        self.bc_data = bc_data
        self.verbose = verbose
        self.pool_size = pool_size
        self.visited = {} if exclude is None else exclude
        if pool_size is None:
            self.pool_size = bc_data[list(bc_data.keys())[0]].data.shape[0]

    def _check_param_consistent(self, data_types):
        for confs, scores in self.visited.items():
            new_scores = self.calc_scores(data_types, list(confs))
            new_total_score = np.sum([new_scores[key][1] for key in data_types])
            if new_total_score == scores:
                break
            self.visited[confs] = new_total_score

    def _calc_per_conf_stats(self):
        score_record = {}
        for combo, s in self.visited.items():
            for conf in combo:
                score_record.setdefault(conf, []).append(s)
        return {conf: [np.mean(s), np.std(s)] for conf, s in score_record.items()}

    def calc_scores(self, dtypes, indices):
        scores = {}
        for prop in dtypes:
            if prop == 'cs':
                cs_stats = list(ENSEMBLE_Scorers['cs'](self.exp_data['cs'], self.bc_data['cs'], indices))
                scores['cs'] = cs_stats[:2]
                atom_types = pd.read_csv(self.exp_data['cs'])['atomname'].values
                cs_sigmas = self.bc_data['cs'].sigma  # per-atom back-calc sigma
                sq_err = cs_stats[-1]  # (exp - bc)^2 per shift
                cs_maes, cs_chi2s = {}, {}
                for a in np.unique(atom_types):
                    mask = atom_types == a
                    cs_maes[a] = np.mean(sq_err[mask] ** 0.5)
                    cs_chi2s[a] = np.mean(sq_err[mask] / cs_sigmas[a] ** 2.0)
                scores['cs_per_atom_mae'] = cs_maes
                scores['cs_per_atom_chi2'] = cs_chi2s
                all_sig = np.array([cs_sigmas[a] for a in atom_types])
                scores['cs_chi2'] = np.mean(sq_err / all_sig ** 2.0)  # SI Eqn 12
            else:
                scores[prop] = list(ENSEMBLE_Scorers[prop](self.exp_data[prop], self.bc_data[prop], indices))
        return scores

    def optimize(self, opt_props, opt_type='max', ens_size=100, beta=0.1, indices=None, iters=None):
        self._check_param_consistent(opt_props)
        if iters is None:
            iters = self.pool_size * 5
        if indices is None:
            indices = list(np.random.choice(np.arange(self.pool_size), ens_size, replace=False))
        old_scores = self.calc_scores(opt_props, indices)
        best_total_score = np.sum([old_scores[key][1] for key in opt_props])
        best_indices, best_scores = indices, old_scores
        for _ in range(iters):
            struct_found = False
            while not struct_found:
                pop_index = np.random.randint(ens_size)
                popped_structure = indices[pop_index]
                new_index = np.random.randint(self.pool_size)
                if new_index != popped_structure and new_index not in indices:
                    indices[pop_index] = new_index
                    if tuple(np.sort(indices)) not in self.visited:
                        struct_found = True
                    else:
                        indices[pop_index] = popped_structure
            new_scores = self.calc_scores(opt_props, indices)
            old_total_score = np.sum([old_scores[key][1] for key in opt_props])
            new_total_score = np.sum([new_scores[key][1] for key in opt_props])
            self.visited[tuple(np.sort(indices))] = new_total_score
            if new_total_score > best_total_score:
                best_indices = [i for i in indices]
                best_scores = deepcopy(new_scores)
            if opt_type == 'max':
                to_accept = old_total_score < new_total_score
            elif opt_type == 'mc':
                to_accept = monte_carlo(beta, old_total_score, new_total_score)
            else:
                print('Opt type not supported...Abort.')
                return
            if not to_accept:
                indices[pop_index] = popped_structure
            else:
                old_scores = deepcopy(new_scores)
        return best_indices, best_scores, self._calc_per_conf_stats()
