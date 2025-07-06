import copy
import torch

def FedAvg(w, args):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            tmp += w[i][k]
        tmp = torch.true_divide(tmp, len(w))
        w_avg[k].copy_(tmp)
    return w_avg


def FedAdam(w_locals, args, prev_moments, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # FedAdam-style aggregation
    w_avg = copy.deepcopy(w_locals[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])

    for w in w_locals:
        for key in w_avg.keys():
            w_avg[key] += w[key] / len(w_locals)

    # Adam moments
    if 'm' not in prev_moments:
        prev_moments['m'] = {k: torch.zeros_like(v) for k, v in w_avg.items()}
        prev_moments['v'] = {k: torch.zeros_like(v) for k, v in w_avg.items()}
    
    for key in w_avg.keys():
        g = w_avg[key] - prev_moments['w_glob_prev'][key]
        prev_moments['m'][key] = beta1 * prev_moments['m'][key] + (1 - beta1) * g
        prev_moments['v'][key] = beta2 * prev_moments['v'][key] + (1 - beta2) * g * g
        m_hat = prev_moments['m'][key] / (1 - beta1)
        v_hat = prev_moments['v'][key] / (1 - beta2)
        prev_moments['w_glob_prev'][key] = prev_moments['w_glob_prev'][key] - args.lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    
    return prev_moments['w_glob_prev'], prev_moments


def FedProx(w_locals, w_glob_prev, args):
    # FedProx utilise la même agrégation que FedAvg
    return FedAvg(w_locals, args)

def SGD_local(w_locals, args):
    # Pas d'agrégation globale : retour des poids locaux (par exemple du premier client)
    return w_locals[0]