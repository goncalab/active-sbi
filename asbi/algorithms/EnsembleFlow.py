import torch as th 


class EnsembleFlow:
    def __init__(self, flows) -> None:
        self.n_flows = len(flows)
        self.flows = flows

    def log_prob(self, x, condition):
        with th.no_grad():
            log_probs = [flow.log_prob(x, condition.unsqueeze(0)) for flow in self.flows]
            stacked = th.stack(log_probs, dim=0).mean(dim=0)
            return stacked

    def sample(self, n_samples, condition):
        # generate samples from mixture of flows 
        with th.no_grad():
            n = n_samples // self.n_flows        
            samples = []
            for flow in self.flows:
                samples.append(flow.sample((n,), condition.unsqueeze(0)))

            if n_samples % self.n_flows != 0:
                remaining_samples = int(n_samples % self.n_flows)
                samples.append(flow.sample((remaining_samples,), condition.unsqueeze(0)))
                
            return th.cat(samples, dim=0)
    
    def compute_marginal_entropy(self, theta, N=1000):
        # compute entropy of the marginal predictive
        samples = self.sample(N, theta)
        return - th.mean(self.log_prob(samples, theta))
    
    def compute_ensemble_entropy(self, theta, N=1000):
        # compute average entropy of the ensemble
        with th.no_grad():
            entropies = []
            for flow in self.flows:
                samples = flow.sample((N,), theta.unsqueeze(0))
                entropies.append(-th.mean(flow.log_prob(samples, theta.unsqueeze(0))))
            return th.mean(th.stack(entropies, dim=0))

    def compute_bald_score(self, theta, N=1000):
        # entropy of marginal predictive + average entropy of the ensemble
        marginal_entropy = self.compute_marginal_entropy(theta, N)
        ensemble_entropy = self.compute_ensemble_entropy(theta, N)
        bald_score = ensemble_entropy - marginal_entropy
        return bald_score