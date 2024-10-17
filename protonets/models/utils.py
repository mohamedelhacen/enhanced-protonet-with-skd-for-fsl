import torch
import torch.nn as nn
from torch.autograd import Variable

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    eucl_dst = torch.pow(x - y, 2).sum(2)

    return eucl_dst


def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    cos = torch.nn.CosineSimilarity()
    return cos(x, y)


def kl_divergence(zs_m, zs_s, zq_m, zq_s):

    outputs = []
    for i in range(len(zq_m)):
        queries = []
        zq_distribution = torch.distributions.normal.Normal(zq_m[i], zq_s[i])
        for j in range(len(zs_m)):
            
            # kl = torch.log(torch.abs(zs_s[j]) / torch.abs(zq_s[i])) + 0.5 * (torch.pow(zq_s[i], 2) + torch.pow(zs_m[j] - zq_m[i], 2))/torch.pow(zs_s[j], 2) - 0.5
            zs_distribution = torch.distributions.normal.Normal(zs_m[j], zs_s[j])
            kl = torch.distributions.kl.kl_divergence(zs_distribution, zq_distribution)
            queries.append(kl.item())
        
        outputs.append(queries)
    
    return torch.tensor(outputs, requires_grad=True).cuda()


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
        
        
def js_divergence(zs, zq):
    outputs = []
    for i in range(len(zs)):
        zs_mean, zs_std = zs[i].mean().cuda(), zs[i].std().cuda()
        
        zs_distr = torch.distributions.normal.Normal(zs_mean, zs_std)
        queries = []
        
        for j in range(len(zq)):
            zq_mean, zq_std = zq[j].mean().cuda(), zq[j].std().cuda()
            zq_distr = torch.distributions.normal.Normal(zq_mean, zq_std)

            M = 0.5 * (zs[i] + zq[j])

            M_distr = torch.distributions.normal.Normal(M.mean(), M.std())

            js = 0.5 * (torch.distributions.kl.kl_divergence(zs_distr, M_distr) + torch.distributions.kl.kl_divergence(zq_distr, M_distr)).cuda()

            queries.append(js.item())
        
        outputs.append(queries)


    return Variable(torch.tensor(outputs), requires_grad=True).cuda()

def js_div(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    

def tensor_to_distribution(tensor):
  # Calculate the sum of the values in the tensor
  tensor_sum = torch.sum(tensor, dim=-1, keepdim=True)
  
  # Divide each value in the tensor by the sum to obtain a probability
  distribution = tensor / tensor_sum
  
  return distribution

def jensen_shannon_divergence(x, y):


    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    A = x
    B = y

    A /= A.sum(dim=1, keepdim=True)
    B /= B.sum(dim=1, keepdim=True)

    A = A.unsqueeze(1).expand(n, m, d)  # Shape: (3, 1, 3)
    B = B.unsqueeze(0).expand(n, m, d)  # Shape: (1, 5, 3)

    avg_prob = (A + B) / 2.0
    epsilon = 1e-10
    kl_div_A = torch.sum(A * torch.log((A + epsilon) / (avg_prob + epsilon)), dim=2)
    kl_div_B = torch.sum(B * torch.log((B + epsilon) / (avg_prob + epsilon)), dim=2)

    jsd = 0.5 * (kl_div_A + kl_div_B)

    # Calculate Jensen-Shannon distance (optional)
    js_distance = torch.sqrt(jsd)

    return js_distance
