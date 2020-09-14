import torch


def detNeuralSort(s, tau=1.0, k=2):
    su = s.unsqueeze(-1).float()
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32).cuda()
    A_s = torch.abs(su - su.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.ones(1, k).cuda()))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).float().cuda()
    C = (su * scaling.unsqueeze(0))[:, :, :k]
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def MSELoss(y, y_hat):
    return ((y - y_hat)**2).sum(-1)


def MultinomialLoss(y, y_hat):
    return -(torch.log_softmax(y_hat, -1) * y).sum(-1)


def VAELoss(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = MultinomialLoss(x, recon_x)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD

def neuPrec(p_hat, y, tau=1.0, k=1, use_top=True):
    ysum = y.sum(1)
    loss = -(((y * p_hat.sum(1)).sum(1) / k) * torch.clamp(ysum, 0, 1))
    return loss

def neuPrecLoss(sc, scores, row, topk=100, k=5, tau=10.0, use_top=True):
    #m, _ = scores.min(1)
    #M, _ = scores.max(1)
    #scores = ((scores.T - m)/ (M-m)).T

    if use_top:
        y_hat, indices = torch.topk(scores, topk)
        y = row.gather(1, indices)
    else:
        indices = torch.randint(0, row.shape[1], size=(row.shape[0], topk)).cuda()
        y_hat = scores.gather(1, indices)
        y = row.gather(1, indices)
    ysum = y.sum(1)
    yy = sc(y_hat, k=k)
    loss = -(((y * yy.sum(1)).sum(1) / k) * torch.clamp(ysum, 0, 1))
    #loss = -((y * (yy.sum(1))).sum(1) * torch.clamp(ysum, 0, 1))
    #loss = -(((y - 0.0) * (yy.sum(1))).sum(1) * torch.clamp(ysum, 0, 1))
    return loss

def neuMapLoss(sc, scores, row, topk=100, k=5, tau=10.0, use_top=True):
    if use_top:
        y_hat, indices = torch.topk(scores, topk)
        y = row.gather(1, indices)
    else:
        indices = torch.randint(0, row.shape[1], size=(row.shape[0], topk)).cuda()
        y_hat = scores.gather(1, indices)
        y = row.gather(1, indices)

    P = sc(y_hat, k)
    pjTyToi = torch.zeros(scores.shape[0]).cuda()
    neuMap = torch.zeros(scores.shape[0]).cuda()
    for i in range(k):
        q = P[:, i, :]
        pjTy = (y * q).sum(1)
        pjTyToi = pjTyToi + pjTy
        neuMap = neuMap + (pjTy * pjTyToi) / (i + 1)
    neuMap = neuMap
    return -neuMap.mean()

def neuNDCGLoss(sc, scores, row, topk=100, k=5, tau=10.0, use_top=True):
    diag = torch.diag(1 / (1 + torch.log(torch.arange(1, k + 1).float()))).cuda()
    if use_top:
        y_hat, indices = torch.topk(scores, topk)
        y = row.gather(1, indices)
    else:
        indices = torch.randint(0, row.shape[1], size=(row.shape[0], topk)).cuda()
        y_hat = scores.gather(1, indices)
        y = row.gather(1, indices)
    ysum = y.sum(1)
    yy = sc(y_hat, k=k)
    loss3 = -((y * torch.matmul(diag.cuda(), yy).sum(1)).sum(1) * torch.clamp(ysum, 0, 1) * (1 / (1e-10 + y.sum(1))))
    return loss3
