import torch


def standardize(tensor):
    # Vectorize each example
    tensor = tensor.reshape(tensor.shape[0], -1)
    sd = torch.sqrt(torch.sum(tensor * tensor, dim=1)).reshape(-1, 1)
    tensor = tensor / (sd + 0.001)
    return tensor


def simclr_loss_func(embedding1, embedding2, lam=0.5):
    assert embedding1.shape == embedding2.shape
    batch_size = embedding1.shape[0]

    # Standardize 64 dim outputs of original and deformed images
    embedding1_stand = standardize(embedding1)
    embedding2_stand = standardize(embedding2)
    # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
    cov12 = torch.mm(embedding1_stand, embedding2_stand.transpose(0, 1))  # COV
    cov11 = torch.mm(embedding1_stand, embedding1_stand.transpose(0, 1))  # COV0
    cov22 = torch.mm(embedding2_stand, embedding2_stand.transpose(0, 1))  # COV1
    # Diagonals of covariances.
    d12 = torch.diag(cov12)  # v
    d11 = torch.diag(cov11)  # v0
    d22 = torch.diag(cov22)  # v1
    # Mulitnomial logistic loss just computed on positive match examples, with all other examples as a separate class.
    lecov = torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov11 - torch.diag(d11), dim=1)))
    lecov += torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov22 - torch.diag(d22), dim=1)))
    lecov = lam * lecov - d12

    loss = torch.mean(lecov)

    '''
    # Accuracy
      if torch.cuda.is_available():
        ID = 2. * torch.eye(batch_size).to('cuda') - 1.
      else:
        ID = 2. *torch.eye(batch_size) - 1
      icov=ID*cov12
      acc=torch.sum((icov>0).type(torch.float))/ batch_size
      '''

    return loss
