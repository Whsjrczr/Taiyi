import torch
import torch.linalg as linalg


def cal_cov_matrix(input):
    if input is None:
        return torch.zeros((1, 1))

    if not torch.is_tensor(input):
        input = torch.as_tensor(input)

    if input.dim() == 0:
        input = input.reshape(1, 1)
    elif input.dim() == 1:
        input = input.reshape(-1, 1)
    elif input.dim() == 3:
        input = input.transpose(0, 2).contiguous().view(input.shape[2], -1).T
    elif input.dim() > 3:
        input = input.reshape(input.shape[0], -1)

    if input.dim() != 2:
        input = input.reshape(input.shape[0], -1)

    # Treat rows as samples and cols as features.
    num_samples, num_features = input.shape
    if num_features == 0:
        return torch.zeros((1, 1), dtype=input.dtype, device=input.device)
    if num_samples <= 1:
        return torch.zeros((num_features, num_features), dtype=input.dtype, device=input.device)
    if num_features == 1:
        var = input[:, 0].float().var(unbiased=True)
        return var.reshape(1, 1).to(dtype=input.dtype)

    cov = torch.cov(input.T)
    if cov.dim() == 0:
        cov = cov.reshape(1, 1)
    elif cov.dim() == 1:
        cov = torch.diag(cov)
    return cov


def cal_eig(input):
    # eigvals = linalg.eigvalsh(input.float())
    # print(input.float())
    if input is None:
        return torch.ones(1)
    if not torch.is_tensor(input):
        input = torch.as_tensor(input)
    if input.dim() == 0:
        return input.reshape(1).float().abs()
    if input.dim() == 1:
        return input.float().abs()
    try:
        _, eigvals, _ = linalg.svd(input.float())
    except Exception:
        lens = min(input.shape) if len(input.shape) > 0 else 1
        eigvals = torch.ones(lens, dtype=torch.float32, device=input.device if torch.is_tensor(input) else None)
    return eigvals

def cal_eig_not_sym(input):
    if input is None:
        return torch.ones(1)
    if not torch.is_tensor(input):
        input = torch.as_tensor(input)
    if input.dim() == 0:
        return input.reshape(1).float().abs()
    if input.dim() == 1:
        return input.float().abs()
    try:
        _, eigvals, _ = linalg.svd(input.float())
    except Exception:
        lens = min(input.shape) if len(input.shape) > 0 else 1
        eigvals = torch.ones(lens, dtype=torch.float32, device=input.device if torch.is_tensor(input) else None)
    return eigvals


if __name__ == '__main__':
    # test cal_cov_matrix
    x = torch.randn((5, 10, 6))
    print(x.T.shape)
    y = cal_cov_matrix(x)
    print(y.shape)

    # test cal_eig
    # x = torch.randn((10, 10))
    # y = cal_eig(x)
    # print(y)
    # print(y.size())
