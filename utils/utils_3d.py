import numpy as np
import torch

quat_diff = lambda q1, q2: 2 * np.arccos(np.abs(np.dot(q1, q2)))
quat_diff_batch = lambda q1, q2: 2 * torch.acos(torch.abs(torch.bmm(q1.unsqueeze(1), q2.unsqueeze(-1)).squeeze(-1)))

def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

def find_rigid_alignment_batch(A, B):
    # A, B: B x N x 3
    a_mean = A.mean(dim=1) # B x 3
    b_mean = B.mean(dim=1) # B x 3
    A_c = A - a_mean.unsqueeze(1)
    B_c = B - b_mean.unsqueeze(1)
    # Covariance matrix
    H = torch.matmul(A_c.transpose(-1, -2), B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.matmul(V, U.transpose(-1, -2))
    # Translation vector
    t = b_mean.unsqueeze(1) - torch.matmul(R, a_mean.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)
    t = t.transpose(-1, -2).squeeze()
    return R, t
def get_execution_dir(hand_transf, direction=[0.0, 0.0, 1.0]):
    dir = torch.tensor(direction, dtype=torch.float32, device='cuda').unsqueeze(0).tile([len(hand_transf), 1])
    dir = torch.matmul(hand_transf[:, :3, :3], dir.unsqueeze(-1)).squeeze(-1)
    return dir