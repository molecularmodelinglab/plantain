import torch
import torch.nn as nn

def rotvec_to_rotmat(rotvec) -> torch.Tensor:
    """ Simplified rotvec to rotmat code from RoMa
    (https://github.com/naver/roma/blob/06e4b0cdc1c802a60a012bb19c581d6600c63358/roma/mappings.py#L371)
    """
    theta = torch.norm(rotvec, dim=-1)
    axis = rotvec / theta[...,None]
    kx, ky, kz = axis[:,0], axis[:,1], axis[:,2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs = kx*sin_theta
    ys = ky*sin_theta
    zs = kz*sin_theta
    xyc = kx*ky*one_minus_cos_theta
    xzc = kx*kz*one_minus_cos_theta
    yzc = ky*kz*one_minus_cos_theta
    xxc = kx**2*one_minus_cos_theta
    yyc = ky**2*one_minus_cos_theta
    zzc = kz**2*one_minus_cos_theta
    R_rodrigues = torch.stack([1 - yyc - zzc, xyc - zs, xzc + ys,
                     xyc + zs, 1 - xxc - zzc, -xs + yzc,
                     xzc - ys, xs + yzc, 1 -xxc - yyc], dim=-1).reshape(-1, 3, 3)
    R = R_rodrigues
    return R

@torch.compile(dynamic=True)
def f(coord, rot, trans):
    rot_mat = rotvec_to_rotmat(rot)
    coord = torch.einsum('...ij,...bj->...bi', rot_mat, coord) + trans
    return coord.sum()

coord = torch.ones((2,3))
rot = nn.Parameter(torch.ones((2,3)))
trans = nn.Parameter(torch.ones((2, 3)))

U = f(coord, rot, trans)
U.backward()