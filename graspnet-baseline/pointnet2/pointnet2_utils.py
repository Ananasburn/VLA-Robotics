# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

_ext = None
if torch.cuda.is_available():
    try:
        import pointnet2._ext as _ext
    except ImportError:
        pass

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

# =============================================================================
# CPU Implementations
# =============================================================================

def furthest_point_sample_cpu(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def gather_points_cpu(features, idx):
    """
    Input:
        features: [B, C, N]
        idx: [B, npoint] matrix of the indices
    Output:
        new_features: [B, C, npoint]
    """
    # features: (B, C, N) -> (B, N, C) for easier indexing
    B, C, N = features.shape
    npoint = idx.shape[1]
    
    # We want features[b, :, idx[b, i]]
    # Expand indices
    features_trans = features.permute(0, 2, 1) # (B, N, C)
    
    # Helper to gather
    batch_indices = torch.arange(B, dtype=torch.long, device=features.device).view(B, 1).expand(B, npoint)
    
    # Gather
    out = features_trans[batch_indices, idx, :] # (B, npoint, C)
    return out.permute(0, 2, 1) # (B, C, npoint)

def three_nn_cpu(unknown, known):
    """
    Input:
        unknown: (B, N, 3)
        known: (B, M, 3)
    Output:
        dist: (B, N, 3) l2 distance to the three nearest neighbors
        idx: (B, N, 3) index of 3 nearest neighbors
    """
    dist2 = torch.cdist(unknown, known, p=2)**2
    dist, idx = torch.topk(dist2, 3, dim=2, largest=False, sorted=True)
    return dist, idx

def three_interpolate_cpu(features, idx, weight):
    """
    Input:
        features: (B, c, m)
        idx: (B, n, 3)
        weight: (B, n, 3)
    Output:
        (B, c, n)
    """
    B, c, m = features.shape
    n = idx.shape[1]
    
    # Interpolate
    # features: (B, C, M)
    # idx: (B, N, 3) - indices into M
    # weight: (B, N, 3)
    
    interp_features = torch.zeros(B, c, n, device=features.device)
    
    for b in range(B):
        # features[b]: (c, m)
        # idx[b]: (n, 3)
        # weight[b]: (n, 3)
        
        # Gather features: (c, n, 3)
        f = features[b][:, idx[b]] # (c, n, 3)
        
        # Weighted sum
        # w: (1, n, 3)
        w = weight[b].unsqueeze(0)
        
        weighted = f * w # (c, n, 3)
        interp_features[b] = torch.sum(weighted, dim=2)
        
    return interp_features

def group_points_cpu(features, idx):
    """
    Input:
        features: (B, C, N)
        idx: (B, npoint, nsample)
    Output:
        (B, C, npoint, nsample)
    """
    B, C, N = features.shape
    npoint = idx.shape[1]
    nsample = idx.shape[2]
    
    features_trans = features.permute(0, 2, 1) # (B, N, C)
    
    # batch_indices: (B, npoint, nsample)
    batch_indices = torch.arange(B, dtype=torch.long, device=features.device).view(B, 1, 1).expand(B, npoint, nsample)
    
    out = features_trans[batch_indices, idx, :] # (B, npoint, nsample, C)
    return out.permute(0, 3, 1, 2) # (B, C, npoint, nsample)

def ball_query_cpu(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: float
        nsample: int
        xyz: (B, N, 3)
        new_xyz: (B, npoint, 3)
    """
    # sq_dist: [B, npoint, N]
    sq_dist = torch.cdist(new_xyz, xyz, p=2)**2
    
    vals, idx = torch.topk(sq_dist, nsample, dim=2, largest=False, sorted=True)
    
    # Mask out points outside radius
    # If a point is outside radius, usually we repeat the FIRST point
    
    valid = vals <= radius**2
    
    # Fill invalid with the first index
    first = idx[:, :, 0:1].expand(-1, -1, nsample)
    idx[~valid] = first[~valid]
    
    return idx

def cylinder_query_cpu(radius, hmin, hmax, nsample, xyz, new_xyz, rot):
    """
    xyz: (B, N, 3)
    new_xyz: (B, npoint, 3)
    rot: (B, npoint, 9)
    return: idx (B, npoint, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = new_xyz.shape[1]
    
    # Expand dims
    xyz_exp = xyz.unsqueeze(1) # (B, 1, N, 3)
    new_xyz_exp = new_xyz.unsqueeze(2) # (B, npoint, 1, 3)
    diff = xyz_exp - new_xyz_exp # (B, npoint, N, 3)
    
    # Rot
    rot_mat = rot.view(B, npoint, 3, 3)
    
    # local = diff @ rot_mat
    local_xyz = torch.matmul(diff, rot_mat) # (B, npoint, N, 3)
    
    x = local_xyz[..., 0]
    y = local_xyz[..., 1]
    z = local_xyz[..., 2]
    
    radial_dist2 = y**2 + z**2
    
    in_cylinder = (x >= hmin) & (x <= hmax) & (radial_dist2 < radius**2) # (B, npoint, N)
    
    dist_fake = radial_dist2.clone()
    dist_fake[~in_cylinder] = 1e10
    
    vals, idx = torch.topk(dist_fake, nsample, dim=2, largest=False, sorted=True)
    
    valid = vals < 1e9
    first = idx[:, :, 0:1].expand(-1, -1, nsample)
    idx[~valid] = first[~valid]
    
    return idx

# =============================================================================
# Modules
# =============================================================================

class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        if _ext is not None and xyz.is_cuda:
            return _ext.furthest_point_sampling(xyz, npoint)
        else:
            return furthest_point_sample_cpu(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        if _ext is not None and features.is_cuda:
            _, C, N = features.size()
            ctx.for_backwards = (idx, C, N)
            return _ext.gather_points(features, idx)
        else:
             return gather_points_cpu(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        if _ext is not None and grad_out.is_cuda:
            idx, C, N = ctx.for_backwards
            grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None
        else:
            # CPU backward not implemented fully, usually strictly inference
            return None, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        if _ext is not None and unknown.is_cuda:
            dist2, idx = _ext.three_nn(unknown, known)
            return torch.sqrt(dist2), idx
        else:
            dist2, idx = three_nn_cpu(unknown, known) # returns sqrt(dist2) if we implemented it?
            # Our cpu impl returned dist, idx (where dist is Euclidean). 
            # Original _ext.three_nn returns dist2 (squared).
            # And the wrapper returns torch.sqrt(dist2).
            # CPU impl `three_nn_cpu` in my code calls `cdist` (p=2) which is dist. 
            
            return dist2, idx # dist2 is already sqrt-ed due to cdist p=2 behavior

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        if _ext is not None and features.is_cuda:
            B, c, m = features.size()
            n = idx.size(1)
            ctx.three_interpolate_for_backward = (idx, weight, m)
            return _ext.three_interpolate(features, idx, weight)
        else:
            return three_interpolate_cpu(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        if _ext is not None and grad_out.is_cuda:
            idx, weight, m = ctx.three_interpolate_for_backward
            grad_features = _ext.three_interpolate_grad(
                grad_out.contiguous(), idx, weight, m
            )
            return grad_features, None, None
        else:
            return None, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        if _ext is not None and features.is_cuda:
            B, nfeatures, nsample = idx.size()
            _, C, N = features.size()
            ctx.for_backwards = (idx, N)
            return _ext.group_points(features, idx)
        else:
            return group_points_cpu(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        if _ext is not None and grad_out.is_cuda:
            idx, N = ctx.for_backwards
            grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None
        else:
            return None, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        if _ext is not None and xyz.is_cuda:
            return _ext.ball_query(new_xyz, xyz, radius, nsample)
        else:
            return ball_query_cpu(radius, nsample, xyz, new_xyz)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class CylinderQuery(Function):
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        if _ext is not None and xyz.is_cuda:
            return _ext.cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)
        else:
            return cylinder_query_cpu(radius, hmin, hmax, nsample, xyz, new_xyz, rot)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None


cylinder_query = CylinderQuery.apply


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        # type: (CylinderQueryAndGroup, float, float, float, int, bool) -> None
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)