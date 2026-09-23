import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean, scatter_add
import MinkowskiEngine as ME

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from mmengine.logging import MessageHub
from .tiling import (SemanticVotes, generate_cylindrical_regions,
                     merge_instances_by_score, relabel_contiguous, sample_region)
from .mask_matrix_nms import mask_matrix_nms
from .ply_io import result_ply_element
import open3d as o3d
import os
import numpy as np
from tools.base_modules import Seq, MLP, FastBatchNorm1d
from .panoptic_losses import offset_loss, discriminative_loss, FastFocalLoss
from torch_cluster import fps
import re
import math
import collections 

# This is the full refactored version using ThreadPoolExecutor for parallel region inference
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement

import contextlib, time


def current_epoch_from_hub() -> int:
    """Epoch published by mmengine's RuntimeInfoHook; 0 when no runner is active."""
    epoch = MessageHub.get_current_instance().get_info('epoch')
    return 0 if epoch is None else int(epoch)


def query_stage_active(prepare_epoch, epoch: int) -> bool:
    """True when query/decoder losses are trained.

    ``prepare_epoch=None`` disables the warm-up entirely; otherwise the decoder
    is trained from epoch ``prepare_epoch + 1`` on (same comparison as before,
    but 0 is no longer treated as "unset").
    """
    return prepare_epoch is None or epoch > prepare_epoch


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

class ScanNetOneFormer3DMixin:
    """Class contains common methods for ScanNet and ScanNet200."""

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
    
    def _select_queries(self, x, gt_instances):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        return queries, gt_instances


@MODELS.register_module()
class ScanNetOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 query_thr,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        x = self.decoder(x, queries)
        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        x = self.decoder(x, x)

        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        #return batch_data_samples
        import os
        import numpy as np
        import open3d as o3d

        pred_pts_seg = batch_data_samples[0].pred_pts_seg
        instance_labels  = pred_pts_seg.instance_labels # tensor, (num_instance,)
        instance_scores = pred_pts_seg.instance_scores # tensor, (num_instance,)
        pts_instance_mask = pred_pts_seg.pts_instance_mask[0] # tensor, (num_instances, num_points)
        input_points = batch_inputs_dict["points"][0] # tensor, (num_points, xyzrgb)
        input_point_name = batch_data_samples[0].lidar_path.split('/')[-1].split('.')[0]

        def save_point_cloud(points, file_path):
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()  # Convert tensor to NumPy array on the CPU
            points = np.asarray(points, dtype=np.float32)  # Ensure points are in float32 format
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])
            pc.colors = o3d.utility.Vector3dVector(points[:, 3:])
            o3d.io.write_point_cloud(file_path, pc)

        def filter_and_save_instances(instance_labels, instance_scores, pts_instance_mask, input_points,input_point_name, threshold=0.2):

            base_dir = f"./work_dirs/{input_point_name}"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            input_pc_path = os.path.join(base_dir, f"{input_point_name}.ply")
            save_point_cloud(input_points, input_pc_path)

            instance_count = {}
            for i in range(len(instance_scores)):
                if instance_scores[i] >= threshold:
                    label = instance_labels[i].item()
                    if label not in instance_count:
                        instance_count[label] = 0
                    instance_count[label] += 1
                    #print(pts_instance_mask[i])
                    instance_mask = pts_instance_mask[i].astype(bool)
                    instance_points = input_points[instance_mask]
                    instance_pc_path = os.path.join(base_dir, f"{input_point_name}_{label}_{instance_count[label]}.ply")
                    save_point_cloud(instance_points, instance_pc_path)

        filter_and_save_instances(instance_labels, instance_scores, pts_instance_mask, input_points, input_point_name)

        return batch_data_samples

@MODELS.register_module()
class ForAINetV2OneFormer3D(Base3DDetector):
    r"""for-instance dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 stuff_classes,
                 thing_cls,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 radius = 16):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.stuff_classes = stuff_classes
        self.thing_cls = thing_cls
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.radius = radius
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))   
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))  

        x = self.extract_feat(x)  

        x = self.decoder(x)  

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i] 
            voxel_superpoints = torch.unique(voxel_superpoints,  
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask 
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask 
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)  
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)    
            
            batch_data_samples[i].gt_instances_3d.labels_3d, batch_data_samples[i].gt_instances_3d.sp_inst_masks, batch_data_samples[i].gt_instances_3d.ratio_inspoint = \
                                self.filter_stuff_masks(batch_data_samples[i].gt_instances_3d, self.stuff_classes, batch_data_samples[i].gt_pts_seg.ratio_inspoint)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)  

        loss = self.criterion(x, sp_gt_instances)  #unified_criterion.py __call__
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        lidar_path = batch_data_samples[0].lidar_path
        base_name = os.path.basename(lidar_path)
        current_filename = os.path.splitext(base_name)[0]
        #if 'val' in lidar_path:
        if 'test' in lidar_path:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            from plyfile import PlyData, PlyElement
            from tqdm import tqdm
            step_size = self.radius
            grid_size = 0.2
            num_points = 640000
            pts_semantic_gt = batch_data_samples[0].eval_ann_info['pts_semantic_mask']
            pts_instance_gt = batch_data_samples[0].eval_ann_info['pts_instance_mask']
            original_points = batch_inputs_dict['points'][0]
            regions = self.generate_cylindrical_regions(original_points, self.radius, step_size)
            all_pre_sem = [list() for _ in range(original_points.shape[0])]
            all_pre_ins = np.full(original_points.shape[0], -1)
            max_instance = 0

            global_instance_scores = np.zeros((original_points.shape[0],), dtype=float) 

            best_masks = []

            for region_idx, region in enumerate(tqdm(regions, desc="Processing regions")):
                region_mask = ((original_points[:, 0] - region[0]) ** 2 + (original_points[:, 1] - region[1]) ** 2) <= self.radius ** 2
                pc1 = original_points[region_mask]
                pc1_indices = torch.where(region_mask)[0]

                if len(pc1) == 0:
                    continue

                pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
                if len(pc2) < num_points:
                    pc3 = pc2
                    pc3_indices = pc2_indices
                elif len(pc2) > num_points:
                    pc3, pc3_indices = self.points_random_sampling(pc2, pc2_indices, num_points)

                coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
                x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
                x = self.extract_feat(x)
                x = self.decoder(x)
                results_list = self.predict_by_feat_test(x, inverse_mapping2, pc3)
                
                # Collect masks and their scores, process them immediately
                masks = results_list[0].pts_instance_mask[0]
                scores = results_list[0].instance_scores
                valid_scores_mask = scores > 0.6
                masks = masks[valid_scores_mask]
                scores = scores[valid_scores_mask]

                # Nearest neighbor mapping for masks to pc1
                for mask, score in zip(masks, scores):
                    mask_pc1 = self.nearest_neighbor_mapping(pc1, pc3, mask)
                    mask_points = pc1_indices[mask_pc1].cpu().numpy()

                    # Vectorized update of global instance mask and scores
                    update_mask = score > global_instance_scores[mask_points]
                    global_instance_scores[mask_points[update_mask]] = score
                    all_pre_ins[mask_points[update_mask]] = max_instance

                    if np.any(update_mask):
                        # Add the new mask
                        best_masks.append((mask_points, max_instance, score))

                    max_instance += 1

                cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, results_list[0].pts_semantic_mask[0])
                 
                originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                all_pre_sem = self.vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre)

                originids = pc3_indices.cpu().numpy()  # Use pc3_indices for ground truth labels
                # Get gt labels for pc3
                pc3_semantic_gt = pts_semantic_gt[originids]
                pc3_instance_gt = pts_instance_gt[originids]
                # Save each pc3 to a separate .ply file
                #region_dir = f"work_dirs/oneformer3d_radius20_e2039_test_bm2/{current_filename}/region_{region_idx}"
                #region_ply_path = os.path.join(region_dir, "pc_ins_sem.ply")
                #self.save_ply(pc3.cpu().numpy(), results_list[0].pts_semantic_mask[0], results_list[0].pts_instance_mask[1], region_ply_path, pc3_semantic_gt, pc3_instance_gt)


                # Save each mask to a separate .ply file
                #for mask_idx, (mask, score) in enumerate(zip(results_list[0].pts_instance_mask[0], results_list[0].instance_scores)):
                #    if score > 0.4:
                #        mask_points = pc3.cpu().numpy()[mask]
                #        mask_file_path = os.path.join(region_dir, f"mask{mask_idx}_{score:.2f}.ply")
                #        self.save_ply_2(mask_points, np.full(mask_points.shape[0], mask_idx), mask_file_path)


                # Save the intermediate results
                #region_ply_path = os.path.join(region_dir, "complete_ins_pre.ply")
                #self.save_ply_2_withscore(original_points.cpu().numpy(), all_pre_ins, region_ply_path, global_instance_scores)

            # Post-processing step
            final_semantic_labels = self.finalize_semantic_labels(all_pre_sem)
            ground_mask = (final_semantic_labels == 0)
            all_pre_ins[ground_mask] = -1

            # Remove instances with fewer than 10 points
            unique_instances, instance_counts = np.unique(all_pre_ins, return_counts=True)
            small_instances = unique_instances[instance_counts < 10]
            for instance in small_instances:
                all_pre_ins[all_pre_ins == instance] = -1

            # Remove replaced old masks
            unique_best_masks = []
            for mask_points, instance_id, score in best_masks:
                if np.any(all_pre_ins[mask_points] == instance_id):
                    unique_best_masks.append((mask_points, instance_id, score))

            # Save the best masks
            #best_mask_dir = f"work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian/{current_filename}/best_masks_before_block_merge"
            #os.makedirs(best_mask_dir, exist_ok=True)
            #for mask_points, instance_id, score in unique_best_masks:
            #    mask_file_path = os.path.join(best_mask_dir, f"best_mask_{instance_id}_{score}.ply")
            #    self.save_ply_2_withscore(original_points[mask_points], np.full(mask_points.shape[0], instance_id), mask_file_path, global_instance_scores[mask_points])

            # Merge masks
            #clean_all_pre_ins, merged_masks = self.merge_overlapping_instances(all_pre_ins, unique_best_masks)
            clean_all_pre_ins, merged_masks = self.merge_overlapping_instances_by_score(all_pre_ins, unique_best_masks)

            # Save the masks after block merge
            #best_mask_after_merge_dir = f"work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian/{current_filename}/best_masks_after_block_merge"
            #os.makedirs(best_mask_after_merge_dir, exist_ok=True)
            #for mask_points, instance_id, score in merged_masks:
            #    mask_file_path = os.path.join(best_mask_after_merge_dir, f"best_mask_{instance_id}_{score}.ply")
            #    self.save_ply_2_withscore(original_points[mask_points], np.full(mask_points.shape[0], instance_id), mask_file_path, global_instance_scores[mask_points])

            # Re-label instances to ensure continuous labeling
            unique_labels = np.unique(clean_all_pre_ins)
            unique_labels = unique_labels[unique_labels >= 0]  # Exclude background label (-1)
            relabel_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            relabel_map[-1] = -1  # Keep background as -1
            clean_all_pre_ins = np.vectorize(relabel_map.get)(clean_all_pre_ins)

            # Save the final combined results
            region_path = f"work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian/{current_filename}_final_results.ply"
            self.save_ply_withscore(original_points.cpu().numpy(), final_semantic_labels, clean_all_pre_ins, global_instance_scores, region_path, pts_semantic_gt, pts_instance_gt)
            
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]
                data_sample.pred_pts_seg['originids'] = originids
            return batch_data_samples
        else:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))

            x = self.extract_feat(x)

            x = self.decoder(x)

            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]

            return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        inst_res = self.pred_inst_sem(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res)
        pan_res = self.pred_pan(pred_masks, pred_scores,
                                superpoints, sem_res)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def predict_by_feat_test(self, out, superpoints, coordinates):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        # Calculate ground_z_max from coordinates of points classified as ground
        ground_points = coordinates[sem_res == 0]
        ground_z_max = ground_points[:, 2].max().item() if ground_points.size(0) > 0 else float('inf')

        
        inst_res = self.pred_inst_sem_test(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res, coordinates, ground_z_max)
        pan_res = self.pred_pan_sem(pred_masks, pred_scores, 
                                superpoints, sem_res, coordinates, ground_z_max)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, 
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem(self, pred_masks, pred_scores,
                  superpoints, score_threshold, sem_res):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem_test(self, pred_masks, pred_scores,
                  superpoints, score_threshold, sem_res, coordinates, ground_z_max):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # Filter instances based on z values
        for i in range(mask_pred.size(0)):
            mask = mask_pred[i]
            if mask.sum().item() == 0:
                scores[i] = 0
                continue
            z_values = coordinates[mask, 2]  # Get z values where mask is True
            if z_values.numel() > 0 and z_values.min().item() > ground_z_max + 5:
                scores[i] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores,
                 superpoints, sem_res):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            superpoints, thr, sem_res)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map
    
    def pred_pan_sem(self, pred_masks, pred_scores,
                 superpoints, sem_res, coordinates, ground_z_max):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem_test(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            superpoints, thr, sem_res, coordinates, ground_z_max)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (num_classes, n_points).
        """

        # Convert mask_src to one-hot encoding
        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes).float()

        # Aggregate class counts for each voxel
        sp_masks = scatter_add(mask, sp_pts_mask, dim=0)

        # Determine the class with the maximum count in each voxel
        sp_masks = sp_masks.argmax(dim=-1)

        # Convert the result back to one-hot encoding
        sp_masks = torch.nn.functional.one_hot(sp_masks, num_classes=num_classes).float()

        # Transpose to get the shape (num_classes, n_points)
        sp_masks = sp_masks.T

        # Ensure the output dimensions match the expected shape
        assert sp_masks.shape == (num_classes, sp_pts_mask.max().item() + 1)

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks
    
    @staticmethod
    def filter_stuff_masks(batch_data_samples_i, stuff_classes, ratio_inspoint):
        """Drop stuff instances; crop ratios are looked up by instance id.

        Row ``i`` of ``sp_inst_masks`` is instance id ``i`` (``get_gt_inst_masks``
        one-hot encodes the ids in order), and ``ratio_inspoint`` is keyed by
        the ids in force after the last transform that compacted them. A
        missing key means the two drifted apart, which would silently rescale
        the wrong instance's IoU, so fail loudly instead.
        """
        labels_3d = batch_data_samples_i.labels_3d
        sp_inst_masks = batch_data_samples_i.sp_inst_masks
        n_inst = len(labels_3d)
        missing = [i for i in range(n_inst) if i not in ratio_inspoint]
        if missing:
            raise KeyError(f'ratio_inspoint lacks instance ids {missing}; '
                           f'keys are {sorted(int(k) for k in ratio_inspoint)}')
        ratio_tensor = torch.tensor([float(ratio_inspoint[i]) for i in range(n_inst)],
                                    device=labels_3d.device)
        keep = ~torch.isin(
            labels_3d, torch.tensor(stuff_classes, device=labels_3d.device))

        return labels_3d[keep], sp_inst_masks[keep], ratio_tensor[keep]
    
    @staticmethod
    def generate_cylindrical_regions(points, radius, step_size):
        x_coords = points[:, 0].cpu()  
        y_coords = points[:, 1].cpu() 

        x_min, x_max = x_coords.min().item(), x_coords.max().item()
        y_min, y_max = y_coords.min().item(), y_coords.max().item()

        regions = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                regions.append((x, y))
                y += step_size
            x += step_size

        return regions

    @staticmethod
    def grid_sample(points, indices, grid_size):
        scaled_points = points / grid_size
        grid_points = torch.floor(scaled_points).int()
        
        # Use unique to find indices of each voxel
        unique_grid_points, inverse_indices = torch.unique(grid_points, return_inverse=True, dim=0)

        # Calculate the mean coordinates for each voxel
        unique_points = torch.zeros((len(unique_grid_points), points.size(1)), dtype=points.dtype, device=points.device)
        unique_indices = torch.zeros(len(unique_grid_points), dtype=indices.dtype, device=indices.device)
        for i in range(len(unique_grid_points)):
            mask = (inverse_indices == i)
            unique_points[i] = points[mask].mean(dim=0)
            unique_indices[i] = indices[mask][0]  # Just pick one of the indices in the voxel

        return unique_points, unique_indices

    @staticmethod
    def points_random_sampling(points, indices, num_points):
        choices = np.random.choice(len(points), num_points, replace=False)
        sampled_points = points[choices]
        sampled_indices = indices[choices]
        return sampled_points, sampled_indices

    @staticmethod
    def nearest_neighbor_mapping(pc1, pc3, predictions):
        from sklearn.neighbors import NearestNeighbors

        # Ensure pc3 is on CPU
        pc3_cpu = pc3.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pc3_cpu)
        distances, indices = nbrs.kneighbors(pc1.cpu().numpy())

        mapped_predictions = predictions[indices.squeeze()]

        return mapped_predictions

    @staticmethod
    def vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre):
        for idx, originid in zip(originids, cylinder_current_semantic_pre):
            all_pre_sem[idx].append(originid.item())

        return all_pre_sem

    @staticmethod
    def region_merging(all_pre_ins, max_instance, pre_ins, originids):
        idx = np.argwhere(all_pre_ins[originids] != -1)  # has label
        idx2 = np.argwhere(all_pre_ins[originids] == -1)  # no label

        if len(idx) == 0:
            mask_valid = pre_ins != -1
            all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance
            max_instance = max_instance + len(np.unique(pre_ins[mask_valid]))
        elif len(idx2) == 0:
            return all_pre_ins, max_instance
        else:
            new_label = pre_ins.reshape(-1)
            unique_labels = np.unique(new_label)
        
            # Ignore the background label (-1)
            unique_labels = unique_labels[unique_labels != -1]
            
            for ii_idx in unique_labels:
                new_label_ii_idx = originids[np.argwhere(new_label == ii_idx).reshape(-1)]
                
                new_has_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] != -1]
                new_not_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] == -1]

                if len(new_has_old_idx) == 0:
                    all_pre_ins[new_not_old_idx] = max_instance
                    max_instance += 1
                elif len(new_not_old_idx) == 0:
                    continue
                else:
                    old_labels_ii = all_pre_ins[new_has_old_idx]
                    un = np.unique(old_labels_ii)
                    max_iou_ii = 0
                    max_iou_ii_oldlabel = 0
                    for g in un:
                        idx_old_all = np.argwhere(all_pre_ins == g).reshape(-1)
                        inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                        iou1 = float(inter_label_idx.size) / float(idx_old_all.size)
                        iou2 = float(inter_label_idx.size) / float(new_label_ii_idx.size)
                        iou = max(iou1, iou2)

                        if iou > max_iou_ii:
                            max_iou_ii = iou
                            max_iou_ii_oldlabel = g

                    if max_iou_ii > 0.3:
                        all_pre_ins[new_not_old_idx] = max_iou_ii_oldlabel
                    else:
                        all_pre_ins[new_not_old_idx] = max_instance
                        max_instance += 1

        return all_pre_ins, max_instance

    @staticmethod
    def save_ply(points, semantic_pred, instance_pred, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)
    
    @staticmethod
    def save_ply_2(points, instance_pred, filename):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i],) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)

    @staticmethod
    def save_ply_2_withscore(points, instance_pred, filename, scores):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4'), ('score', 'f4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i], valid_scores[i]) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)

    @staticmethod
    def save_ply_withscore(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        """Write the per-point result cloud as a binary little-endian PLY.

        Same field names and dtypes as before (see `oneformer3d/ply_io.py`); only
        the encoding changed from ASCII to binary, which removes a `numpy.savetxt`
        over every point plus a per-row Python tuple comprehension (5-10 s per
        100 m tile, see docs/benchmarks/2026-09-23-inference-profile.md).
        """
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)

        el = result_ply_element(points, semantic_pred, instance_pred, scores,
                                semantic_gt, instance_gt)
        PlyData([el], text=False, byte_order='<').write(filename)

    @staticmethod
    def finalize_semantic_labels(all_pre_sem):
        from collections import Counter
        final_semantic_labels = np.full(len(all_pre_sem), -1)
        for i, labels in enumerate(all_pre_sem):
            if labels:
                final_semantic_labels[i] = Counter(labels).most_common(1)[0][0]
        return final_semantic_labels
    
    @staticmethod
    def merge_overlapping_instances(all_pre_ins, best_masks, iou_threshold=0.6):
        """
        Merge overlapping instances based on IoU.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id).
            iou_threshold (float): IoU threshold for merging instances.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
        """
        from scipy.sparse import csr_matrix

        # Create a sparse matrix for each mask using csr_matrix
        num_points = all_pre_ins.shape[0]
        num_masks = len(best_masks)
        data = []
        row_indices = []
        col_indices = []

        for idx, (mask_points, instance_id, score) in enumerate(best_masks):
            data.extend([1] * len(mask_points))
            row_indices.extend(mask_points)
            col_indices.extend([idx] * len(mask_points))

        mask_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_points, num_masks), dtype=np.float32)

        # Compute the IoU between masks
        intersection = mask_matrix.T @ mask_matrix
        mask_sizes = mask_matrix.sum(axis=0).A1
        union = mask_sizes[:, None] + mask_sizes - intersection

        # Ensure no division by zero
        union[union == 0] = 1

        iou = intersection / union

        print(f"Computed IoU matrix:\n{iou}")

        # Use Union-Find to manage merging of masks
        uf = UnionFind(num_masks)
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                if iou[i, j] > iou_threshold:
                    print(f"Merging instances {best_masks[i][1]} and {best_masks[j][1]} with IoU {iou[i, j]}")
                    uf.union(i, j)

        # Update masks dynamically after merging
        merged_masks = []
        merged_instance_labels = np.copy(all_pre_ins)

        for i in range(num_masks):
            root = uf.find(i)
            if root == i:  # If this is the root, create a new merged mask
                merged_points = np.unique(np.concatenate([best_masks[k][0] for k in range(num_masks) if uf.find(k) == i]))
                merged_instance_id = best_masks[i][1]  # Use the instance_id of the root
                merged_score = max([best_masks[k][2] for k in range(num_masks) if uf.find(k) == i])  # Take the max score
                merged_masks.append((merged_points, merged_instance_id, merged_score))

                # Update point-wise labels
                merged_instance_labels[merged_points] = merged_instance_id

        return merged_instance_labels, merged_masks

    @staticmethod
    def merge_overlapping_instances_by_score(all_pre_ins, best_masks, overlap_threshold=0.3):
        """
        Merge overlapping instances based on score and point overlap ratio.
        
        This method compares each mask's points with the union of two masks and determines 
        which mask to keep based on its points' proportion in the union.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id, score).
            overlap_threshold (float): Overlap threshold for merging instances based on mask proportion in union.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
            list: List of merged masks after applying the score-based merging.
        """
        num_masks = len(best_masks)

        # Initialize all points as unassigned (-1) if necessary
        all_pre_ins = np.full(all_pre_ins.shape, -1, dtype=int)
        mask_kept = np.ones(num_masks, dtype=bool)  # Track which masks are kept

        for i in range(num_masks):
            if not mask_kept[i]:
                continue
            mask_i_points = set(best_masks[i][0])
            for j in range(i + 1, num_masks):
                if not mask_kept[j]:
                    continue
                mask_j_points = set(best_masks[j][0])

                # Calculate the intersection of the point sets
                intersection_points = mask_i_points & mask_j_points  # Intersection of the point sets
                if len(intersection_points) == 0:
                    continue  # No overlap, skip

                # Calculate the proportion of intersection relative to each mask
                mask1_ratio = len(intersection_points) / len(mask_i_points)  # intersection / mask1
                mask2_ratio = len(intersection_points) / len(mask_j_points)  # intersection / mask2

                # If either mask's proportion in the intersection is greater than the threshold, merge
                if mask1_ratio > overlap_threshold or mask2_ratio > overlap_threshold:
                    if best_masks[i][2] >= best_masks[j][2]:  # Keep the one with the higher score
                        mask_kept[j] = False  # Discard mask2
                    else:
                        mask_kept[i] = False  # Discard mask1
                        break  # If mask i is discarded, no need to compare with others


        # Filter the masks to keep only those that are not discarded
        masks_after_score_merge = [best_masks[i] for i in range(num_masks) if mask_kept[i]]

        # Update point-wise instance labels
        merged_instance_labels = np.copy(all_pre_ins)
        for mask_points, instance_id, _ in masks_after_score_merge:
            merged_instance_labels[mask_points] = instance_id

        return merged_instance_labels, masks_after_score_merge
    
@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery(Base3DDetector):
    r"""FOR-instance dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 stuff_classes,
                 thing_cls,
                 query_point_num=200,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 prepare_epoch=None,
                 #prepare_epoch2=None,
                 radius = 16,
                 chunk = 20_000):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.stuff_classes = stuff_classes
        self.thing_cls = thing_cls
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.prepare_epoch = prepare_epoch
        #self.prepare_epoch2 = prepare_epoch2
        self._init_layers(in_channels, num_channels)
        self.Embed = Seq().append(MLP([num_channels, num_channels], bias=False))
        self.Embed.append(torch.nn.Linear(num_channels, 5))
        self.query_point_num = query_point_num
        self.radius = radius
        self.chunk = chunk
        self.BiSemantic = (
            Seq()  
            .append(MLP([num_channels, num_channels], bias=False))  
            .append(torch.nn.Linear(num_channels, 2))  
            .append(torch.nn.LogSoftmax(dim=-1))  
        )
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))    
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))  

        x = self.extract_feat(x) 

        embed_logits = [self.Embed(y) for y in x]
        bi_semantic_logits = [self.BiSemantic(y) for y in x] 
        
        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bi = 0 
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            # Get voxel indices for the current sample
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            
            pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask  # Instance labels (per-point)
            instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask  # Boolean mask for foreground points
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask  # Semantic labels (per-point)

            device = pts_instance_mask.device

            # Filter out background points using instance_mask
            valid_instance_mask = instance_mask

            # First, compute voxel_instance_labels using the previously provided method
            voxel_instance_labels = self.get_voxel_instance_labels(
                pts_instance_mask[valid_instance_mask], 
                voxel_superpoints[valid_instance_mask]
            )
            

            #Find the valid voxels containing foreground points
            valid_voxel_indices = torch.unique(voxel_superpoints[valid_instance_mask])  # Unique voxel indices with valid points
            
            # Filter embed_logits based on valid voxels
            filtered_embed_logits = embed_logits[i][valid_voxel_indices]  # Select only the valid voxels
        
            # Use voxel_instance_labels for discriminative loss
            batch = torch.full_like(voxel_instance_labels, i)  # Simulating batch index for each voxel
            discriminative_losses = discriminative_loss(
                filtered_embed_logits,  # Using the filtered embedding logits
                voxel_instance_labels,
                batch,
                5
            )

            # Accumulate discriminative loss for this sample
            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += loss if loss_name == "ins_loss" else 0

            # Use the precomputed bi_semantic_logits for the current sample
            bi_semantic_logit = bi_semantic_logits[i]

            # Sum the foreground (instance_mask) per voxel and determine whether the voxel is background/foreground
            instance_mask = torch.as_tensor(instance_mask).to(device)
            voxel_point_counts = scatter_add(torch.ones_like(instance_mask.float()), voxel_superpoints, dim=0)
            foreground_voxel_counts = scatter_add(instance_mask.float(), voxel_superpoints, dim=0)

            # A voxel with more than half foreground points is foreground.
            bi_y = ((foreground_voxel_counts / voxel_point_counts) > 0.5).long()
            # Vegetation without a tree id (instance -1 but not ground) is
            # ignored by the binary head; it keeps its class in the 3-class loss.
            ignore_pts = (sem_mask != 0) & (pts_instance_mask == -1)
            ignore_voxel_counts = scatter_add(ignore_pts.float(), voxel_superpoints, dim=0)
            bi_y[(ignore_voxel_counts / voxel_point_counts) > 0.5] = -100

            if (bi_y != -100).any():
                semantic_loss_bi = torch.nn.functional.nll_loss(
                    bi_semantic_logit, bi_y, ignore_index=-100)
            else:
                semantic_loss_bi = bi_semantic_logit.sum() * 0.0
            
            # Accumulate semantic loss for the batch
            total_semantic_loss_bi += semantic_loss_bi

        # Average the accumulated losses over the batch
        total_discriminative_loss /= batch_size
        total_semantic_loss_bi /= batch_size

        # Add to total loss
        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi
        }

        queries = []
        queries_inslabel = []
        queries_idx = []

        if query_stage_active(self.prepare_epoch, current_epoch_from_hub()):
            total_qscore_loss = 0
            for i in range(batch_size):
                voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
                pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask  # Instance labels (per-point)
                instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask
                valid_voxel_indices = torch.unique(voxel_superpoints[instance_mask])
                
                with torch.no_grad():
                    if valid_voxel_indices.numel() < 10:
                        queries.append([])
                        queries_inslabel.append([])
                        continue
                    voxel_instance_labels = self.get_voxel_instance_labels(
                                        pts_instance_mask[instance_mask], 
                                        voxel_superpoints[instance_mask]
                                    )
                    
                    wood_class = 1
                    
                    semantic_predictions_bi = torch.argmax(bi_semantic_logits[i], dim=1)
                    tree_indices = torch.where(semantic_predictions_bi == wood_class)[0]  #all voxel
                    if tree_indices.numel() == 0:
                        # no predicted foreground in this crop: nothing to sample queries from
                        queries.append([])
                        queries_inslabel.append([])
                        continue

                    #FPS from all tree points
                    batch_tensor_4 = torch.zeros(embed_logits[i][tree_indices].size(0), dtype=torch.long).to(embed_logits[i].device)  # Ensure batch_tensor on same device
                    topk_indices_4 = fps(embed_logits[i][tree_indices], batch_tensor_4, ratio=min(self.query_point_num / embed_logits[i][tree_indices].size(0), torch.tensor([1.0]).to(embed_logits[i].device)))
                    selected_indices_case4 = tree_indices[topk_indices_4]

                    # Retrieve relevant information for the selected points
                    current_points = batch_inputs_dict['points'][i]
                    current_points_add = scatter_add(current_points, voxel_superpoints, dim=0)
                    voxel_counts = scatter_add(torch.ones_like(current_points[:, 0].float()), voxel_superpoints, dim=0)
                    avg_points = current_points_add / voxel_counts.unsqueeze(-1).clamp(min=1)

                    # add content queries
                    queries.append(x[i][selected_indices_case4])
                    
                    pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask  # Instance labels (per-point)
                    voxel_instance_labels = self.get_voxel_instance_labels(
                        pts_instance_mask, 
                        voxel_superpoints
                    )

                    # ins labels for queries
                    queries_inslabel.append(voxel_instance_labels[selected_indices_case4])

                    queries_idx.append(selected_indices_case4)

            if all(len(q) == 0 for q in queries):
                pass
            else:
                # First check if the length of x and queries are the same
                if any(len(q) == 0 for q in queries):

                    # Use list comprehension to filter out empty queries and save original indices
                    filtered_results = [
                        (x[i], queries[i], batch_data_samples[i], queries_inslabel[i], i)  # Keep the original index i
                        for i in range(len(queries))
                        if len(queries[i]) > 0  # Only keep non-empty queries
                    ]
                    # Unpack filtered results into separate lists
                    x, queries, batch_data_samples, queries_inslabel, original_indices = zip(*filtered_results)
                    # Convert the zipped result back to list format
                    x = list(x)
                    queries = list(queries)
                    batch_data_samples = list(batch_data_samples)
                    queries_inslabel = list(queries_inslabel)
                    original_indices = list(original_indices)  # Keep track of original indices
                else:
                    original_indices = list(range(len(batch_data_samples)))
                    
                x = self.decoder(x, queries)

                sp_gt_instances = []
                for i in range(len(batch_data_samples)):
                    voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                                inverse_mapping] == original_indices[i]] #[326894]
                    voxel_superpoints = torch.unique(voxel_superpoints,  
                                                    return_inverse=True)[1]
                    inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask 
                    sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask 
                    assert voxel_superpoints.shape == inst_mask.shape

                    batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                        self.get_gt_semantic_masks(sem_mask,
                                                                    voxel_superpoints,
                                                                    self.num_classes)  
                    batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                        self.get_gt_inst_masks(inst_mask,
                                                            voxel_superpoints) 
                    
                    batch_data_samples[i].gt_instances_3d.labels_3d, batch_data_samples[i].gt_instances_3d.sp_inst_masks, batch_data_samples[i].gt_instances_3d.ratio_inspoint = \
                                        self.filter_stuff_masks(batch_data_samples[i].gt_instances_3d, self.stuff_classes, batch_data_samples[i].gt_pts_seg.ratio_inspoint)
                    
                    batch_data_samples[i].gt_instances_3d.query_inslabel = queries_inslabel[i]
                    

                    sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)  

                loss = self.criterion(x, sp_gt_instances) 
                loss_final.update(loss)
        return loss_final

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict semantic and instance masks.

        ``test_cfg.full_plot`` (default True) runs tiled inference over a whole
        plot and writes ``<test_cfg.output_dir>/<scan>.ply``; ``False`` scores
        the batch as pre-cropped cylinders (validation during training).
        """
        if self.test_cfg.get('full_plot', True):
            return self._predict_full_plot(batch_inputs_dict, batch_data_samples)
        return self._predict_crop(batch_inputs_dict, batch_data_samples)

    def _predict_crop(self, batch_inputs_dict, batch_data_samples):
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        x = self.extract_feat(x)

        queries = []
        for i in range(len(x)):
            max_len = min(self.query_point_num, len(x[i]))
            queries.append(x[i][0:max_len])
        x = self.decoder(x, queries)

        results_list = self.predict_by_feat(x, inverse_mapping)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def _predict_full_plot(self, batch_inputs_dict, batch_data_samples):
        """Tiled inference: cylinders of ``self.radius`` on a lattice with
        step ``radius / 4``; per-tile masks are merged by score, semantics are
        voted per point."""
        assert len(batch_data_samples) == 1, 'full-plot inference expects batch_size 1'
        data_sample = batch_data_samples[0]
        scan_name = os.path.splitext(os.path.basename(data_sample.lidar_path))[0]
        eval_ann = data_sample.eval_ann_info if data_sample.eval_ann_info is not None else {}
        pts_semantic_gt = eval_ann.get('pts_semantic_mask', None)
        pts_instance_gt = eval_ann.get('pts_instance_mask', None)

        cfg = self.test_cfg
        output_dir = cfg.get('output_dir', None) or 'work_dirs/default_output'
        score_th = float(cfg.get('score_th', 0.4))
        overlap_threshold = float(cfg.get('overlap_threshold', 0.3))
        num_cls = cfg.num_sem_cls
        step_size = self.radius / 4
        grid_size = 0.2          # voxel size of the tile downsampling
        max_points = 640_000     # cap per tile; lower it on small GPUs

        points = batch_inputs_dict['points'][0]
        n_total = points.shape[0]
        device = points.device
        regions = generate_cylindrical_regions(points[:, :2], self.radius, step_size)
        votes = SemanticVotes(n_total, num_cls)
        mask_indices = []        # global point indices of every kept tile mask (CPU)
        mask_scores = []         # its score

        for cx, cy in regions.tolist():
            region_mask = ((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) <= self.radius ** 2
            pc1_indices = torch.where(region_mask)[0]
            if pc1_indices.numel() == 0:
                continue
            pc1 = points[pc1_indices]
            pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
            pc3, pc3_indices = sample_region(pc2, pc2_indices, max_points)

            coordinates, features, inverse_mapping, spatial_shape = self.collate([pc3])
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape, 1)
            x = self.extract_feat(x)
            embed_logits = self.Embed(x[0])
            bi_semantic_logits = self.BiSemantic(x[0])
            tree_indices = torch.where(torch.argmax(bi_semantic_logits, dim=1) == 1)[0]

            # nearest pc3 point for every pc1 point, chunked to bound memory
            nn_idx = torch.cat([
                torch.cdist(pc1[s:s + self.chunk, :3].float(), pc3[:, :3].float()).argmin(1)
                for s in range(0, pc1.shape[0], self.chunk)])

            if tree_indices.numel() > 1:
                batch_vec = torch.zeros(tree_indices.numel(), dtype=torch.long, device=device)
                ratio = min(self.query_point_num / tree_indices.numel(), 1.0)
                selected = tree_indices[fps(embed_logits[tree_indices], batch_vec, ratio=ratio)]
                x = self.decoder(x, [x[0][selected]])
                result = self.predict_by_feat_test(x, inverse_mapping, pc3, selected)[0]

                sem_pc3 = torch.as_tensor(result.pts_semantic_mask[0], device=device).long()
                votes.add(pc1_indices, sem_pc3[nn_idx])

                masks = torch.as_tensor(result.pts_instance_mask[0], device=device)   # (K, N3) bool
                scores = torch.as_tensor(result.instance_scores, device=device).float()
                if masks.shape[0] > 0:
                    near_edge = torch.sqrt((pc3[:, 0] - cx) ** 2 + (pc3[:, 1] - cy) ** 2) \
                        > (self.radius - 0.5)
                    touches_edge = (masks & near_edge.unsqueeze(0)).any(1)
                    keep = torch.where((scores > score_th) & ~touches_edge)[0]
                    for k in keep.tolist():
                        mask_indices.append(pc1_indices[masks[k][nn_idx]].cpu())
                        mask_scores.append(float(scores[k]))
            else:
                bi_pc3 = torch.argmax(bi_semantic_logits[inverse_mapping], dim=1)
                votes.add_binary(pc1_indices, bi_pc3[nn_idx] == 1)

            del pc1, pc2, pc3, x, embed_logits, bi_semantic_logits, nn_idx
            torch.cuda.empty_cache()

        semantic_pred = votes.resolve()                                   # (N,) cpu
        instance_pred, kept = merge_instances_by_score(
            mask_indices, torch.tensor(mask_scores), overlap_threshold, num_points=n_total)
        point_scores = torch.full((n_total,), -1.0)
        if kept.numel():
            kept_scores = torch.tensor(mask_scores)[kept]
            assigned = instance_pred >= 0
            point_scores[assigned] = kept_scores[instance_pred[assigned]]
        instance_pred[semantic_pred == 0] = -1                            # ground has no instance
        ids, counts = torch.unique(instance_pred, return_counts=True)
        small = ids[(ids >= 0) & (counts < cfg.npoint_thr)]
        if small.numel():
            instance_pred[torch.isin(instance_pred, small)] = -1
        instance_pred = relabel_contiguous(instance_pred)
        point_scores[instance_pred < 0] = -1.0

        sem_np = semantic_pred.numpy().astype(np.int32)
        inst_np = instance_pred.numpy().astype(np.int32)
        score_np = point_scores.numpy().astype(np.float32)
        self.save_ply_withscore(points[:, :3].cpu().numpy(), sem_np, inst_np, score_np,
                                os.path.join(output_dir, f'{scan_name}.ply'),
                                pts_semantic_gt, pts_instance_gt)
        # index 1 is what UnifiedSegMetric reads; index 0 keeps the shape of crop mode
        data_sample.pred_pts_seg = PointData(
            pts_semantic_mask=[sem_np, sem_np],
            pts_instance_mask=[inst_np, inst_np],
            instance_scores=score_np)
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        #pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        #inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
        #                          pred_scores[:-self.test_cfg.num_sem_cls, :],
        #                          #pred_labels[:-self.test_cfg.num_sem_cls, :],
        #                          superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        inst_res = self.pred_inst_sem(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res)
        pan_res = self.pred_pan(pred_masks, pred_scores, #pred_labels,
                                superpoints, sem_res)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def predict_by_feat_test(self, out, superpoints, coordinates, queries):
        """Instance and semantic masks for one tile (no panoptic pass).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask` ([array (N,)]),
                `pts_instance_mask` ([bool array (K, N)]), `instance_labels`,
                `instance_scores`, `query_select_voxel_idx`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]
        n_cls = self.test_cfg.num_sem_cls

        sem_res = self.pred_sem(pred_masks[-n_cls:, :], superpoints)
        ground_points = coordinates[sem_res == 0]
        ground_z_max = ground_points[:, 2].max().item() if ground_points.size(0) > 0 else float('inf')

        inst_res = self.pred_inst_sem_test(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :], superpoints,
            self.test_cfg.inst_score_thr, sem_res, coordinates, ground_z_max, queries)

        return [PointData(
            pts_semantic_mask=[sem_res.cpu().numpy()],
            pts_instance_mask=[inst_res[0].cpu().bool().numpy()],
            instance_labels=inst_res[1].cpu().numpy(),
            instance_scores=inst_res[2].cpu().numpy(),
            query_select_voxel_idx=inst_res[3].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold, sem_res):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.query_point_num,#203 - self.test_cfg.num_sem_cls, #self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)
        #for i in range(mask_pred.size(0)):
        #    mask = mask_pred[i]
        #    # Get the semantic categories of the points within the mask
        #    mask_sem_res = sem_res[mask == 1]
        #    # Check if the majority of the points' semantic categories belong to stuff_cls
        #    if torch.isin(mask_sem_res, stuff_cls_tensor).sum().item() > mask_sem_res.size(0) / 2:
        #        # If true, set the corresponding score to 0
        #        scores[i] = 0

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem_test(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold, sem_res, coordinates, ground_z_max, queries):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                queries.shape[0],
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            min(self.test_cfg.topk_insts, queries.shape[0]), sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        queries_select = queries[topk_idx]
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)
        else:
            keep_inds = torch.arange(scores.shape[0], device=scores.device)

        queries_select = queries_select[keep_inds]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)
        #for i in range(mask_pred.size(0)):
        #    mask = mask_pred[i]
        #    # Get the semantic categories of the points within the mask
        #    mask_sem_res = sem_res[mask == 1]
        #    # Check if the majority of the points' semantic categories belong to stuff_cls
        #    if torch.isin(mask_sem_res, stuff_cls_tensor).sum().item() > mask_sem_res.size(0) / 2:
        #        # If true, set the corresponding score to 0
        #        scores[i] = 0

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # Filter instances based on z values.
        # Vectorised form of the former per-mask Python loop: a mask with no
        # points, or whose lowest point sits more than 5 m above the highest
        # ground point, scores 0. The loop issued three blocking `.item()` GPU
        # syncs per mask (~590 per tile, 81 % of prediction time; see
        # docs/benchmarks/2026-09-23-inference-profile.md); this does the same
        # work in a handful of kernels with bit-identical results.
        if mask_pred.shape[0] > 0:
            if mask_pred.shape[1] == 0:
                # no points at all -> every mask is empty
                scores[:] = 0
            else:
                z = coordinates[:, 2]
                inf = torch.tensor(float('inf'), dtype=z.dtype, device=z.device)
                # `torch.where` materialises a (K, N) float block; chunk over the
                # masks so the peak allocation stays bounded on large tiles. The
                # result does not depend on the chunk size.
                chunk = max(1, 8_000_000 // mask_pred.shape[1])
                z_min = torch.empty(
                    mask_pred.shape[0], dtype=z.dtype, device=z.device)
                for start in range(0, mask_pred.shape[0], chunk):
                    block = mask_pred[start:start + chunk]
                    z_min[start:start + chunk] = torch.where(
                        block, z.unsqueeze(0), inf).min(dim=1).values
                # An empty mask has z_min == inf, which the height test alone
                # would not catch when ground_z_max is inf (no ground points).
                empty = ~mask_pred.any(dim=1)
                scores[empty | (z_min > ground_z_max + 5)] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries_select = queries_select[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries_select = queries_select[npoint_mask]

        return mask_pred, labels, scores, queries_select
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, #pred_labels,
                 superpoints, sem_res):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            #pred_labels[:-n_cls, :], superpoints, thr)
            superpoints, thr, sem_res)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map
    
    def pred_pan_sem(self, pred_masks, pred_scores, #pred_labels,
                 superpoints, sem_res, coordinates, ground_z_max, queries):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores, queries_select = self.pred_inst_sem_test(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            #pred_labels[:-n_cls, :], superpoints, thr)
            superpoints, thr, sem_res, coordinates, ground_z_max, queries)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]
        queries_select = queries_select[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map, queries_select

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]
        queries_select = queries_select[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        # Track which queries are kept
        queries_retained = torch.ones_like(queries_select, dtype=torch.bool)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0
                # Mark the corresponding query as removed
                queries_retained[inst-1] = False  # Note: inst-1 is used as inst_idxs starts from 1

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask

        # Return queries that were retained (those not deleted)
        queries_select = queries_select[queries_retained]
        return sem_map_src_mapping, sem_map, queries_select

    @staticmethod
    def get_voxel_instance_labels(pts_instance_mask, voxel_superpoints):
        """Aggregate instance labels for each voxel.
        
        Args:
            pts_instance_mask (Tensor): Instance labels for each point (n_raw_points,).
            voxel_superpoints (Tensor): Voxel indices for each point (n_raw_points,).
        
        Returns:
            voxel_instance_labels (Tensor): Aggregated instance labels for each voxel.
        """
        # Treat the background (-1) points as a new instance, for proper aggregation
        instance_mask_clone = pts_instance_mask.clone()
        background_label = None
        if torch.any(instance_mask_clone == -1):
            background_label = torch.max(instance_mask_clone) + 1
            instance_mask_clone[instance_mask_clone == -1] = background_label
        
        _, instance_mask_clone = torch.unique(instance_mask_clone, return_inverse=True, sorted=True)
        # Convert the instance labels to one-hot encoding
        one_hot_inst_mask = torch.nn.functional.one_hot(instance_mask_clone)

        _, inverse_indices = torch.unique(voxel_superpoints, return_inverse=True, sorted=True)

        # Use scatter_add to aggregate one-hot labels for each voxel
        voxel_instance_counts = scatter_add(one_hot_inst_mask.float(), inverse_indices, dim=0)
        
        # Determine the most frequent instance label in each voxel (argmax along the label dimension)
        voxel_instance_labels = torch.argmax(voxel_instance_counts, dim=-1)
        
        # Convert background label back to -1, if background_label exists
        if background_label is not None:
            voxel_instance_labels[voxel_instance_labels == background_label] = -1
        
        return voxel_instance_labels
    
    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (num_classes, n_points).
        """

        # Convert mask_src to one-hot encoding
        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes).float()

        # Aggregate class counts for each voxel
        sp_masks = scatter_add(mask, sp_pts_mask, dim=0)

        # Determine the class with the maximum count in each voxel
        sp_masks = sp_masks.argmax(dim=-1)

        # Convert the result back to one-hot encoding
        sp_masks = torch.nn.functional.one_hot(sp_masks, num_classes=num_classes).float()

        # Transpose to get the shape (num_classes, n_points)
        sp_masks = sp_masks.T

        # Ensure the output dimensions match the expected shape
        assert sp_masks.shape == (num_classes, sp_pts_mask.max().item() + 1)

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks
    
    @staticmethod
    def filter_stuff_masks(batch_data_samples_i, stuff_classes, ratio_inspoint):
        """Drop stuff instances; crop ratios are looked up by instance id.

        Row ``i`` of ``sp_inst_masks`` is instance id ``i`` (``get_gt_inst_masks``
        one-hot encodes the ids in order), and ``ratio_inspoint`` is keyed by
        the ids in force after the last transform that compacted them. A
        missing key means the two drifted apart, which would silently rescale
        the wrong instance's IoU, so fail loudly instead.
        """
        labels_3d = batch_data_samples_i.labels_3d
        sp_inst_masks = batch_data_samples_i.sp_inst_masks
        n_inst = len(labels_3d)
        missing = [i for i in range(n_inst) if i not in ratio_inspoint]
        if missing:
            raise KeyError(f'ratio_inspoint lacks instance ids {missing}; '
                           f'keys are {sorted(int(k) for k in ratio_inspoint)}')
        ratio_tensor = torch.tensor([float(ratio_inspoint[i]) for i in range(n_inst)],
                                    device=labels_3d.device)
        keep = ~torch.isin(
            labels_3d, torch.tensor(stuff_classes, device=labels_3d.device))

        return labels_3d[keep], sp_inst_masks[keep], ratio_tensor[keep]
    
    @staticmethod
    def grid_sample(points: torch.Tensor,
                    indices: torch.Tensor,
                    grid_size: float):
        """
        Voxel‑downsample point cloud by averaging points in each voxel.

        Args
        ----
        points  : (N, 3)  xyz or xyzf  GPU tensor
        indices : (N,)    original indices (int64 / int32) GPU tensor
        grid_size : float voxel size

        Returns
        -------
        vox_points  : (M, 3)   averaged coords per voxel (same dtype/device)
        vox_indices : (M,)     one representative original index per voxel
        """
        # 1. voxel coordinate (int32)
        voxel = torch.floor(points / grid_size).to(torch.int32)          # (N,3)

        # 2. find unique voxels
        uniq, inverse = torch.unique(voxel, return_inverse=True, dim=0)   # (N,) inverse ∈ [0,M)

        M = uniq.size(0)

        # 3. scatter‑add coords  (sum / count → mean)
        ones   = torch.ones_like(inverse, dtype=points.dtype)             # (N,)

        sum_xyz = torch.zeros((M, points.size(1)), device=points.device, dtype=points.dtype)
        cnt_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)

        sum_xyz.index_add_(0, inverse, points)        # Σ xyz
        cnt_xyz.index_add_(0, inverse, ones)          # Σ 1

        vox_points = sum_xyz / cnt_xyz.unsqueeze(1)   # mean

        # 4. pick a representative original index  
        vox_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
        vox_indices.index_copy_(0, inverse, indices)

        return vox_points, vox_indices


    @staticmethod
    def save_ply(points, semantic_pred, instance_pred, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)
    
    @staticmethod
    def save_ply_withscore(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        """Write the per-point result cloud as a binary little-endian PLY.

        Same field names and dtypes as before (see `oneformer3d/ply_io.py`); only
        the encoding changed from ASCII to binary, which removes a `numpy.savetxt`
        over every point plus a per-row Python tuple comprehension (5-10 s per
        100 m tile, see docs/benchmarks/2026-09-23-inference-profile.md).
        """
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)

        el = result_ply_element(points, semantic_pred, instance_pred, scores,
                                semantic_gt, instance_gt)
        PlyData([el], text=False, byte_order='<').write(filename)

    @staticmethod
    def save_bluepoints(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement

        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename and index
        match = re.search(r'_(\d+)\.ply$', filename)
        if match:
            num = int(match.group(1)) + 1
            base_name = filename[:match.start(1)]
        else:
            num = 1
            base_name = filename.replace('.ply', '')
            #if not base_name.endswith('_'):
            #    base_name += '_'

        # Ensure "bluepoints" does not get repeated in filename
        if "bluepoints" in base_name:
            base_name = re.sub(r'_bluepoints_', '', base_name)  # Remove trailing "_bluepoints" if it exists

        new_filename = f"{base_name}_{num}.ply"
        new_filename_filtered = f"{base_name}_bluepoints_{num}.ply"

        # Determine previous bluepoints file
        prev_bluepoints_filename = f"{base_name}_bluepoints_{num-1}.ply"

        # If previous bluepoints file exists, load its semantic_pred
        if os.path.exists(prev_bluepoints_filename):
            print(f"Loading semantic_pred from {prev_bluepoints_filename}")
            plydata = PlyData.read(prev_bluepoints_filename)
            prev_semantic_pred = np.array(plydata['vertex']['semantic_pred'])  # 读取存储的 semantic_pred
            semantic_pred = prev_semantic_pred  # Override input semantic_pred

        # Filter points that meet the condition
        mask = (semantic_pred != 0) & (instance_pred == -1)
        points_filtered = points[mask]
        semantic_pred_filtered = semantic_pred[mask]  # 也保存 semantic_pred
        semantic_gt_filtered = semantic_gt[mask] if semantic_gt is not None else None
        instance_gt_filtered = instance_gt[mask] if instance_gt is not None else None

        # Keep points that do not meet the condition
        points_remain = points[~mask]
        semantic_pred_remain = semantic_pred[~mask]
        instance_pred_remain = instance_pred[~mask]
        scores_remain = scores[~mask]
        semantic_gt_remain = semantic_gt[~mask] if semantic_gt is not None else None
        instance_gt_remain = instance_gt[~mask] if instance_gt is not None else None

        # Save the unfiltered point cloud only if it contains points
        if points_remain.shape[0] > 0:
            dtype_remain = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
            if semantic_gt is not None and instance_gt is not None:
                dtype_remain += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
                vertex_remain = np.array(
                    [tuple(points_remain[i]) + (semantic_pred_remain[i], instance_pred_remain[i], scores_remain[i], semantic_gt_remain[i], instance_gt_remain[i])
                    for i in range(points_remain.shape[0])], dtype=dtype_remain)
            else:
                vertex_remain = np.array(
                    [tuple(points_remain[i]) + (semantic_pred_remain[i], instance_pred_remain[i], scores_remain[i])
                    for i in range(points_remain.shape[0])], dtype=dtype_remain)

            el_remain = PlyElement.describe(vertex_remain, 'vertex')
            PlyData([el_remain], text=False).write(new_filename)

        # Save the filtered point cloud (with semantic_pred)
        if points_filtered.shape[0] > 0:
            dtype_filtered = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('semantic_pred', 'i4'), ('semantic_seg', 'i4'), ('treeID', 'i4')]  # 添加 semantic_pred
            vertex_filtered = np.array(
                [tuple(points_filtered[i]) + (semantic_pred_filtered[i], semantic_gt_filtered[i]+1, instance_gt_filtered[i])
                for i in range(points_filtered.shape[0])], dtype=dtype_filtered)

            el_filtered = PlyElement.describe(vertex_filtered, 'vertex')
            PlyData([el_filtered], text=False).write(new_filename_filtered)


@MODELS.register_module()
class ScanNet200OneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field).features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        queries, gt_instances = self._select_queries(x, gt_instances)
        x = self.decoder(x, queries)
        return self.criterion(x, gt_instances)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        x = self.decoder(x, x)
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples


@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints,
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  pred_labels[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels,
                                superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels,
                 superpoints):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :], superpoints, thr)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(
            mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class InstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(InstanceOnlyOneFormer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        self.num_classes_2dataset = num_classes_2dataset
        
        self.prefix_1dataset = prefix_1dataset 
        self.prefix_2dataset = prefix_2dataset
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[
                coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_1dataset in scene_name:
            labels = torch.arange(
                self.num_classes_1dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_1dataset,  
                    1).flatten(0, 1)
        elif self.prefix_2dataset in scene_name:
            labels = torch.arange(
                self.num_classes_2dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_2dataset,
                    1).flatten(0, 1)          
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_1dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_1dataset, 
                                 rounding_mode='floor')
        elif self.prefix_2dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_2dataset,
                                 rounding_mode='floor')        
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores)
        ]
