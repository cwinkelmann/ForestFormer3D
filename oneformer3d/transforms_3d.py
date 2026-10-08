import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample

from mmdet3d.registry import TRANSFORMS

# Single source of truth for the "ground / treeID 0 is not an instance" rule
# and for renumbering instance ids; see oneformer3d/labels.py.
from .labels import compact_instance_ids_with_ratio, normalize_instance_gt


@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag

@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        if self.merge_non_stuff_cls:
            gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class PointInstClassMapping_(BaseTransform):
    """Delete instances from non-instaces classes.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Added Keys:
    - gt_labels_3d (int)

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes, structured3d=False):
        self.num_classes = num_classes
        self.structured3d = structured3d

    def transform(self, input_dict):
        """Private function for deleting 
            instances from non-instaces classes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            are updated in the result dict. 'gt_labels_3d' is added.
        """

        # because pts_instance_mask contains instances from non-instaces 
        # classes
        pts_instance_mask = np.array(input_dict['pts_instance_mask'])
        pts_semantic_mask = input_dict['pts_semantic_mask']

        if self.structured3d:
            # wall as one instance
            pts_instance_mask[pts_semantic_mask == 0] = \
                pts_instance_mask.max() + 1
            # floor as one instance
            pts_instance_mask[pts_semantic_mask == 1] = \
                pts_instance_mask.max() + 1
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1

        pts_instance_mask, ratio = compact_instance_ids_with_ratio(
            pts_instance_mask, input_dict.get('ratio_inspoint', None))
        input_dict['pts_instance_mask'] = pts_instance_mask
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        if ratio is not None:
            input_dict['ratio_inspoint'] = ratio

        instance_ids = np.unique(pts_instance_mask[pts_instance_mask >= 0])
        gt_labels = np.zeros(len(instance_ids), dtype=int)
        for inst in instance_ids:
            gt_labels[inst] = pts_semantic_mask[pts_instance_mask == inst][0]

        input_dict['gt_labels_3d'] = gt_labels

        return input_dict

@TRANSFORMS.register_module()
class PointSample_(PointSample):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.
        
        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """

        choices = np.random.choice(
            len(points), min(num_samples, len(points)), replace=False)

        return points[choices], choices

    def transform(self, input_dict):
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask',
            'pts_semantic_mask', sp_pts_mask' keys are updated in the 
            result dict.
        """
        points = input_dict['points']

        # if point number smaller than num_point, skip
        if len(points) < self.num_points:
            return input_dict

        points, choices = self._points_random_sampling(
            points, self.num_points)
        input_dict['points'] = points
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            # Dropping points can make an instance disappear entirely, so the
            # ids are renumbered - and ratio_inspoint has to follow them.
            pts_instance_mask, ratio = compact_instance_ids_with_ratio(
                pts_instance_mask[choices], input_dict.get('ratio_inspoint', None))
            input_dict['pts_instance_mask'] = pts_instance_mask
            if ratio is not None:
                input_dict['ratio_inspoint'] = ratio

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask
            
        return input_dict
    
@TRANSFORMS.register_module()
class SkipEmptyScene(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict['gt_labels_3d']) != 0:
            self.inst = input_dict['pts_instance_mask']
            self.sem = input_dict['pts_semantic_mask']
            self.gt_labels = input_dict['gt_labels_3d']
            self.points = input_dict['points']
        else:
            input_dict['pts_instance_mask'] = self.inst
            input_dict['pts_semantic_mask'] = self.sem 
            input_dict['gt_labels_3d'] = self.gt_labels
            input_dict['points'] = self.points

        return input_dict

@TRANSFORMS.register_module()
class SkipEmptyScene_(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict["points"]) == 0:
            return None
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if len(np.unique(pts_instance_mask)) < 2:
            return None

        return input_dict

@TRANSFORMS.register_module()
class CylinderCrop(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict):
        assert 'points' in input_dict.keys()
        points_tensor = input_dict['points'].tensor.numpy()
        center = points_tensor[np.random.randint(points_tensor.shape[0])]
        choices = np.where(
            np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius ** 2)[0]
        input_dict['points'] = input_dict['points'][choices]

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            # normalize_instance_gt() is the single place that knows the
            # ground / treeID-0 rule: ground (semantic 0), points already
            # marked -1 and vegetation whose raw treeID is 0 (no annotated
            # tree) all become -1 instead of forming one big instance. Both
            # the legacy (ground == 0) and the current (ground == -1) .npy
            # conventions end up as -1 here. The crop sees raw ids, so raw=True.
            plot_ids = normalize_instance_gt(
                pts_semantic_mask, pts_instance_mask, raw=True)
            instance_mask_full = plot_ids != -1

            cropped_ids, _ = compact_instance_ids_with_ratio(plot_ids[choices])
            input_dict['pts_instance_mask'] = cropped_ids

            vote_label = np.full((len(choices), 3), np.nan)
            ratio_inspoint = {}
            for new_id in np.unique(cropped_ids):
                if new_id == -1:
                    continue
                in_crop = np.where(cropped_ids == new_id)[0]
                plot_id = plot_ids[choices[in_crop[0]]]
                in_plot = np.where(plot_ids == plot_id)[0]
                # Fraction of the whole-plot tree that survived the crop; the
                # loss rescales the GT mask with it (see get_iou_with_crop).
                ratio_inspoint[int(new_id)] = len(in_crop) / len(in_plot)
                pos = points_tensor[in_plot, :3]
                tree_center = 0.5 * (pos.min(0) + pos.max(0))
                vote_label[in_crop, :] = tree_center - points_tensor[choices[in_crop], :3]

            input_dict['ratio_inspoint'] = ratio_inspoint
            input_dict['vote_label'] = torch.tensor(vote_label, dtype=torch.float32)
            input_dict['instance_mask'] = instance_mask_full[choices]

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = \
                input_dict.get('pts_instance_mask')
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = \
                input_dict.get('instance_mask')

        return input_dict


@TRANSFORMS.register_module()
class GridSample(BaseTransform):
    def __init__(self, grid_size=0.2, mode="train", hash_type="fnv"):
        self.grid_size = grid_size
        self.mode = mode
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec

    def transform(self, input_dict):
        assert "points" in input_dict.keys()
        points = input_dict["points"]
        
        scaled_points = points.tensor / self.grid_size
        grid_points = torch.floor(scaled_points).int()
        min_points = torch.min(grid_points, dim=0).values
        grid_points -= min_points
        scaled_points -= min_points
        min_points = min_points * self.grid_size

        key = self.hash(grid_points)
        idx_sort = torch.argsort(key)
        key_sort = key[idx_sort]
        unique_results = torch.unique(key_sort, return_inverse=True, return_counts=True)
        _, inverse, count = unique_results

        if self.mode == "train":  # train mode
            idx_select = (
                torch.cumsum(torch.cat((torch.tensor([0]), count[:-1])), dim=0)
                + torch.randint(0, count.max(), count.size()) % count
            )
            choices = idx_sort[idx_select]
        else:
            raise NotImplementedError("Only train mode is implemented in this example")

        # Subsampled data
        input_dict["points"] = points[choices]

        #print(input_dict["points"].shape)
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            # Dropping points can make an instance disappear entirely, so the
            # ids are renumbered - and ratio_inspoint has to follow them.
            pts_instance_mask, ratio = compact_instance_ids_with_ratio(
                pts_instance_mask[choices], input_dict.get('ratio_inspoint', None))
            input_dict['pts_instance_mask'] = pts_instance_mask
            if ratio is not None:
                input_dict['ratio_inspoint'] = ratio

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask

        return input_dict

    def fnv_hash_vec(self, vec):
        # Use smaller values to avoid overflow issues
        FNV_prime = torch.tensor(16777619, dtype=torch.int64)
        offset_basis = torch.tensor(2166136261, dtype=torch.int64)
        hash = torch.full((vec.shape[0],), offset_basis, dtype=torch.int64)
        for i in range(vec.shape[1]):
            hash = hash ^ vec[:, i].to(torch.int64)
            hash = hash * FNV_prime
        return hash

    def ravel_hash_vec(self, vec):
        # Implement the ravel hash function for vectors
        vec_max = torch.max(vec, dim=0).values + 1
        hash = torch.ravel_multi_index(vec.t(), vec_max)
        return hash