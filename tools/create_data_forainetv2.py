# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from converter_forainetv2 import ALL_SPLITS, create_info_file
from update_infos_to_v2 import update_pkl_infos


def forainetv2_data_prep(root_path, info_prefix, out_dir, workers,
                         splits=ALL_SPLITS, test_list=None):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        splits (Sequence[str]): Splits to build. Splits not named here keep
            whatever pkl they already have.
        test_list (str, optional): Scan-name list file for the test split,
            used instead of ``<root_path>/meta_data/test_list.txt``.
    """
    written = create_info_file(
        root_path, info_prefix, out_dir, workers=workers,
        splits=tuple(splits), test_list=test_list)
    for pkl_path in written:
        update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=pkl_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='forainetv2', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/ForAINetV2',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/ForAINetV2',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='forainetv2')
parser.add_argument(
    '--splits',
    nargs='+',
    default=list(ALL_SPLITS),
    choices=list(ALL_SPLITS),
    help='which splits to build; the others keep their existing pkl')
parser.add_argument(
    '--test-list',
    type=str,
    default=None,
    help='scan-name list file for the test split, instead of '
         'meta_data/test_list.txt (leaves the tracked list untouched)')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset in ('forainetv2'):
        forainetv2_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            splits=args.splits,
            test_list=args.test_list)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
