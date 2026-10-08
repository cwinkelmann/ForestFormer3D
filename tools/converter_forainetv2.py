# Modified from mmdetection3d/tools/dataset_converters/indoor_converter.py
# We just support ScanNet 200.
import os

import mmengine

from forainetv2_data_utils import ForAINetV2Data


ALL_SPLITS = ('train', 'val', 'test')


def create_info_file(data_path,
                        pkl_prefix='forainetv2',
                        save_path=None,
                        workers=4,
                        splits=ALL_SPLITS,
                        test_list=None):
    """Create forainetv2 dataset information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        workers (int, optional): Number of threads to be used. Default: 4.
        splits (Sequence[str], optional): Which splits to build. Default: all
            three. Splits that are not named are left completely alone, so an
            existing pkl of theirs is never rewritten.
        test_list (str, optional): Scan-name list file for the ``test`` split,
            used instead of ``meta_data/test_list.txt``. Default: None.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['forainetv2'], \
        f'unsupported dataset {pkl_prefix}'
    unknown = [s for s in splits if s not in ALL_SPLITS]
    assert not unknown, f'unknown split(s) {unknown}, expected any of {ALL_SPLITS}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    train_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_train.pkl')
    val_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_val.pkl')
    test_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_test.pkl')
    written = []
    for split, filename in [('train', train_filename), ('val', val_filename), ('test', test_filename)]:
        if split not in splits:
            continue
        dataset = ForAINetV2Data(root_path=data_path, split=split,
                                 split_file=test_list if split == 'test' else None)
        if len(dataset) == 0:
            print(f'{pkl_prefix}: no {split} scans with preprocessed data, skipping {filename}')
            continue
        infos = dataset.get_infos(num_workers=workers, has_label=True)
        mmengine.dump(infos, filename, 'pkl')
        print(f'{pkl_prefix} info {split} file is saved to {filename}')
        written.append(filename)
    return written
