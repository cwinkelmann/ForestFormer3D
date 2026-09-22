# Modified from mmdetection3d/tools/dataset_converters/indoor_converter.py
# We just support ScanNet 200.
import os

import mmengine

from forainetv2_data_utils import ForAINetV2Data


def create_info_file(data_path,
                        pkl_prefix='forainetv2',
                        save_path=None,
                        workers=4):
    """Create forainetv2 dataset information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['forainetv2'], \
        f'unsupported dataset {pkl_prefix}'
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
        dataset = ForAINetV2Data(root_path=data_path, split=split)
        if len(dataset) == 0:
            print(f'{pkl_prefix}: no {split} scans with preprocessed data, skipping {filename}')
            continue
        infos = dataset.get_infos(num_workers=workers, has_label=True)
        mmengine.dump(infos, filename, 'pkl')
        print(f'{pkl_prefix} info {split} file is saved to {filename}')
        written.append(filename)
    return written
