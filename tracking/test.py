import os
import sys
import argparse
import os.path as osp


# from lib_mine.mutils.add_sys_path import add_sys_path

def add_sys_path(path):
    if path not in sys.path:
        sys.path.append(path)


prj_path = osp.normpath(osp.join(os.path.dirname(__file__), '..'))
add_sys_path(os.path.join(prj_path, 'lib_mine'))
add_sys_path(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=1):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    # lasher_test vtuav_test rgbt234
    # dataset_name = 'lasher_test'
    # threads = 5
    # num_gpus = 1

    threads = threads * num_gpus
    dataset = get_dataset(dataset_name)

    # sequence = 'Otcbvs'
    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    print(trackers[0].get_parameters().checkpoint)

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=str, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    # parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    args = parser.parse_args()

    # GlobalVar.my_test_checkpoint = args.checkpoint

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    from lib_mine.mutils.gpu_info import GPUInfo
    import socket

    hostname = socket.gethostname()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        if hostname == 'ECCV':
            # cuda_list = GPUInfo().get_topk(topk=2)
            # cuda_list.remove(0)
            # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_list))
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        elif hostname == 'CVPR':  # CVPR
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, GPUInfo().get_topk(topk=5)))
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
