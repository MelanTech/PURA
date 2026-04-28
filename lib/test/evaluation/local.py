from lib.test.evaluation.environment import EnvSettings
import socket


def local_env_settings():
    hostname = socket.gethostname()
    settings = EnvSettings()

    settings.lasher_path = '/data1/RGBT_Dataset/LasHeR0428'
    settings.vtuav_path = '/data1/RGBT_Dataset/VTUAV'
    settings.rgbt234_path = '/data1/RGBT_Dataset/RGBT234'
    settings.rgbt210_path = '/data1/RGBT_Dataset/RGBT210'
    settings.gtot_path = '/data1/RGBT_Dataset/GTOT'

    settings.save_dir = '/data1/szk/Project/PURA/test_output'
    settings.check_dir = '/data1/szk/Project/PURA/train_output'
    if hostname in ['ICLR', 'ICML']:
        settings.lasher_path = '/data2/RGBT_Dataset/LasHeR0428'
        settings.vtuav_path = '/data2/RGBT_Dataset/VTUAV'

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/szk/Project/PURA/data1/got10k_lmdb'
    settings.got10k_path = '/data1/szk/Project/PURA/data1/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/szk/Project/PURA/data1/itb'
    settings.lasot_extension_subset_path_path = '/data1/szk/Project/PURA/data1/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/szk/Project/PURA/data1/lasot_lmdb'
    settings.lasot_path = '/data1/szk/Project/PURA/data1/lasot'
    settings.network_path = '/data1/szk/Project/PURA/output/test/networks'  # Where tracking networks are stored.
    settings.nfs_path = '/data1/szk/Project/PURA/data1/nfs'
    settings.otb_path = '/data1/szk/Project/PURA/data1/otb'
    settings.prj_dir = '/data1/szk/Project/PURA/'
    settings.result_plot_path = '/data1/szk/Project/PURA/output/test/result_plots'
    settings.results_path = '/data1/szk/Project/PURA/output/test/tracking_results'  # Where to store tracking results

    settings.segmentation_path = '/data1/szk/Project/PURA/output/test/segmentation_results'
    settings.tc128_path = '/data1/szk/Project/PURA/data1/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/szk/Project/PURA/data1/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/szk/Project/PURA/data1/trackingnet'
    settings.uav_path = '/data1/szk/Project/PURA/data1/uav'
    settings.vot18_path = '/data1/szk/Project/PURA/data1/vot2018'
    settings.vot22_path = '/data1/szk/Project/PURA/data1/vot2022'
    settings.vot_path = '/data1/szk/Project/PURA/data1/VOT2019'
    settings.youtubevos_dir = ''

    return settings
