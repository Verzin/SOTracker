from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = '/data1/TrackingDataset/LaSOT'
    settings.mobiface_path = ''
    settings.network_path = '/data2/Verzin/d3s/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/data1/TrackingDataset/OTB100/'
    settings.results_path = '/data2/Verzin/d3s/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot16_path = '/data1/TrackingDataset/VOT/VOT2016/vot2016/'
    settings.vot18_path = '/data1/TrackingDataset/VOT/VOT2017and2018/'
    settings.vot_path = '/data1/TrackingDataset/VOT/VOT2016/vot2016/'

    return settings

