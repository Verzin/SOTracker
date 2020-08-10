class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data2/Verzin/d3s/ltr/my'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/data1/TrackingDataset/LaSOT'
        self.got10k_dir = '/data1/TrackingDataset/GOT10k/train_data'
        self.trackingnet_dir = '/data1/TrackingDataset/TrackingNet'
        self.coco_dir = '/data1/TrackingDataset/COCO/PytrackingCOCO'
        self.imagenet_dir = '/data1/TrackingDataset'
        self.imagenetdet_dir = '/data1/TrackingDataset'
        self.vos_dir = '/data1/TrackingDataset/YouTubeVOS2018/train'
