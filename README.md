# SOTracker:An computer vision software base on deep learning for single object tracking

Python (PyTorch) implementation of SOTracker


## Summary of the SOTracker
SOTracker is a software based on python, which can be run on Windows, MacOS and Linux. Different from any other tracker, SOTracker has its own GUI and supports a wide variety of video format with different resolution and frame rate.Note that,the accuracy of tracker online tracking depends on a large number of images to learning, it requires the frame rate should as high as possible. Generally, 30 fps is enough for the low speed object.

The procedure of the program can be divided into three steps, which is shown in Fig.1. First, user needs to set up several parameters, including the file path, length of object and the information they want to display. Second, user needs to select ROI (region of interest) of the tracking object by drawing a rectangle using the mouse (the target should be framed accurately). After that, press the space bar, and the real time tracking information will be showed on window. During this process, the program can be pause at any time and reselect the tracking target. With the manual intervention, the tracking can be more accurate. At last, the tracking information (target position, motion direction, speed and etc.) will be saved as an excel file, and the trajectory and real-time information will be saved as a video file.

<p style="width:100%, text-align:center"><a href="url"><img src="https://raw.githubusercontent.com/Verzin/SOTracker/Program_processing.png" width="640"></a></p>
## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/Verzin/SOTracker.git.
```

#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```
To install the dependencies on a Windows machine, use the `install.bat` script.
The pre-trained network for the D3S is not part of this repository. You can download it [here](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar).

The tracker was tested on the Ubuntu 16.04 machine with a NVidia GTX 1080 graphics card and cudatoolkit version 9.
It was tested on Window 10 as well, but network training is tested on Linux only.


#### Run the tracker
1.) Specify the path to the D3S [pre-trained segmentation network](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar) by setting the `params.segm_net_path` in the `pytracking/parameters/segm/default_params.py`. <br/>
2.) Specify the path to the VOT 2018 dataset by setting the `vot18_path` in the `pytracking/evaluation/local.py`. <br/>
3.) Activate the conda environment
```bash
conda activate pytracking
```
4.) Run the script pytracking/run_tracker.py to run D3S using VOT18 sequences.  
```bash
cd GUI
python GUI.py
```


#### Training the network
The SOTracker is pre-trained for segmentation task only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files `vos-list-train.txt` and `vos-list-val.txt` from `ltr/data_specs` to the `train` directory of the VOS dataset. 
Set the `vos_dir` variable in `ltr/admin/local.py` to the VOS `train` directory on your machine. 
Download the bounding boxes from [this link](http://data.vicos.si/alanl/d3s/rectangles.zip) and copy them to the sequence directories.
Run training by running the following command:
```bash
python run_training.py segm segm_default
```

## Pytracking
This is a modified version of the python framework pytracking based on **PyTorch**. We would like to thank the authors Martin Danelljan and Goutam Bhat for providing such a great framework.


## Contact
* Verzin (email: verzin@qq.com)
