# u-net_solar

**Solar PV Panels Damage Segmentation with U-Net**

U-Net (plain-vanilla) module developed for specific PV failures segmentation task (6 classes). It tested and verified to work only with specific data (inference-only model). If there is Nvidia GPU in the system module will automatically detect GPU device and runs on it (device should be configured, additional software needed). Otherwise it will use CPU available. 

**The module expects 3 files (located in the same folder with main script)**
- The set of images need to analyzed in ‘img’ folder. Images assumed to be 3-channel color RGB 640x480 IR PV images (it resized to 448x336 px (x0.7) to fit GPU memory for bigger batch size during model training). Module tested with .jpg file extensions. Dataset available on request.
- State dictionary with model weights (model.pth file available on request).
- Configuration file cfg.txt. Here you can change the names for img folder, weights file and names for results returned. This will be convenient when you decide to work with separated sets of images and/or other weights file to perform other segmentation task, so all inputs and outputs for each ''project'' could be located in common directory.

**Results returned**
- Segmentation mask images (human-readable images, see example below). By defaults they stored in newly created ‘view’ folder located in the same directory with main script. For convenience all resulting segmentation masks named as name of the original image with adding prefix ‘_m’ at start of original filename.

![results.png](https://github.com/merr-src/u-net_solar/blob/results.png?raw=true)
  
- Data file data.json with total data collected (see example below), which saved in the same folder with main script. Example of json generated for images above:

~~~{
  "Record_2021-08-20_12-44-15.jpg": 
  [
    {
      "hotspot": {"0": [208,234],
				 "1": [282,228]}
	}
  ],
  "Record_2021-08-20_12-27-38.jpg": 
  [
    {
      "string": {"0": [426,267]}
    },
    {
      "multi_cell": {"0": [278,70]}
    }
  ],
  "Record_2021-08-20_12-43-54.jpg": 
  [
    {
      "hotspot": {"0": [54,262]}
    },
    {
      "vegetation": {"0": [79,262]}
    }
  ]
}
~~~



**Requirements**

All code were developed and tested with python 3.8.5 in Ubuntu 20.04 LTS.
Main python dependences are: cv2, torch, torchvision, numpy, natsort, random, os, json, configparser
All requirements could be installed with terminal: `pip install -r requirements.txt`


**Contributions**

- Working with data (EDA, cleaning, annotation in VIA, preprocessing)  - merr;
- Model (train, check the metrics, train on additional data)  - merr;
- Model results processing and analysis  - merr;
- Documentation - merr :)


**Aknowledgements**

The project was done in 2022 with support of https://fasie.ru and opened by agreement with the customer (Integration Laboratory LLC, Saint-Petersburg).
