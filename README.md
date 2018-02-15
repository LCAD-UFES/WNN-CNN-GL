# WNN-CNN-GL

Visual Global Localization with a Hybrid WNN-CNN Approach

| AVAILABLE DOWNLOADS |
| :------------------: |
| [DATASETS](#datasets) |
| [VIDEOS](#videos) |

## Datasets

### UFES Dataset 

[Click here to download]() 

This dataset pertains to the ring road of the _Universidade Federal do Espírito Santo - UFES_ (Vitória, Brazil). It has a total extension of 3.7 km and contains 658 directories identified by their UTM (Universal Transverse Mercator) Zone 24K coordinates {latitude}_{longitude} in meters. 
```
 Each directory contains 504 files in the format: <type><latitude>_<longitude>_<translation>_<rotation>.png 
 <type>: i = crop of remission grid map; r = crop of road grid map; t = color coded road grid map 
 <latitude>_<longitude>: UTM Zone 24K coordinates in meters 
 <translation>: in meters (-1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50) 
 <rotation>: in degrees (0.00, 15.00, 30.00, ..., 315.00, 330.00, 345.00) 
 Example: r7756533_-363795_0.50_315.00.png 
```

## Videos 

These videos shows the operation of the Visual Global Localization module of the IARA Software System. 
They show a 2D grid map on the right and a 3D environment on the left to visualize IARA positions in a global frame of reference. IARA is shown here in our 3D simulation environment with two pictures over it and a dot trail under it indicating its previous positions. The most ahead picture shows a live camera view while the other one is a key frame image recorded previously. Those images are draw at the specific position and orientation where they were taken by the left stereo camera. The key frame image shows a far view of the place depicted in the live image. Also note that, as they are from different recordings, some objects appear in one image but not in the other. The key frame is used for training our place recognition system about the different locations in our university campus, while the pair of key and live-frame is used for training our visual localization system about the relative camera pose represented by the pair of images.

### Video 1: Visual Global Localization using WNN output as key frame 

[Click here to access](https://youtu.be/uVYQZQDbZsA)

Visual Global Localization using WNN output as key frame pose and a CNN to compute 6D relative pose between key frame and live frame.

### Video 2: Visual Global Localization using ground-truth key frame pose 

[Click here to access](https://youtu.be/B_UgAlsW99s)

Visual Global Localization using ground-truth key frame pose and a CNN to compute 6D relative pose between key frame and live frame.
