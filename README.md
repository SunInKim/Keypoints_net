# Keypointsnet
This algorithm is a neural network that uses the left and right images of an object to detect key points determined according to the type of object.

Keypoint recognition is performed by detecting a heatmap of keypoints on an image. If a key point is detected in each image, the position of the key point in the three-dimensional space can be found using the camera parameters. This enables general manipulation of objects of the same type. The structure of Keypoints-net is shown in the figure below.

figure

Pass two images through the generator to generate heatmap, RGB, and depth images, and then pass them through the generator again. At this time, RGB and depth images are learned to increase the expressive power of the feature map. The keypoint detection result is as follows.

figure

## Data generating

Learning of Keypoints-net is performed through virtual environment data.

Generalization through domain randomization in a virtual environment, eliminating the labor of labeling.

Experimental objects are drawers and mugs, and data are collected for 5 and 10 types, respectively.

Five keypoints were set for the drawer, and three keypoints were set for the mug.

The urdfs for drawers and mugs can be downloaded from Keypoint_urdf at the following link.

link: https://drive.google.com/drive/folders/1vb5gxkhQ9PLQNBXB_dwg47qoHiaw5EOL?usp=sharing

Data generation proceeds through the following code.

```p
cd Data_generator
python get_drawer_data.py
python get_mug_data.py
```
## Training

Learning proceeds by setting the number of keypoints.

```p
cd Keypoint_net
python train.py
```

## UI for checking

The UI is configured so that you can check the results of learning in the real environment.

The first UI 3D_check_ui.ui is a UI that displays 3D coordinates on two images.

figure

The second UI is the UI to set the area of the object in the two images.

figure
