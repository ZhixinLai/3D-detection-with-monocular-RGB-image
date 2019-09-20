# 3D-detection-with-monocular-RGB-image  
Reference Paper: 
Paper1: 3D Bounding Box Estimation Using Deep Learning and Geometry  URL: https://arxiv.org/abs/1612.00496
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Note: I did the 3D detection research during my internship in MEGVII and most codes including training, testing, lib codes are not allowed posted online, because the codes contains the basemodel and framework information of MEGVII. I want to share my viewpoint and thoughts about 3D detection with monoculr RGB images, so I decide to post the code used for predicting location information in 3D space with 2D boxes after being addimitted by mentor.
The code only uses Numpy and math module.
## Data set and structure
Kitti 2d object: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
Input: monoculr RGB image; 2D boxes, dimension, orientation, and location of objects; Camera's inner and outer parameter.
Data form: JSON.

## Overall Thought
2D box prediction + Orientation prediction + dimentation prediction --> Location inference --> visualization
### 2D box prediction
Applied Faster RCNN and resnet as backbone with 2FNN as header to predict the 2D boxes (left-top point and right-bottom point).
### Orientation prediction
Because of the 2pi range is hard for model to learn, thus divides the 2pi range into several bins and predict the bin class and offset regression has better performance.
#### Different thought:
* predic the alpha directly and used the space constraint to inference Sida_y  vs   predict the Sida_y directly
Paper1 explains the reason why should use method one to do angle prediction, but actually the method two has better performance after doing experients.
* predict angle direcly vs predict sin & cos
same performance
* numbers of bin
4 is best
## BaseModel and Backbone

## Experient

## Results


## 
