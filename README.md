# 3D-detection-with-monocular-RGB-image  
Reference Paper: 
Paper1: 3D Bounding Box Estimation Using Deep Learning and Geometry    
URL: https://arxiv.org/abs/1612.00496  
Paper2: Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction    
URL: https://arxiv.org/abs/1904.01690    
Paper3: 3D Bounding Boxes for Road Vehicles: A One-Stage, Localization Prioritized Approach using Single Monocular Images URL: https://link.springer.com/chapter/10.1007%2F978-3-030-11021-5_39  

I did the 3D detection research during my internship in MEGVII and most codes including training, testing, lib codes are not allowed posted online, because the codes contains the basemodel and framework information of MEGVII.  
Don't leave!!!
I want to share my viewpoint and thoughts about 3D detection with monoculr RGB images. The hardest part and the most tricky part is how to use monocular RGB images to predict location, so I decide to post the code used for predicting location information in 3D space with 2D boxes after being addimitted by mentor. This part code only uses Numpy and math module rather than deep learning framework. Besides, I will also compare different methods to predict orientation and location inference and show the results.
## Data set and structure
Kitti 2d object: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d  
Input: monoculr RGB image; 2D boxes, dimension, orientation, and location of objects; Camera's inner and outer parameter.
Data form: JSON.
## Overall Thought
Class and 2D box prediction + Orientation prediction + dimentation prediction --> Location inference --> visualization
### Class and 2D box prediction
Applied Faster RCNN and resnet as backbone with 2FNN as header to predict the 2D boxes (left-top point and right-bottom point).
### Orientation prediction
Because of the 2pi range is hard for model to learn, thus divides the 2pi range into several bins and predict the bin class and offset regression has better performance.
#### Different thought:
* predic the alpha directly and used the space constraint to inference Sida_y  vs   predict the Sida_y directly
Paper1 explains the reason why should use method one to do angle prediction, but actually the method two has better performance after doing experients.
* predict angle direcly vs predict sin & cos
they have nearly the same performance.
* numbers of bin
Dividing 2Pi into 4 bins has best performance.
### Dimension prediction
Because the objects in Kitti set ranges a lot, it will has bad performance if we predict the dimension for all object directly.
First, calculate the average dimension for each class. Second, regress the offset of each object. Third, according to the object class predicted in the first step, add the average dimension and offset.
### Location inference(Tricky part)
This part is hardest to comprehend and needs background of projection principle and different coordinate systems.
#### Different thought:
* Method one: According to paper1, we can use the 2D box and 3D box relationship to inference 3D location. As we can see from the figure blew, some vertexs of 3D boxes locats in the line of 2D boxes. We can use this principle to do inference location coordinate. First, assuming the location coordinate(center point in 3D box) as x, y, z and then use dimension and orientation figure to calculate the coordinates of 8 vertexs with xyz. Second, transfer the 8 coordinates from world coordinate system into camera coordinate system. Third, each vertex of 3D box is possible located in each line of 2D box, thus there are 4^8 cases. However, because of some priori knowledge and angle prediction result, the number of cases reduce to 64. In this step, we get 64 equations and each quations comtains 4 equation. Forth, solve the 64 equations and then select the best solution as location coordinate xyz.
<img>

* Method two: According to papaer2, because of the priori knowledge that objects in self-driving scene are on the ground, we can use the height, 2D box and projection constraint to inference the depth information(z). And then use z to inference xy with projection constraint.

<img>
*  Method three: The method two supposes that after being projected into 2D images, the center point of 3D box coincide with center point of 2D boxes. Actually, the two points will not coincides with each other exactly. According to paper3, we can predict the 3D projected point in 2D image first(shown as blew) and then use the point to step into method two.
<img>

## Results
### 2D box prediction
<img>
### 2D box prediction
#### method one
<img>
#### method two
<img>
#### method three
<img>
## conclusion
<img>
