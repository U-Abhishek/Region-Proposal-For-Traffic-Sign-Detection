## Region Proposal Network
Traffic sign recognition has multiple real-world applications, 
such as driver safety monitoring and decision-making in 
self-driving cars. The key function of these comuputer 
vision systems is to recognize and interpret the objects in a 
traffic situation.

## Datset
[German Traffic Sign Detection Benchmark (GTSDB)](https://benchmark.ini.rub.de/gtsdb_news.html) dataset is used 
in this project.
The GTSDB dataset comprises real-life traffic environment recorded
on diverse types of roads (rural, highway, urban) during the
daylight and at nightfall, and various weather conditions are 
featured.\
This dataset is composed of 1206 traffic signs that are split into:
- a training set of 600 images with 846 traffic signs.
- a testing set of 300 images with 360 traffic signs.\
Which were extracted from 900 full images each of these images contains one, multiple, or zero traffic signs which naturally suffer 
from contrasts in light conditions and orientations.

## Selective Search
Object proposal algorithms are developed to find regions in the 
image that are most likely to contain objects. In this way,
background regions can be taken out of the pipeline to attain
higher speeds.\
Selective search is one of the most robust object proposal methods
used in R-CNN. It divides the image into patches by grouping
similar regions based on color, size, texture and shape.

Code for Selective Search: [selective search.ipynb](https://github.com/U-Abhishek/Region-Proposal-For-Traffic-Sign-Detection/blob/master/selective%20search.ipynb)

#### Object proposal by Selective Search:
<img src="https://user-images.githubusercontent.com/86155658/151668255-58976f37-84b1-439c-8757-121270687a57.png" alt="drawing" style="width:700px;"/>

## Customized Object Proposal
As you can see in above picture selective search is giving may region proposals 
with no traffic signs in them taking this into consideration COP(customize Object
Proposal) is developed.

Code for COP : [Custom Region Proposal.ipynb](https://github.com/U-Abhishek/Region-Proposal-For-Traffic-Sign-Detection/blob/master/Custom%20Region%20Proposal.ipynb)

#### Object proposal by COP:
<img src="https://user-images.githubusercontent.com/86155658/151670115-009272fd-2aae-45b0-88bc-b41818b111a3.png" alt="drawing" style="width:700px;"/>

## Elements of COP
- Color Threhsolding: [color thresholding.ipynb](https://github.com/U-Abhishek/Region-Proposal-For-Traffic-Sign-Detection/blob/master/color%20thresholding.ipynb) 
[*****************]
- Contour Filtering: [COLOUR Filter and Pixal density.ipynb](https://github.com/U-Abhishek/Region-Proposal-For-Traffic-Sign-Detection/blob/master/COLOUR%20Filter%20and%20Pixal%20density.ipynb) 
[*****************]

