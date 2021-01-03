<h1>  License Plate Detection and Recognition API  </h1>
  <h3>
  Using : <br>
  
  YOLOV3 for Plate detection and localization<br>
  Contour detection using opencv for character segmentation<br>
  35 class ConvNet classifier for character recognition on a pretrained dataset<br>
  </h3>
 
 <h3>
  This repo contains the code to host pretrained lpr model as a flask microservice
</h3>

<h3>Prerequisite (Requirements) </h3>

```
pip3 install tensorflow==1.14
pip3 install opencv-python
pip3 install pillow 
sudo apt-get install libsm6 libxrender1 libfontconfig1 libxext6 

```
<h3>To run:</h3>

```
export FLASK_APP=app.py

python3 -m flask run --host=0.0.0.0
```


