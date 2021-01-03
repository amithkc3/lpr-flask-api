<h1>  License Plate Detection and Recognition API  </h1>
  <h3>  The LPR pipeline used in this repo consists of : </h3>
  
  1.  YOLOV3 for Plate detection and localization<br>
  2.  Contour detection using opencv for character segmentation<br>
  3.  35 class ConvNet classifier for character recognition on a pretrained dataset<br>  
 
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
<h3>To run thhe flask server:</h3>

```
export FLASK_APP=app.py

python3 -m flask run --host=0.0.0.0
```

Test the server using 
```
python3 request.py
```

<h3>To run the lpr on a image</h3>
change the image path in the execution block of the code and run <br>

```
python3 lpr.py
```
