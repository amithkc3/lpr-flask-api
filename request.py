import io
import numpy as np
from PIL import Image
import requests
import sys

def get_plate(img_path='./images/nope.jpg'):
	url = 'http://35.239.52.119:5000/get_plate'
	# url = 'http://localhost:5000/get_plate'

	img = Image.open(img_path)
	buff = io.BytesIO()
	img.save(buff,format='JPEG')
	buff.seek(0)
	response = requests.post(url,data=buff.read())
	print(response._content)
	img.close()

if __name__=='__main__':
	if(len(sys.argv)>1):
		img_path = str(sys.argv[1])
		get_plate(img_path=img_path)
	else:
		get_plate()