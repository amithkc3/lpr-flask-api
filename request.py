import io
import numpy as np
from PIL import Image
import requests

url = 'http://localhost:5000/get_plate'

img = Image.open('./images/car5.jpg')
buff = io.BytesIO()
img.save(buff,format='JPEG')
buff.seek(0)
response = requests.post(url,data=buff.read())
print(response._content)
img.close()