import json
import requests

url = 'http://localhost:5000/dump_registered_plates'
data = json.dumps(['MH12ET1773', 'AP05BK6304', 'MH17AA1199', 'DL4CAH8648', 'AP29E12277', 'MH02BY3294', 'MH01AC9388','MH05AJ9929'])

print(requests.post(url,json=data))


import json
import requests

def get_plates_from_file(path='./registered_plates.json'):
	with open(path,'r') as file:
		json_data = json.load(file)
	return json_data

print(get_plates_from_file())