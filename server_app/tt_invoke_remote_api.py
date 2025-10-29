import requests


url = "http://192.168.3.11:9999/predict?features=1,2,3,4;0.1,0.2,0.5,0.8"
response = requests.get(url)
if response.status_code == 200:
    result = response.json()
    print(result)
    print(type(result))
print(response)