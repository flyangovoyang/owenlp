import requests
import BeautifulSoup as bs


content = requests.get('http://www.nmc.cn/publish/forecast/ABJ/beijing.html').content
print(content)
