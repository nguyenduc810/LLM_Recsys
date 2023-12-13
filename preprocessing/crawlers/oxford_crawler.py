'''
https://www.oxfordreference.com/display/10.1093/acref/9780199640249.001.0001/acref-9780199640249
'''


from bs4 import BeautifulSoup
import requests


url = 'https://www.oxfordreference.com/display/10.1093/acref/9780199640249.001.0001/acref-9780199640249?btog=chap&hide=true&pageSize=20&skipEditions=true&sort=titlesort&source=%2F10.1093%2Facref%2F9780199640249.001.0001%2Facref-9780199640249'
web = requests.get('https://www.oxfordreference.com/display/10.1093/acref/9780199640249.001.0001/acref-9780199640249?btog=chap&hide=true&page=5&pageSize=20&skipEditions=true&sort=titlesort&source=%2F10.1093%2Facref%2F9780199640249.001.0001%2Facref-9780199640249')
web.text