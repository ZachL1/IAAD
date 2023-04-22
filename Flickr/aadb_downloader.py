import json
import os

os.environ['http_proxy'] = 'http://172.23.192.1:7890'
os.environ['https_proxy'] = 'http://172.23.192.1:7890'


with open('/mnt/d/data/AADB/annotations/AADB_all.json', 'r') as f:
    data = json.load(f)

id_list = []
for sample in data['files']:
    img = sample['image']
    img_split = img.split('/')[-1].split('_')
    assert(len(img_split) == 5)

    id_list.append(img_split[2])


api_key = '00258f9effd82b57b188e8e27de18f1f'
api_secret = '8d9d1c075f25d892'
from flickr_crawler import FlickrCrawler
crawler = FlickrCrawler(api_key=api_key, api_secret=api_secret, save_dir='/mnt/d/data/AADB')
crawler.do_crawler(id_list, save_img=False)




with open('/mnt/d/data/FLICKR-AES/annotations/FLICKR-AES_all.json', 'r') as f:
    data = json.load(f)
    
id_list = []
for sample in data['files']:
    img = sample['image']
    img_split = img.split('/')[-1].split('_')
    assert(len(img_split) == 5 or len(img_split) == 4)

    id_list.append(img_split[2])

crawler = FlickrCrawler(api_key=api_key, api_secret=api_secret, save_dir='/mnt/d/data/FLICKR-AES')
crawler.do_crawler(id_list, save_img=False)