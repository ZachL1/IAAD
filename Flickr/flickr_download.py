import flickrapi
import json
import os

os.environ['http_proxy'] = 'http://172.23.192.1:7890'
os.environ['https_proxy'] = 'http://172.23.192.1:7890'

api_key = '00258f9effd82b57b188e8e27de18f1f'
api_secret = '8d9d1c075f25d892'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')
extras = ['description', 'license', 'owner_name', 'targs', 'machine_tags', 'views', 'media']
extras = 'description, license, owner_name, targs, machine_tags, views, media, geo, url_c'
r = flickr.interestingness.getList(date='2022-04-11', extras=extras, per_page='500', page=1)
r_json = json.loads(r.decode('utf-8'))
raw_info_list = r_json['photos']

