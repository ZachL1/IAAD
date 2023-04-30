import flickrapi
import json
import os
import pickle
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Lock
# from django.core.cache import cache as dcache
import os
import time
from tqdm import tqdm
import requests

# os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'

class FlickrCrawler:
    def __init__(self, api_key, api_secret, save_dir='./temp', cache=True, filter_views=None):
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json', cache=cache)

        self.save_dir = save_dir
        self.files = {}
        self.files_lock = Lock()
        self.filter_views = filter_views

        self.cache_db = None
        if cache:
            # use django cache for flickrapi requests
            # self.flickr.cache = dcache 

            # use sqlite3 cache for already downloaded photos
            self.cache_db = sqlite3.connect(os.path.join(save_dir, '.cache.db'), check_same_thread=False)
            self.cache_db.execute('CREATE TABLE IF NOT EXISTS downloads (photo_id text, size_label text, available integer, posted_year integer, server text, secret text, favorites integer, views integer, license integer, PRIMARY KEY (photo_id, size_label))')
            self.cache_lock = Lock()

    def abs2rela(self, abs_path: str):
        base_len = len(self.save_dir)
        if not self.save_dir.endswith('/'):
            base_len += 1
        return abs_path[base_len:]

    def save_meta(self, save_path:str, meta:dict):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                self.save_meta(save_path, meta)

        with open(save_path, 'wb') as f:
            pickle.dump(meta, f)

    def get_photo_info(self, id, retry=3):
        if retry <= 0:
            print('continue in ', id)
            return None
        try:
            info_raw = self.flickr.photos.getInfo(photo_id=id)
            info = json.loads(info_raw.decode('utf-8'))
            if info['stat'] != 'ok':
                print('[FAIL] when getInfo get ', info)
                print('continue in ', id)
                return None
            photo_info = info['photo']

            fav_raw = self.flickr.photos.getFavorites(photo_id=id, page=1, per_page=1)
            fav = json.loads(fav_raw.decode('utf-8'))
            if fav['stat'] != 'ok':
                print('[FAIL] when getFavorites get ', info)
                print('continue in ', id)
                return None
            photo_info.update({'favorites': fav['photo']['total']})
        except flickrapi.FlickrError as e:
            print('[except] when getInfo/Favorites get except: ', e)
            return self.get_photo_info(id, retry-1)
        except requests.exceptions.ConnectTimeout as e:
            time.sleep(1)
            print('[except] when getInfo/Favorites get except: ', e)
            return self.get_photo_info(id, retry-1)
        except Exception as e:
            print('[except] when getInfo/Favorites get except: ', e)
            return self.get_photo_info(id, retry-1)


        return photo_info

    def save_rgb(self, rgb_link:str, rgb_file:str, timeout=10, retry=3):
        # if os.path.exists(rgb_file):
        #     return True
        
        if retry <= 0:
            print('continue in ', rgb_link)
            return None
        try:
            save_dir = os.path.dirname(rgb_file)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with urllib.request.urlopen(rgb_link, timeout=timeout) as r:
                with open(rgb_file, 'wb') as f:
                    f.write(r.read())
                    f.close()
        except urllib.error.HTTPError as e:
            print('[except] when save rgb get except: ', e)
            return self.save_rgb(rgb_link, rgb_file, retry=retry-1)
        except urllib.error.URLError as e:
            print('[except] when save rgb get except: ', e)
            return self.save_rgb(rgb_link, rgb_file, retry=retry-1)
        except IOError as e:
            print('[except] when save rgb get except: ', e)
            return self.save_rgb(rgb_link, rgb_file, retry=0)

        return True
    
    def update_files(self, sub_files:dict):
        with self.files_lock:
            for k, v in sub_files.items():
                if k not in self.files:
                    self.files[k] = v
                else:
                    self.files[k].extend(v)
    
    def update_cache_db(self, id_list:list, meta_list:list):
        with self.cache_lock:
            for id, meta in zip(id_list, meta_list):
                posted_year = server = secret = favorites = views = license = None
                if meta['available'] == 1:
                    posted_year = meta['posted_year']
                    server = meta['server']
                    secret = meta['secret']
                    favorites = meta['favorites']
                    views = meta['views']
                    license = meta['license']
                self.cache_db.execute('INSERT OR REPLACE INTO downloads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (id, 'b', meta['available'], posted_year, server, secret, favorites, views, license))
            self.cache_db.commit()

    def get_metadata(self, id_list:list, save_img:bool):
        try:
            # meta_list = [{'available':0}] * len(id_list) # bug, careful
            meta_list = [{'available':0} for i in range(len(id_list))]

            sub_files = {}
            for index, id in enumerate(tqdm(id_list)):
                # get and save photo info meta data
                photo_info = self.get_photo_info(id)
                if photo_info is None:
                    continue
                server = photo_info['server']
                id = photo_info['id']
                secret = photo_info['secret']
                posted_year = int(time.ctime(int(photo_info['dates']['posted']))[-4:])
                rgb_file = os.path.join(self.save_dir, f'rgb/{posted_year}/{id}_{secret}_b.jpg')
                meta_file = os.path.join(self.save_dir, f'meta/{posted_year}/{server}_{id}_{secret}.pkl')
                url = f'https://live.staticflickr.com/{server}/{id}_{secret}_b.jpg'
                photo_info.update({'rgb': self.abs2rela(rgb_file)})
                photo_info.update({'url_b': url})
                # Note: save everything and everyone to .pkl, but just save image for filtered one
                self.save_meta(meta_file, photo_info)

                # meta list for cache
                # Note: cache everyone, but just sotre filtered in json
                meta_list[index]['available'] = 1
                meta_list[index]['posted_year'] = posted_year
                meta_list[index]['server'] = server
                meta_list[index]['secret'] = secret
                meta_list[index]['favorites'] = int(photo_info['favorites'])
                meta_list[index]['views'] = int(photo_info['views'])
                meta_list[index]['license'] = int(photo_info['license'])

                # filter by view number
                if (self.filter_views is not None) and (int(photo_info['views']) < self.filter_views):
                    continue

                # save rgb image, just for filtered one
                if save_img and self.save_rgb(url, rgb_file) is None:
                    continue

                # simple meta data store in json
                meta = {
                    'rgb': self.abs2rela(rgb_file),
                    'meta': self.abs2rela(meta_file),
                    'favorites': int(photo_info['favorites']),
                    'views': int(photo_info['views']),
                    'license': int(photo_info['license']),
                }
                if posted_year not in sub_files.keys():
                    sub_files[posted_year] = []
                sub_files[posted_year].append(meta)

            self.update_files(sub_files)
            if self.cache_db is not None:
                self.update_cache_db(id_list, meta_list)
            print('Great work, done of 1W')
        except Exception as e:
            print('[exception] in get_metadata: ', e)
    
        
    def do_crawler(self, id_list:list, save_img=True, thread_num=10, batch_size=1000, save_json='annos.json', continued=False):
        try:
            # clear
            self.files = {}

            # checkout if id already downloaded
            need_work_id_list = []
            if continued and self.cache_db is not None:
                for img_id in id_list:
                    # generate json by cache
                    ret = self.cache_db.execute('SELECT * FROM downloads WHERE photo_id=? AND size_label=?', (img_id, 'b')).fetchone()
                    if ret is not None:
                        id, size_label, available, posted_year, server, secret, favorites, views, license = ret
                        # unavailable
                        if available != 1:
                            continue
                        # filter by view number
                        if (self.filter_views is not None) and (views < self.filter_views):
                            continue

                        # cache to json
                        if posted_year not in self.files.keys():
                            self.files[posted_year] = []
                        meta = {
                            'rgb': os.path.join('rgb', str(posted_year), f'{id}_{secret}_b.jpg'),
                            'meta': os.path.join('meta', str(posted_year), f'{server}_{id}_{secret}.pkl'),
                            'favorites': favorites,
                            'views': views,
                            'license': license,
                        }

                        # check rgb downloaded
                        rgb_file = os.path.join(self.save_dir, meta['rgb'])
                        if not os.path.exists(rgb_file):
                            print('no rgb, donloading... ', rgb_file)
                            rgb_link = f'https://live.staticflickr.com/{server}/{id}_{secret}_b.jpg'
                            if self.save_rgb(rgb_link, rgb_file) is None:
                                continue

                        self.files[posted_year].append(meta)
                    else:
                        need_work_id_list.append(img_id)
            else:
                need_work_id_list = id_list

            with ThreadPoolExecutor(thread_num) as executor:
                for i in range(0, len(need_work_id_list), batch_size):
                    # self.get_metadata(need_work_id_list[i:i+batch_size], save_img)
                    executor.submit(self.get_metadata, need_work_id_list[i:i+batch_size], save_img)
            
            with open(os.path.join(self.save_dir, save_json), 'w') as f:
                total = 0
                for k,v in self.files.items():
                    total += len(v)
                    print(f'{k}: {len(v)}')
                print(f'total: ', total)
                json.dump({'files': self.files}, f)
        except Exception as e:
            print('[exception] in do_crawler: ', e)
