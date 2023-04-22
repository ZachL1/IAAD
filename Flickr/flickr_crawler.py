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

# os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'

class FlickrCrawler:
    def __init__(self, api_key, api_secret, save_dir='./temp', cache=True):
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json', cache=cache)

        self.save_dir = save_dir
        self.files = {}
        self.files_lock = Lock()

        self.cache_db = None
        if cache:
            # use django cache for flickrapi requests
            # self.flickr.cache = dcache 

            # use sqlite3 cache for already downloaded photos
            self.cache_db = sqlite3.connect(os.path.join(save_dir, '.cache.db'))
            self.cache_db.execute('CREATE TABLE IF NOT EXISTS downloads (photo_id text, size_label text, available integer, taken_year text, server text, secret text, PRIMARY KEY (photo_id, size_label))')
            self.cache_lock = Lock()

    def abs2rela(self, abs_path: str):
        base_len = len(self.save_dir)
        if not self.save_dir.endswith('/'):
            base_len += 1
        return abs_path[base_len:]

    def save_meta(self, save_path:str, meta:dict):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            print(f'Retry {retry} times more')
            return self.get_photo_info(id, retry-1)

        return photo_info

    def save_rgb(self, rgb_link:str, rgb_file:str, timeout=10, retry=3):
        if retry <= 0:
            print('continue in ', rgb_link)
            return None
        try:
            save_dir = os.path.dirname(rgb_file)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            r = urllib.request.urlopen(rgb_link, timeout=timeout)
            with open(rgb_file, 'wb') as f:
                f.write(r.read())
                f.close()
        except urllib.error.HTTPError as e:
            print('[except] when save rgb get except: ', e)
            self.save_rgb(rgb_link, rgb_file, retry=retry-1)
        except urllib.error.URLError as e:
            print('[except] when save rgb get except: ', e)
            self.save_rgb(rgb_link, rgb_file, retry=retry-1)
        except IOError as e:
            print('[except] when save rgb get except: ', e)
            self.save_rgb(rgb_link, rgb_file, retry=0)

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
                taken_year = server = secret = None
                if meta['available'] == 1:
                    taken_year = meta['taken_year']
                    server = meta['server']
                    secret = meta['secret']
                self.cache_db.execute('INSERT INTO downloads VALUES (?, ?, ?, ?, ?, ?)', (id, 'b', meta['available'], taken_year, server, secret))
            self.cache_db.commit()

    def get_metadata(self, id_list:list, save_img:bool):
        # meta_list = [{'available':0}] * len(id_list)
        meta_list = [{'available':0} for i in range(len(id_list))]

        sub_files = {}
        for index, id in enumerate(id_list):
            photo_info = self.get_photo_info(id)
            if photo_info is None:
                continue

            server = photo_info['server']
            id = photo_info['id']
            secret = photo_info['secret']
            taken_year = photo_info['dates']['taken'][:4] # TODO: just str year?
            rgb_file = os.path.join(self.save_dir, f'rgb/{taken_year}/{id}_{secret}_b.jpg')
            meta_file = os.path.join(self.save_dir, f'meta/{taken_year}/{server}_{id}_{secret}.pkl')
            url = f'https://live.staticflickr.com/{server}/{id}_{secret}_b.jpg'
            photo_info.update({'rgb': self.abs2rela(rgb_file)})
            photo_info.update({'url_b': url})

            if save_img and self.save_rgb(url, rgb_file) is None:
                continue
            self.save_meta(meta_file, photo_info)

            meta_list[index]['available'] = 1
            meta_list[index]['taken_year'] = taken_year
            meta_list[index]['server'] = server
            meta_list[index]['secret'] = secret
            if taken_year not in sub_files.keys():
                sub_files[taken_year] = []
            sub_files[taken_year].append(self.abs2rela(meta_file))

        self.update_files(sub_files)
        if self.cache_db is not None:
            self.update_cache_db(id_list, meta_list)
        
    def do_crawler(self, id_list:list, save_img=True, thread_num=10, cnt_per=1000):
        # checkout if id already downloaded
        need_work_id_list = []
        if self.cache_db is not None:
            for img_id in id_list:
                ret = self.cache_db.execute('SELECT * FROM downloads WHERE photo_id=? AND size_label=?', (img_id, 'b')).fetchone()
                if ret is not None:
                    id, size_label, available, taken_year, server, secret = ret
                    if available==1 and taken_year not in self.files.keys():
                        self.files[taken_year] = []
                    if available==1 and taken_year in self.files.keys():
                        self.files[taken_year].append(os.path.join('meta', taken_year, f'{server}_{id}_{secret}.pkl'))
                else:
                    need_work_id_list.append(img_id)
        else:
            need_work_id_list = id_list

        with ThreadPoolExecutor(thread_num) as executor:
            for i in range(0, len(need_work_id_list), cnt_per):
                executor.submit(self.get_metadata, need_work_id_list[i:i+cnt_per], save_img)
        
        with open(os.path.join(self.save_dir, 'annos.json'), 'w') as f:
            json.dump({'files': self.files}, f)