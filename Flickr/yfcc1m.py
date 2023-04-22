import flickrapi
import json
import os
import pickle
from django.core.cache import cache
import urllib.request
import urllib.error
import random
from concurrent.futures import ThreadPoolExecutor

api_key = '00258f9effd82b57b188e8e27de18f1f'
api_secret = '8d9d1c075f25d892'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json', cache=True)
# flickr.cache = cache

# extras = ['description', 'license', 'owner_name', 'targs', 'machine_tags', 'views', 'media']
# extras = 'description, license, owner_name, targs, machine_tags, views, media, geo, url_c'
# r = flickr.interestingness.getList(date='2022-04-11', extras=extras, per_page='500', page=1)
# r_json = json.loads(r.decode('utf-8'))
# raw_info_list = r_json['photos']

base_dir = '/media/dji/新加卷/zach_data/'
base_len = len(base_dir)

def abs2rela(abs_path: str):
    return abs_path[base_len:]

def save_meta(save_path:str, meta:dict):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, 'wb') as f:
        pickle.dump(meta, f)

def get_photo_info(id, retry=3):
    if retry <= 0:
        print('continue in ', id)
        return None
    try:
        info_raw = flickr.photos.getInfo(photo_id=id)
        info = json.loads(info_raw.decode('utf-8'))
        if info['stat'] != 'ok':
            print('[FAIL] when getInfo get ', info)
            print('continue in ', id)
            return None
        photo_info = info['photo']

        fav_raw = flickr.photos.getFavorites(photo_id=id, page=1, per_page=1)
        fav = json.loads(fav_raw.decode('utf-8'))
        if fav['stat'] != 'ok':
            print('[FAIL] when getFavorites get ', info)
            print('continue in ', id)
            return None
        photo_info.update({'favorites': fav['photo']['total']})
    except flickrapi.FlickrError as e:
        print('[except] when getInfo/Favorites get except: ', e)
        print(f'Retry {retry} times more')
        return get_photo_info(id, retry-1)

    return photo_info

def save_rgb(rgb_link:str, rgb_file:str, timeout=10, retry=3):
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
        save_rgb(rgb_link, rgb_file, retry=retry-1)
    except urllib.error.URLError as e:
        print('[except] when save rgb get except: ', e)
        save_rgb(rgb_link, rgb_file, retry=retry-1)
    except IOError as e:
        print('[except] when save rgb get except: ', e)
        save_rgb(rgb_link, rgb_file, retry=0)

    return True


files = {}
def get_metadata(id_list:list):
    for id in id_list:
        photo_info = get_photo_info(id)
        if photo_info is None:
            continue

        server = photo_info['server']
        id = photo_info['id']
        secret = photo_info['secret']
        taken_year = photo_info['dates']['taken'][:4] # TODO: just str year?
        rgb_file = f'{base_dir}yfcc1m/rgb/{taken_year}/{id}_{secret}_b.jpg'
        meta_file = f'{base_dir}yfcc1m/meta/{taken_year}/{server}_{id}_{secret}.pkl'
        url = f'https://live.staticflickr.com/{server}/{id}_{secret}_b.jpg'
        photo_info.update({'rgb': abs2rela(rgb_file)})
        photo_info.update({'url_b': url})

        if save_rgb(url, rgb_file) is None:
            continue
        save_meta(meta_file, photo_info)

        # with open(f'{base_dir}yfcc1m/rgb/{taken_year}/annos_all_year.txt', 'a') as f:
        #     f.write(abs2rela(meta_file) + '\n')
        if taken_year not in files.keys():
            files[taken_year] = []
        files[taken_year].append(abs2rela(meta_file))
    print('great work! 10000 done!')
    with open('done_id.txt', 'a') as f:
        for id in id_list:
            f.write(id + '\n')



yfcc1m_list = '/home/dji/IAA/data/YFCC15M/yfcc100m_subset_data.tsv'
yfcc1m_shuffle_list = '/home/dji/IAA/data/YFCC15M/yfcc100m_subset_data_shuffle.txt'
yfcc1m_id = []
# with open(yfcc1m_list, 'r') as f:
#     for line in f:
#         id = line.strip().split()[1]
#         yfcc1m_id.append(id)
# random.shuffle(yfcc1m_id)
# with open(yfcc1m_list.replace('.tsv', '_shuffle.txt'), 'w') as f:
#     for id in yfcc1m_id:
#         f.write(id+'\n')

with open(yfcc1m_shuffle_list, 'r') as f:
    for line in f:
        id = line.strip()
        yfcc1m_id.append(id)
        if len(yfcc1m_id) >= 2000000:
            break

for sub_i in range(2):
    subset = yfcc1m_id[sub_i*1000000:(sub_i+1)*1000000]
    with ThreadPoolExecutor(max_workers=20) as pool:
        for subsub_i in range(100):
            pool.submit(get_metadata, subset[subsub_i*10000:(subsub_i+1)*10000])

    print('great great work!!! 1M done!!!')





with open('yfcc2m_annos_all.json', 'w') as f:
    print('all year: ', len(files))
    json.dump({'files': files}, f)
