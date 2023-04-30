import json
import os
from flickr_crawler import FlickrCrawler

# os.environ['http_proxy'] = 'http://172.23.192.1:7890'
# os.environ['https_proxy'] = 'http://172.23.192.1:7890'

api_key = '00258f9effd82b57b188e8e27de18f1f'
api_secret = '8d9d1c075f25d892'

yfcc15m_list = '/home/dji/IAA/data/YFCC15M/yfcc100m_subset_data.tsv'
yfcc15m_shuffle_list = '/home/dji/IAA/data/YFCC15M/yfcc100m_subset_data_shuffle.txt'
yfcc15m_id = []
# with open(yfcc1m_list, 'r') as f:
#     for line in f:
#         id = line.strip().split()[1]
#         yfcc1m_id.append(id)
# random.shuffle(yfcc1m_id)
# with open(yfcc1m_list.replace('.tsv', '_shuffle.txt'), 'w') as f:
#     for id in yfcc1m_id:
#         f.write(id+'\n')

with open(yfcc15m_shuffle_list, 'r') as f:
    for line in f:
        yfcc15m_id.append(line.strip())
        # if len(yfcc15m_id) >= 2000000:
        #     break

loop_size = 1000000
crawler = FlickrCrawler(api_key=api_key, api_secret=api_secret, save_dir='/media/dji/data3/zach_data/yfcc15m', cache=True, filter_views=1000)
for i in range(loop_size, len(yfcc15m_id), loop_size):
    if i >= 2000000 and i <= 8999999:
        continue
    yfcc_sub = yfcc15m_id[i:i+loop_size]

    crawler.do_crawler(yfcc_sub, save_img=True, thread_num=20, batch_size=10000, save_json=f'annos_{i}_{i+loop_size-1}_filtered.json', continued=True)
    print(f'Great great work, done of {i} to {i+loop_size} !!!')

