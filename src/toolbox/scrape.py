import os
import re
import shutil
import time

import requests
from bs4 import BeautifulSoup

'''
scrape images under a tag of flicker.
'''

BASE_URL = 'https://www.flickr.com/photos/tags/night/'
DOMAIN = 'https://www.flickr.com/'
LOCAL_DIR = '/Users/why/Desktop/scrape'
LOCAL_FILES = os.listdir(LOCAL_DIR)
PAGE_BEGIN_ID = 12
PAGE_END_ID = 41


def sleeped_request(url, stream=False):
    response = ''
    while response == '':
        try:
            response = requests.get(url, stream=stream)
            return response
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("Zzzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue


def download_file(dirpath, url):
    fname = url.split('/')[-1]
    if fname in LOCAL_FILES:
        print(f'Already exist: {fname}')
        return

    local_filename = os.path.join(dirpath, fname)
    with sleeped_request(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f'File downloaded: {url} -> {local_filename}')
    return local_filename


def get_one_img(size_page_link):
    response = sleeped_request(size_page_link)
    soup = BeautifulSoup(response.text, 'lxml')
    img_src = soup.find('img', src=re.compile('https://live.staticflickr.com/.*'))['src']

    # while RUNNING_CMDS_NUM <= MAX_RUNNING_CMDS_NUM:
    # os.spawnl(os.P_DETACH, f'wget {img_src}')
    download_file(LOCAL_DIR, img_src)

    # ipdb.set_trace()


def run_one_page(soup, page_id):
    imgs_hrefs = soup.find_all('a', href=re.compile('^/photos/*'), class_='overlay')
    print(f'[*] Found [{len(imgs_hrefs)}] images in page [{page_id}]')
    for x in imgs_hrefs:
        # 优先爬1024p图片：
        size_page_link = DOMAIN + x['href'] + 'sizes/l/'
        get_one_img(size_page_link)


def main():
    for i in range(PAGE_BEGIN_ID, PAGE_END_ID):
        url = BASE_URL + f'page{i}'
        # headers = {
        #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        # response = requests.get(url, headers=headers)
        response = sleeped_request(url)

        content = response.text
        soup = BeautifulSoup(content, 'lxml')

        # ipdb.set_trace()
        run_one_page(soup, i)
        # break


if __name__ == "__main__":
    main()
