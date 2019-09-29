import time

import requests
import wget
from bs4 import BeautifulSoup

baseUrl = 'http://expressexpense.com/view-receipts.php?page='

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0'}

failed = []
for i in range(1, 91):
    url = baseUrl + str(i)
    response = requests.get(url, headers=headers)
    time.sleep(1)
    soup = BeautifulSoup(response.text, 'lxml')
    links = soup.find_all('div', class_='record')
    print("page no", i)
    print("\n")
    j = 1
    for link in links:
        # time.sleep(1)
        print(j)
        print("\n")
        try:
            if link:
                print(link.img['src'])
                print("\n")
                wget.download(link.img['src'])


            else:
                failed.append({i, j, link.img['src']})
        except Exception as e:
            print("Error occured", e)
            failed.append({i, j})
        j = j + 1

print(failed)
with open("failed-log", "w") as outputfile:
    outputfile.write("\n ".join(failed))
