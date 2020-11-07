# ==========================================================================================
# Arwa Ashi - HW4 - Week9 - Nov 7th, 2020
# Saudi Digital Academy
# ==========================================================================================
# Stock Ticker with Dash.
# ------------------------------------------------------------------------------------------
# Scrapping stock prices code !

# Packages
# ------------------------------------------------------------------------------------------
from bs4 import BeautifulSoup
import requests
import pandas as pd
import xlwt
import xlsxwriter


# Calling the website HTML  >>Aramco<<
# ------------------------------------------------------------------------------------------
url      = "https://uk.finance.yahoo.com/quote/2222.SR/history?p=2222.SR"
response = requests.get(url)
html     = response.content
scraped  = BeautifulSoup(html ,'html.parser')
#print(scraped)


# Call the books' titles and prices
results = []

trs = scraped.find_all("tr", class_="BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)")
#print(trs)

for tr in trs:
    date  = tr.find("td",class_="Py(10px) Ta(start) Pend(10px)").select_one("[class*='Py(10px) Ta(start) Pend(10px)'] span").text
    price = tr.select_one("[class*='Py(10px) Pstart(10px)'] span").text
    results.append({"dates":date,"prices":price,"companies":"Saudi Arabian Oil Company (2222.SR)"})
# print(results)

df_01 = pd.DataFrame.from_dict(data=results)
#print(df_01)

# Calling the website HTML  >>STC<<
# ------------------------------------------------------------------------------------------
url      = "https://uk.finance.yahoo.com/quote/7010.SR/history?p=7010.SR"
response = requests.get(url)
html     = response.content
scraped  = BeautifulSoup(html ,'html.parser')
#print(scraped)


# Call the books' titles and prices
results = []

trs = scraped.find_all("tr", class_="BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)")
#print(trs)

for tr in trs:
    date  = tr.find("td",class_="Py(10px) Ta(start) Pend(10px)").select_one("[class*='Py(10px) Ta(start) Pend(10px)'] span").text
    price = tr.select_one("[class*='Py(10px) Pstart(10px)'] span").text
    results.append({"dates":date,"prices":price,"companies":"Saudi Telecom Company (7010.SR)"})
# print(results)

df_02 = pd.DataFrame.from_dict(data=results)
# print(df_02)

# Calling the website HTML  >>Alinma Bank (1150.SR)<<
# ------------------------------------------------------------------------------------------
#url      = "https://uk.finance.yahoo.com/quote/2222.SR/history?p=2222.SR"
url      = "https://uk.finance.yahoo.com/quote/1150.SR/history?p=1150.SR"
response = requests.get(url)
html     = response.content
scraped  = BeautifulSoup(html ,'html.parser')
#print(scraped)


# Call the books' titles and prices
results = []

trs = scraped.find_all("tr", class_="BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)")
#print(trs)

for tr in trs:
    date  = tr.find("td",class_="Py(10px) Ta(start) Pend(10px)").select_one("[class*='Py(10px) Ta(start) Pend(10px)'] span").text
    price = tr.select_one("[class*='Py(10px) Pstart(10px)'] span").text
    results.append({"dates":date,"prices":price,"companies":"Alinma Bank (1150.SR)"})
# print(results)

df_03 = pd.DataFrame.from_dict(data=results)
#print(df_03)

df_04 = df_01.append(df_02, ignore_index = True)
df_05 = df_04.append(df_03, ignore_index = True)
#print(df_05)

# exporting into excel
scraping_writer = pd.ExcelWriter('stock_prices.xlsx',engine='xlsxwriter')
df_05.to_excel(scraping_writer, sheet_name='Sheet1')
scraping_writer.save()

