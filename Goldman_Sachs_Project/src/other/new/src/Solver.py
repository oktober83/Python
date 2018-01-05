"""
Scrape a table from wikipedia using python. Allows for cells spanning multiple rows and/or columns. Outputs csv files for
each table
"""

from bs4 import BeautifulSoup
import wikipedia
import urllib2
import os
import codecs
wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
req = urllib2.Request(wiki,headers=header)
page = urllib2.urlopen(req)
soup = BeautifulSoup(page,'html.parser')

print soup.prettify()

table = soup.find("table", { "class" : "wikitable" })

# preinit list
rows = table.findAll("tr")
row_lengths = [len(r.findAll(['th','td'])) for r in rows]
ncols = max(row_lengths)
nrows = len(rows)
data = []
links = []
for i in range(nrows):
    rowD = []
    links.append('')
    for j in range(ncols+1):
        rowD.append('')
    data.append(rowD)

# process html
for i in range(nrows):
    row = rows[i]
    rowD = []
    cells = row.findAll(["td","th"])
    for link in cells[1].find_all('a'):
        links[i] = link.get('href')
    for j in range(len(cells)):
        cell = cells[j]

        #lots of cells span cols and rows so lets deal with that
        cspan = int(cell.get('colspan',1))
        rspan = int(cell.get('rowspan',1))
        for k in range(rspan):
            for l in range(cspan):
                data[i+k][j+l] += cell.text
        data[i+k][ncols] = links[i]
    data.append(rowD)

# write data out

page = os.path.split(wiki)[1]
fname = 'output_{}.csv'.format(page)
f = codecs.open(fname, 'w',encoding='utf-8')
for i in range(nrows):
    rowStr = ','.join(data[i])
    rowStr = rowStr.replace('\n','')
    print rowStr
    rowStr = rowStr.encode('unicode_escape')
    f.write(rowStr+'\n')


f.close()


#Begin matching data from company summaries

for i in xrange(1,nrows-2):
    wiki = "https://en.wikipedia.org" + data[i][8]
    req = urllib2.Request(wiki, headers=header)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    summary = soup.find("p")
    print summary

