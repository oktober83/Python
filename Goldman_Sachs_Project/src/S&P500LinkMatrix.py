"""
Using Python 2.7
Install the following modules using these command in command terminal
Please contact Deep Patel if you run into any problems!

sudo easy_install -U sqlitedict
sudo pip install bs4
sudo pip install urllib2
sudo pip install numpy
sudo pip install time
sudo pip install logging
sudo pip install nltk
sudo pip install gensim

For nltk.download():
#Opens GUI: Go to Corpus tab and download "stopwords" only

"""
from bs4 import BeautifulSoup
import urllib2
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import nltk
from nltk.corpus import stopwords
# nltk.download() # Go to Corpus tab and download "stopwords" only
from gensim import corpora, models, similarities
t1 = time.time()


# Company class creates Company object using data list to initialize variables
# Used in getLinkMatrix method internally
class Company(object):

    def __init__(self,row):

        self.ticker = row[0]
        self.name = row[1]
        self.sector = row[3]
        self.subIndustry = row[4]
        self.loc = row[5]
        self.hasWebsite = False
        self.webEnd = ''
        self.website = ''
        if row[8] != '' and "redlink=1"not in row[8]:
            self.hasWebsite = True
            self.webEnd = row[8][6:]
            self.website = "http://www.wikipedia.org/wiki/"+self.webEnd

    # Returns semiprocessed text of Wikipedia article as string if the company has a website
    # If not, returns an empty string
    def getWikiData(self):
        if self.hasWebsite:
            wiki = self.website
            header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
            req = urllib2.Request(wiki, headers=header)
            page = urllib2.urlopen(req)
            soup = BeautifulSoup(page, 'html.parser')
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out
            # get text
            text = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            end = text.find("Navigation menu")
            text = text[:end]
            return text
        else:
            return ""
    # This is method is used if a company does not have a website
    # It returns an integer multiplier that is less than or greater than 1 by a certain amount based on the
    # similarity of the 2 companys' sector and subindustry
    def CompLink(self,compB):
        multiplier = 1
        if self.sector == compB.sector:
            multiplier = multiplier*1.2
            if self.subIndustry == compB.subIndustry:
                multiplier = multiplier*1.3
        else:
            multiplier = multiplier*0.8

        return multiplier



# Method that contains algorithm that calculates the link matrix
# Returns link matrix
def getLinkMatrix():
    #Scrapes the S&P 500 table of companies & adds the link to the company as an additional column
    wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
    req = urllib2.Request(wiki,headers=header)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page,'html.parser')
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

            # Some cells span cols and rows so this deals with that
            cspan = int(cell.get('colspan',1))
            rspan = int(cell.get('rowspan',1))
            for k in range(rspan):
                for l in range(cspan):
                    data[i+k][j+l] += cell.text
            data[i+k][ncols] = links[i]

    # Removes headings column
    data.pop(0)
    # Removes extra classes of companies with multiple classes using CIK number to check if 2 stocks are for the same company
    # Keeps first class in the list and keeps the name of the company the same, so you may ignore the class in the name of the company
    ptr = 0;
    ptr2 = 1;
    while len(data) != 500:
        if data[ptr][7] == data[ptr2][7]:
            data.pop(ptr2)
        else:
            ptr+=1
            ptr2+=1

    # Makes a list of 500 Company objects using rows in data
    list = [None]*len(data)
    for i in range(len(data)):
        list[i] = Company(data[i])
    # Delete data variable to free up memory (not needed anymore)
    del data

    companyNames = [None] * len(list)
    for i in range(0, len(list)):
        companyNames[i] = str(i) + ' ' + list[i].name
    f = open('companyNames.txt', 'w')
    for ele in companyNames:
        f.write(ele + '\n')
    f.close()

    # Creates list of documents containing Wikipedia pages for those that have one or an empty string for those that do not
    documents = []
    for i in range(0,len(list)):
        documents.append(list[i].getWikiData())

    # Logs activity of Gensim
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Get a list of stopwords from NLTK and adds some of our own that apply to this situation
    stoplist = set(stopwords.words('english')+u'company industry wikipedia the free encyclopedia jump to navigation search article traded history external links notes references'.split())

    # remove common words and tokenize
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1 and token.isalpha()]
             for text in texts]

    dictionary = corpora.Dictionary(texts)                  # creates dictionary of words
    corpus = [dictionary.doc2bow(text) for text in texts]   # creates the corpus

    # Creates a zero matrix (2D array) of size 500 by 500
    matrix = np.zeros((len(list),len(list)))

    # Creates a model for the corpus of Wiki documents using Latent Semantic Indexing with
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=70)

    """
    Iterates through each row of matrix and gets a row of link values
    Sets all links on the main diagonal (Company A compared to Company A) equal to 0
    The document of a company without a Wikipedia page is an empty string so its links are 0 everywhere in the matrix
    To get a link for these companies, the row of links is determined using the CompLink method in the Company class
    The column of those companies is determined in the same way but is made relevent to the other links in the row by
    multiplying it by the mean of that row.
    For each row of links, the whole row is divided by the sum of that row, which automatically normalizes the row of links
    to have a sum of 1.
    """

    for i in range(0,len(matrix)):
        doc = texts[i]
        vec_bow = dictionary.doc2bow(doc)
        vec_lsi = lsi[vec_bow] # convert the query to LSI space
        index = similarities.MatrixSimilarity(lsi[corpus],num_features=lsi.num_topics)
        matrix[i] = index[vec_lsi] # perform a similarity query against the corpus and assign result to matrix[i]
        # If row is of a company without a Wikipedia page
        if list[i].hasWebsite is False:
            for j in range(0, len(matrix[0])):
                # Gets a multiplier for each column using the CompLink method and assigns it.
                # Multiplier is basically multiplied by 1. It does not matter the row will be normalized at the end of this row loop
                matrix[i][j] = list[i].CompLink(list[j])
            # Overwrites link of company compared to itself to equal 0
            matrix[i][i] = 0
        # if row is a company with a Wikipedia page
        else:
            # Overwrites link of company compared to itself to equal 0
            matrix[i][i] = 0
            # For every column in the row
            for j in range(0, len(matrix[0])):
                # If the company in a column does not have a Wikipedia page
                if list[j].hasWebsite is False:
                    # Gets a multiplier using the CompLink method and multiplies it by the mean of the matrix row
                    matrix[i][j] = list[i].CompLink(list[j]) * matrix[i].mean()
                # if the link is below 0, set the link to zero
                elif matrix[i][j] < 0:
                    matrix[i][j] = 0

        sum = matrix[i].sum()       # Gets sum of row
        matrix[i] = matrix[i]/sum   # Divides row by sum

    # Return the final matrix of links
    return matrix

linkMatrix = getLinkMatrix()
np.savetxt("LinkMatrix.csv", linkMatrix, delimiter=",")
t2 = time.time()

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=10000)



plt.imshow(linkMatrix, cmap='hot', interpolation='nearest')
plt.show()


for i in range(0,len(linkMatrix)):
    print linkMatrix[i]
print "Total time taken:", t2 - t1, "sec"

