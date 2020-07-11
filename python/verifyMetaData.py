# ============================================================ #
# Author:       S e a B a s s
# Create Date:  6 June 2020
# 
# Project:      Lux
# Filename:     verifyMetaData.py
# 
# Description:  Refresh/populate the server metadataa files by
#               extracting information from database websites.
# 
# ============================================================ #
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve, Request
from urllib.error import HTTPError 
import googlesearch
import os
import re

# Minimum required metadata
REQUIRED_METADATA = [
    "title",
    "studio",
    "yearStart",
    "iconAddr",
    "description"
]

DATA_DIR = "./"
# DATA_DIR = "../lux-backend/" # Debug environment

DATA_DIR_MANGA = DATA_DIR+"manga/"
DATA_DIR_VIDEOS = DATA_DIR+"videos/"
DATA_DIR_MUSIC = DATA_DIR+"music/"
DATA_DIR_IMAGES = DATA_DIR+"images/"

THUMBNAIL_CACHE = "images/thumbnails/"
THUMBNAIL_CACHE_VIDEOS = THUMBNAIL_CACHE + "videos/"
THUMBNAIL_CACHE_MUSIC = THUMBNAIL_CACHE + "music/"
THUMBNAIL_CACHE_MANGA = THUMBNAIL_CACHE + "manga/"

USER_AGENT = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

def determineMissingRequiredParams(metaData: dict):
    """Check that the requried parameters are stored in the metadata."""
    
    # Search for missing parameters
    missingParams = []
    for key in REQUIRED_METADATA:
        if key not in metaData.keys() or metaData[key] == "": missingParams.append(key)
    
    # Check that the thumbnail is on file
    if "iconAddr" in metaData.keys():
        iconAddr = metaData["iconAddr"]
        if not os.path.exists(DATA_DIR+iconAddr):
            metaData["iconAddr"] = ""
            missingParams.append("iconAddr")

    return missingParams

def fmtCheck(parameter: str, parameterType: str):
    """
        Helper function: Does a check on the given parameter for formating issues

        Inputs:
            parameter - The parameter string itself
            parameterType - String describing the parameter class, TODO: Could be an enum.
    """

    if parameterType == 'title':
        return re.match(r'[\w :;\\,\.\-]+', parameter)
    elif parameterType == 'titleAlt':
        return re.match(r'[一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[ａ-ｚＡ-Ｚ０-９]+|[々〆〤]+', parameter)
    elif parameterType == 'animator':
        return re.match(r'[\w ,\.\-]+', parameter)
    elif parameterType == 'tags':
        return re.match(r'[\w ,\-]+', parameter)
    elif parameterType == 'numberOfEpisodes':
        return re.match(r'\d+', parameter)
    elif parameterType == 'date':
        return re.match(r'\d\d\.\d\d\.\d\d\d\d', parameter)
    elif parameterType == 'description':
        return 0 < len(parameter)
    else:
        return True

def getSoup(url: str):
    """
        Gets a accessable datastructure for the given webpage. 
        Returns None on failure.
    """

    try:
        req = Request(url, headers={'User-Agent':USER_AGENT})
        html = urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        return soup
    except HTTPError:
        print(f"HTTP connection was denied to '{url}'.")
    except:
        print(f"Unknown Error in accessing '{url}'.")

    return None

def extractAniDB(url: str):
    """
        Extract the information from an aniDB webpage. 
        Takes the url of the page as an arguement.
    """
    
    metaData = {}
    
    print(f"Grabbing data from AniDB: '{url}'")
    soup = getSoup(url)
    if soup:
        metaData["visited_AniDB"] = True

        # Extract what we need
        e0 = soup.find_all("label", {"itemprop": "alternateName"})
        if e0 and 1 < len(e0): 
            txt = e0[1].get_text()
            if fmtCheck(txt, 'titleAlt'): metaData['title-jp'] = txt
        
        e0 = soup.find("table", {"class":"staff"})
        if e0:
            e1 = e0.find_all("a")
            if e1 and 0 < len(e1):
                txt = e1[-1].get_text()
                if fmtCheck(txt, 'studio'): metaData['staff'] = txt
        
        e0 = soup.find("tr", {"class": "tags"})
        if e0:
            e1 = e0.find_all("span", {"itemprop": "genre"})
            if e1:
                txt = ", ".join( [span.get_text() for span in e1] )
                if fmtCheck(txt, 'tags'): metaData['tags'] = txt
        
        e0 = soup.find("span", {"itemprop": "numberOfEpisodes"})
        if e0:
            txt = e0.get_text()
            if fmtCheck(txt, 'numberOfEpisodes'): metaData['numEpisodes'] = txt
        
        e0 = soup.find("span", {"itemprop": "startDate"})
        if e0:
            startDate = e0.get_text()
            if startDate != "?":
                if fmtCheck(startDate, 'date'): 
                    [startDay, startMonth, startYear] = startDate.split(".")
                    metaData['dayStart'] = startDay
                    metaData['monthStart'] = startMonth
                    metaData['yearStart'] = startYear
        
        e0 = soup.find("span", {"itemprop": "endDate"})
        if e0:
            endDate = e0.get_text()
            if endDate != "?":
                if fmtCheck(endDate, 'date'): 
                    [endDay, endMonth, endYear] = endDate.split(".")
                    metaData['dayEnd'] = endDay
                    metaData['monthEnd'] = endMonth
                    metaData['yearEnd'] = endYear

        e0 = soup.find("div", {"itemprop": "description"})
        if e0:
            txt = e0.get_text().replace("\n", "---") # NOTE: We replace newlines with --- since its going in an ini file
            if fmtCheck(txt, 'description'): metaData['description'] = txt

        # Cache the image for thumbnails
        # NOTE title is required for caching, if its not on this page maybe dont save the image
        if 'title' in metaData.keys():
            title = metaData['title']
            e0 = soup.find("img", {"itemprop":"image"})
            if e0:
                icon_url = e0['src']
                
                # Cache the data
                if icon_url:
                    local_cache_addr = DATA_DIR+THUMBNAIL_CACHE_VIDEOS
                    if not os.path.exists(local_cache_addr): os.makedirs(local_cache_addr)

                    try:
                        urlretrieve(icon_url, local_cache_addr+title+".png")

                        # Save the location of the icon
                        metaData['iconAddr'] = THUMBNAIL_CACHE_VIDEOS+title+".png"

                    except HTTPError:
                        print(f"HTTP connection was denied to '{icon_url}'.")
                    except:
                        print(f"Unknown Error in accessing '{icon_url}'.")

    # print(metaData)
    return metaData

def loadMetaData(address: str):
    
    metaData = {}

    metaDataAddr = address + "/info.meta"
    if os.path.exists(metaDataAddr):
        with open(metaDataAddr, mode='r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')

            for line in lines:
                if line != "":

                    cols = line.split("=")

                    key = cols[0]
                    val = "=".join(cols[1:])

                    metaData[key] = val

    return metaData

def storeMetaData(address: str, metaData: dict):
    """
        Formats a 'metaData' dictionary as an ini file and stores it in 'address'
    """

    if not os.path.exists(address): os.makedirs(address)

    strOut = ""
    for key, val in metaData.items():
        strOut += f"{key}={val}\n"

    with open(address+"/info.meta", mode='w', encoding="utf-8") as fp:
        fp.write(strOut)

def verifyVideoData():
    """
        Loop through the video metadata files and verify that the necessary parameters are present.
        If they are not extract the information from the web.
    """
    
    for show_dir in os.listdir(DATA_DIR_VIDEOS):
        address = DATA_DIR_VIDEOS+show_dir

        # initialize the meta data file
        metaData = loadMetaData(address)

        # This is localz
        metaData['title'] = show_dir

        # Check that we got enough data
        missingParams = determineMissingRequiredParams(metaData)
        if 0 < len(missingParams):
            print(show_dir)
            print(f"WARNING! Cache is missing: [{', '.join(missingParams)}].")

            # Read from AniDB
            if 'visited_AniDB' not in metaData.keys():
                for url in googlesearch.search(show_dir + " anidb", num=1, stop=1, pause=1):

                    if re.match(r"https:\/\/anidb\.net\/anime\/\d+", url):
                        extractedData = extractAniDB(url)
                        if extractedData: 
                            metaData.update(extractedData)
                    else:
                        print(f"\tWarning. Unsure about anidb link: '{url}'. Skipping.")

            # Check that we got enough data
            # print(metaData)
            missingParams = determineMissingRequiredParams(metaData)
            if 0 < len(missingParams):
                print(f"WARNING! AniDB is missing: [{', '.join(missingParams)}].")

                # TODO: Read from MAL next

        # Stash the metadata retrieved
        storeMetaData(address, metaData)

if __name__ == "__main__":
    verifyVideoData()