import numpy as np
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse

# given the name of the tubestation, output a string that is part of the url of the wikipedia page
def process_name(line, extension, note_at_end=True):
    #--------------------------------------------------
    station = line.replace("\n", "").replace(" ", "_");
    #--------------------------------------------------
    if (line.find('(') == -1):
        search_str = station + "_" + extension;
    else:
        note = station[station.find("(") : station.find(")")+1];
        if (note_at_end):
            search_str = station[0:station.find("(")] + extension + "_" + note;
        else:
            search_str = station[0:station.find("(")] + note + "_" + extension;

    return search_str


if __name__ == "__main__":
    url = 'https://en.wikipedia.org/wiki/'
    headers = {'User-Agent' : "Chrome 64.0.3282.186"}
    
    coords = [];
    f = open("station.names");
    for line in f.readlines():
        # the wiki page could be under different names, therefore try different possibilities
        try:
            req = urllib.request.Request(url + process_name(line, "_tube_station", True), headers=headers)
            dat = urllib.request.urlopen(req).read()
        except:
            try:
                req = urllib.request.Request(url + process_name(line, "_DLR_station", True), headers=headers)
                dat = urllib.request.urlopen(req).read()
            except:
                try:
                    req = urllib.request.Request(url + process_name(line, "_station", True), headers=headers)
                    dat = urllib.request.urlopen(req).read()
                except:
                    try:
                        req = urllib.request.Request(url + process_name(line, "_tube_station", False), headers=headers)
                        dat = urllib.request.urlopen(req).read()
                    except:
                        try:
                            req = urllib.request.Request(url + process_name(line, "_DLR_station", False), headers=headers)
                            dat = urllib.request.urlopen(req).read()
                        except:
                            try:
                                req = urllib.request.Request(url + process_name(line, "_station", False), headers=headers)
                                dat = urllib.request.urlopen(req).read()
                            except:
                                raise Exception("can not find a wikipedia page for " + line)
    
        # create a BeautifulSoup object, with the html file
        soup = BeautifulSoup(dat, "lxml")

        # the keywords we are looking for is "geo" 
        coord = soup.find_all("span", class_="geo")

        if (len(coord) == 0):
            raise Exception("can not find coordinates for " + line)
        else:
            coord0 = np.array(coord[0].get_text().split(';'), dtype=float)
            coord1 = np.array(coord[1].get_text().split(';'), dtype=float)
            if (np.allclose(coord0, coord1)):
                coords.append(coord0)
                print(coord0)
            else:
                raise Exception("inconsistent coordinates found for " + line + str(coord))

    np.savetxt("station.coords", np.array(coords))
