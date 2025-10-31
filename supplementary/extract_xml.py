import os
import pandas as pd
from lxml import etree

fp = "/path/to/118/congress/data/118/bills/hr/"
subfolders = os.listdir(fp)

def parse_text(fp):
    agg = []
    sponsors = []
    cosponsors = []
    tree = etree.parse(fp)
    root = tree.getroot()
    text_nodes = root.xpath(".//text")
    for text_node in text_nodes:
        text_content = text_node.text
        if text_content.startswith("<!"):
            continue
        agg.append(text_content)
    try:
        sponsor_node = root.xpath(".//sponsors")[0]
    
        items = sponsor_node.xpath("./item/fullName")
        for item in items:
            sponsors.append(item.text)
    except:
        pass
    try:
        cosponsor_node = root.xpath(".//cosponsors")[0]
        items = cosponsor_node.xpath("./item/fullName")
        for item in items:
            cosponsors.append(item.text)
    except:
        pass
    
    return agg, sponsors, cosponsors

count = 0
agg = []
for subfolder in subfolders:
    if count >= 1000:
        break
    fp2 = fp + subfolder + "/fdsys_billstatus.xml"
    text, sponsor, cosponsor = parse_text(fp2)
    agg.append((text, sponsor, cosponsor))
    count += 1

all_text = [x[0] for x in agg]
all_sponsors = [x[1] for x in agg]
all_cosponsors = [x[2] for x in agg]

edges = []
for sponsor_list, cosponsor_list in zip(all_sponsors, all_cosponsors):
    for sponsor in sponsor_list:
        for cosponsor in cosponsor_list:
            edges.append({'sponsor': sponsor, 'cosponsor': cosponsor})

df = pd.DataFrame(edges)

edge_counts = df.groupby(['sponsor', 'cosponsor']).size().reset_index(name='count')

json.dump(all_text, open("./bills_text.json", "w"))
edge_counts.to_csv("./bills_edges.csv", sep="\t")