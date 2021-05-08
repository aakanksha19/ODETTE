import os
import re
import random
import math
import xml.etree.ElementTree as ET
from nltk import sent_tokenize, word_tokenize

root_dir = "../timebank_1_2/data/timeml/"
out_dir = "../processed_timebank/"

all_files = []
for file in os.listdir(root_dir):

    all_files.append(file.split(".tml")[0])

    tree = ET.parse(open(os.path.join(root_dir, file), "r"))
    root = tree.getroot()

    events_to_keep = []
    for tag in root.findall('MAKEINSTANCE'):
        if tag.get('polarity') == "NEG" or tag.get("tense") == "FUTURE":
            continue
        events_to_keep.append(tag.get('eventID'))

    # Mark the events that we want to keep
    doc_text = ET.tostring(tree.getroot(), encoding='utf-8')
    events_tags = re.findall(re.compile("<EVENT [^/]*>[^<]*</EVENT>"), str(doc_text))
    for tag in events_tags:
        event_id = tag.split("eid=\"")[1].split('"')[0]
        if event_id not in events_to_keep:
            continue
        doc_text = str(doc_text).replace(tag, "{}_{}".format(event_id, tag.split(">")[1].split("</")[0]))

    # Strip out rest of the XML tags
    doc_text = re.sub("<[^<]+>", "", str(doc_text))

    # Additional document cleaning
    doc_text = doc_text.replace(file.split(".tml")[0], "")
    doc_text = doc_text.replace("\\n", " ")
    doc_text = doc_text.replace("\\'", "'")
    if "NEWS STORY" in doc_text:
        doc_text = doc_text.replace("NEWS STORY", "")
    if "WALL STREET JOURNAL (J)" in doc_text:
        doc_text = doc_text.split("WALL STREET JOURNAL (J) ")[-1]
    if "Cx11" in doc_text:
         doc_text = doc_text.split("Cx11")[-1]
    # found = re.findall("BC-[\w]+", str(doc_text))
    # if found:
    #     print(doc_text.split(found[-1]))
    doc_text = " ".join(doc_text.split())

    out_file = open(os.path.join(out_dir, file.split('.tml')[0]+'.tsv'), "w")
    sents = sent_tokenize(doc_text)
    for i, sent in enumerate(sents):
        words = word_tokenize(sent)
        if words[0] == 'b':
            words = words[1:]
        if i > 0:
            out_file.write("\n")
        for word in words:
            found = re.findall("e[0-9]+_", word)
            if found:
                word = word.split("_")[1]
                # TODO: Check why this condition is needed...how are some event words empty
                if word == "":
                    continue
                out_file.write("{}\t{}\n".format(word, "EVENT"))
            else:
                out_file.write("{}\t{}\n".format(word, "O"))
    out_file.close()

random.shuffle(all_files)
num_files = len(all_files)
train_end = math.ceil(0.6*num_files)
dev_end = train_end + math.ceil(0.1*num_files)
train_files = all_files[:train_end]
dev_files = all_files[train_end:dev_end]
test_files = all_files[dev_end:]

out_file = open("train_ids.txt", "w")
for line in train_files:
    out_file.write(line+"\n")
out_file.close()
out_file = open("dev_ids.txt", "w")
for line in dev_files:
    out_file.write(line+"\n")
out_file.close()
out_file = open("test_ids.txt", "w")
for line in test_files:
    out_file.write(line+"\n")
out_file.close()