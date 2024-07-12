import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Text2TextGenerationPipeline,AutoModelForTokenClassification, pipeline
from conllu import parse, parse_tree, parse_incr
from termcolor import colored
import numpy as np
import json
from tqdm import tqdm



# Download the WordNet data if you haven't already
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def get_supersenses(word):
    # Get all synsets for the word
    synsets = wn.synsets(word)
    l_def_and_supersenses = []
    # Loop through each synset and print its supersense information
    for synset in synsets:
        definition = synset.definition()
        supersense = synset.lexname()
        if "noun." in supersense:
            l_def_and_supersenses.append((definition,supersense))
    return l_def_and_supersenses





def get_in_context_supersense(word,context,pipe):

    l_def_and_supersenses = get_supersenses(word)
    set_categ = set(dict(l_def_and_supersenses).values())
    inter = set_categ & {'noun.animal','noun.person'} 
    if inter == set():
        return ("I")
    elif set_categ.issubset({'noun.animal','noun.person'}):
        return("A")
    else:
        question = '''question: which description describes the word "'''+word+'''" best in the following context? \n'''
        all_desript = ''''''

        for desc,s in l_def_and_supersenses:
            Desc = ''' " %s ", ''' %desc
            all_desript = all_desript+ Desc

        descriptions='''desription: [  %s ] \n''' %all_desript
        context = '''context: %s ''' %context
        input = question+descriptions+context
        output = pipe(input)[0]['generated_text']

        for defin,categ in l_def_and_supersenses:
            if defin[:15]==output[:15]:
                return get_label_nn(categ)
        print("no matching def found")
        #print(l_def_and_supersenses,output)

def NER_label(word,text,distilbert_out,ner_pipeline):
    
    if distilbert_out == None:
        distilbert_out = ner_pipeline(text)
        print(colored("Ner computed","green"))
    print("raw",distilbert_out)
    l_NER_conact = concetenate_NER_outputs(distilbert_out)
    print("after_concat",l_NER_conact)
    for w,label in l_NER_conact:
        if word == w:
            return label, distilbert_out
    for w,label in l_NER_conact:
        if set([*w]).issubset(set([*word])):
            return label, distilbert_out    
    print(colored("unknowed elem, set at I","red"),text,word)
    return "I", distilbert_out   





def get_dict_UD(UD_file,max_len):
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    l_dict = []
    idx = 0
    

    for elem in tqdm(dd_data_UD):
        idx +=1
        #if idx % 10 == 0:
        #    print(idx)
        if max_len >0:
            if idx >max_len:
                break
        
        text = elem.metadata['text']
        l_text_split = text.split()
        sent_len = len(l_text_split)
        
        l = list(elem)
        l_text_split_without_punct = remove_punct(l_text_split)
        distilbert_out = None
        for d_word in l:
            if "NOUN" in d_word.values() or "PROPN" in d_word.values():
                word = d_word["form"]
                if l_text_split_without_punct.count(word) == 0:
                    print(colored(word,"red"),text)
                #if l_text_split.count(noun) == 0:
                #    print(colored(noun,"blue"),text)
                if l_text_split_without_punct.count(word) == 1:
                    #pos_in_sent = d_word["id"] -> includes ponctuations
                    pos_in_sent = l_text_split_without_punct.index(word)
                    if "nsubj" in d_word.values():
                        subj = True
                    else:
                        subj = False
                    if "obj" in d_word.values():
                        obj = True
                    else: 
                        obj = False
                    if "NOUN" in d_word.values():
                        label = get_in_context_supersense(word,text,pipe)
                        pos = "noun"
                    if "PROPN" in d_word.values():
                        label,distilbert_out = NER_label(word,text,distilbert_out,ner_pipeline)
                        pos = "propn"

                    d = {"context":text,"word":word,"subj":subj,"obj":obj,"pos_in_sent":pos_in_sent,"sent_len":sent_len, "label":label, "pos":pos}
                    if label == None:
                        print(d)
                    else:
                        l_dict.append(d)
    return l_dict

def concetenate_NER_outputs(distillbert_out):
    #distillbert_out.reverse()
    l = []
    for d in distillbert_out:
        word = d["word"]
        if "PER" in d["entity"]:
                label = "A"
        else:
            label = "I"

        if word[0] == "#": #"suffix" of a proper noun
            l[-1] = (l[-1][0] + word[2:], l[-1][1]) 
            if label != l[-1][1]:
                print(colored("red","same entity diff labels"),d)
        else:
            l.append((word,label)) #start of Proper noun
    return l

def remove_punct(l_text_split):
    l_text_split_without_punct = []
    for w in l_text_split:
        if w[-1] in [".","!",",","?",":",";",")","]","\'","\""]:
            #print(colored(w,"green"))
            if w[0] in ["(","[","\"","\'"]:
                l_text_split_without_punct.append(w[1:-1])
            else:   
                l_text_split_without_punct.append(w[:-1])
        elif w[0] in ["(","[","\"","\'"]:
            
            l_text_split_without_punct.append(w[1:])
        elif "-" in w:
            l = w.split("-")
            l_text_split_without_punct = l_text_split_without_punct+l
        #elif w[-2:]=="\'s":
        #    l_text_split_without_punct = l_text_split_without_punct.append(w[:-2])
        else:
            l_text_split_without_punct.append(w)
    return l_text_split_without_punct




def get_label_nn(categ):
    """
    H: Human A: animal I: rest (inanimate)
    """
    l_categ = ['noun.animal','noun.person']
    if categ in l_categ:
        return "A"
    else:
        return "I"

#UD_file = "../subjecthood/ud-treebanks-v2.13/UD_English-GUM/en_gum-ud-dev.conllu"
UD_file = "UD_English-GUM/en_gum-ud-train.conllu"
#print(get_dict_UD(file,3))

#exit()



pipe = Text2TextGenerationPipeline(
    model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation"),
    tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
)

tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner)

#example = "I visited the damaged area of the 1999 storm near Bhubane with Shwarst in the summer of 2012"

#lab,dist = NER_label("Shwarst",example,None,ner_pipeline)

#print(lab,dist)
#exit()

#word = "dog"
#context = "the dog is swimming in  a lake"

#get_in_context_supersense(word,context,pipe)

def compute_stats(UD_file,max_len):
    l_dict = get_dict_UD(UD_file,max_len)
    d_anim_subj = {"I":0,"A":0}
    d_anim_obj = {"I":0,"A":0}
    d_anim_all = {"I":0,"A":0}
    d_pos = {"anim_pos":[],"inam_pos":[],"all_pos":[],"subj_pos":[],"obj_pos":[]}
    for d in l_dict:
        label = d["label"]
        d["label"] = label
        if d["subj"]:
            d_anim_subj[label]+=1
        elif d["obj"]:
            d_anim_obj[label]+=1
        d_anim_all[label]+=1
        if d["sent_len"] == 0:
            print("sent len nul",d)
        else:
            pos_ratio = d["pos_in_sent"]/d["sent_len"]
            d_pos["all_pos"].append(pos_ratio)
            if d["subj"]:
                d_pos["subj_pos"].append(pos_ratio)
            if d["obj"]:
                d_pos["obj_pos"].append(pos_ratio)
            if label == "I":
                d_pos["inam_pos"].append(pos_ratio)
            else:
                d_pos["anim_pos"].append(pos_ratio)
    d_pos_avg = {}
    for k in d_pos.keys():
        d_pos_avg[k] = np.mean(d_pos[k])        
    return d_anim_subj,d_anim_obj, d_anim_all, d_pos, d_pos_avg, l_dict


d_subj,d_obj, d_all,d_pos,d_pos_avg, l_dict = compute_stats(UD_file,10)

data = {"l_dict":l_dict, "nb_A_I_subj":d_subj, "nb_A_I_obj":d_obj, "nb_A_I_all": d_all, "pos_ration_distrib":d_pos,"pos_ration_avg":d_pos_avg}
with open("output_stats_anim_ang_UD_gum", 'w') as json_file:
    json.dump(data, json_file, indent=4)
print("subj:", d_subj,"obj:", d_obj,"all:",d_all)


