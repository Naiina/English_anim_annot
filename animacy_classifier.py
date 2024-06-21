import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Download the WordNet data if you haven't already
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_supersenses(word):
    # Get all synsets for the word
    synsets = wn.synsets(word)
    l_def_and_supersenses = []
    # Loop through each synset and print its supersense information
    for synset in synsets:
        definition = synset.definition()
        supersense = synset.lexname()
        l_def_and_supersenses.append((definition,supersense))
        #print(f"Synset: {synset.name()}")
        #print(f"  Definition: {synset.definition()}")
        #print(f"  Lemmas: {[lemma.name() for lemma in synset.lemmas()]}")
        #print(f"  Part of Speech: {synset.pos()}")
        #print(f"  Supersense: {synset.lexname()}")
        #print()
    return l_def_and_supersenses


# Get supersense information for the word "cat"
#get_supersenses("cat")



model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")

input = '''question: which description describes the word " java "\
           best in the following context? \
descriptions:[  " A drink consisting of an infusion of ground coffee beans ", 
                " a platform-independent programming language ", or
                " an island in Indonesia to the south of Borneo " ] 
context: I like to drink " java " in the morning .'''


example = tokenizer.tokenize(input, add_special_tokens=True)

answer = model.generate(input_ids=example['input_ids'], 
                                attention_mask=example['attention_mask'], 
                                max_length=135)

print(answer)                            
# "a drink consisting of an infusion of ground coffee beans"

def get_in_context_supersnese(word,context):
    l_def_and_supersenses = get_supersenses(word)
    question = '''question: which description describes the word " ''' +word+ ''' "\
           best in the following context? \n'''
    
    list_desript = ''''''
    for desc,s in l_def_and_supersenses:
        Desc = ''' " %s ", ''' %desc
        list_desript.append(Desc)
    descriptions='''desription: [  %s ] \n''' %Desc
    context = '''context: %s ''' %context
    input = question+descriptions+context

    example = tokenizer.tokenize(input, add_special_tokens=True)

    answer = model.generate(input_ids=example['input_ids'], 
                                attention_mask=example['attention_mask'], 
                                max_length=135)






