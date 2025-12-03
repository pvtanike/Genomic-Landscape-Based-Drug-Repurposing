### Import necessary libraries
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import huggingface_hub
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

### Login to hugging face to ensure effective download of gemma model
login(token='<enter your hugging face token here>', add_to_git_credential=True)
print('1. Logged in hugging face succesfully')

### Check if gpu available
if torch.backends.mps.is_available():
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print(x)
else:
    print("CUDA DEVICE NOT FOUND")

print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)

if torch.cuda.is_available() == False:
    print("No GPU available")
    exit()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('2. Checking GPU usage')

# Load gemma model model directly, choice of lighter 2b model or stronger 7b model
variant = '2b'
model_name = f'google/gemma-{variant}-it'  # or "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto')
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
model = model.to(device)
print('3. loaded gemma model succesfully')

import torch.nn as nn
print(isinstance(model, nn.DataParallel))
print(isinstance(model, nn.parallel.DistributedDataParallel))
print(model)
print(" Double check model using mutliple GPU")

### Import data- data curated from PUBMED api calls and extraction of abstracts related to head and neck cancers
data_file = 'Data/head and neck cancer query abstracts.csv'
inputData = pd.read_csv(data_file)
print('4. read in data succesfully')

## Task definition, define the task for the LLM AI model to utilize for extracting drug targets
TASK_DESCRIPTION = """<start_of_turn>user
You are required to read the given abstract and determine any key disease targets for the cancer mentioned. Responding with the disease target, if there are multiple seperate by comma. Explain your answer with the sentence where the information is found. If there is no information related to the disease target, respond with "No information available". Always place answer of disease targets after "Key disease targets:" and sentence where information is found after "Sentence where information is found:".
You should either respond:
- Key disease targets: Respond with identified disease targets, and the sentence where information is found \n Sentence where information is found: Respond with sentence where information is found.
- "No information available": The abstract did not contain information related to the disease target
"""
USER_CHAT_TEMPLATE = 'Analyze the following abstract to determine key disease targets for the cancer type mentioned. If there are multiple targets seperate by comma. Provide sentence where the information is found for explanation.:\n{abstract}<end_of_turn>\n'

# Template for model response based on the analysis of the clinical notes
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n'
print('4. Created prompt')

# Define data to work with. currently extracting from year 2000 and after to identify relavent literature for drug repurposing targets
year_cut_off = 2000
workingData = inputData[inputData['YEAR']>year_cut_off]


# extract data
extractedData = []
for abstract in tqdm(workingData['ABSTRACT']):
    # Constructing the prompt for the model
    prompt = (
        TASK_DESCRIPTION
        + USER_CHAT_TEMPLATE.format(abstract=abstract)
        + MODEL_CHAT_TEMPLATE
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, max_length = 5000)
    extractedData.append(outputs[0])


def extractTarget(inputText):
    inputText = str.lower(inputText)
    #print(inputText)
    word = '<start_of_turn>model\n'
    idx = inputText.index(word)
    # print(idx)
    sentenceToFocus = inputText[idx+len(word)::]
    sentenceToFocus = sentenceToFocus.replace('**','')
    try:
        #print(sentenceToFocus)
        word3 = 'key disease targets: '
        idx3 = sentenceToFocus.index(word3)
        word4 = '\nsentence'
        idx4 = sentenceToFocus.index(word4)
        target = sentenceToFocus[idx3:idx4]
        target = target.replace('\n', '')
        target = target.replace('key disease targets: ','')
        return target
    except:
        sentenceToFocus = sentenceToFocus.replace('\n', ' ')
        sentenceToFocus = sentenceToFocus.replace('<eos>', '')
        sentenceToFocus = sentenceToFocus.replace('key disease targets: ','')
        return sentenceToFocus
    
def extractSentences(inputText):
    inputText = str.lower(inputText)
    #print(inputText)
    word = '<start_of_turn>model\n'
    idx = inputText.index(word)
    # print(idx)
    sentenceToFocus = inputText[idx+len(word)::]
    sentenceToFocus = sentenceToFocus.replace('**','')
    try:
        #print(sentenceToFocus)
        word3 = 'sentence'
        idx3 = sentenceToFocus.index(word3)
        sentence = sentenceToFocus[idx3::]
        word4 = ': '
        idx4 = sentence.index(word4)
        sentence = sentence[idx4+ len(word4):]
        sentence = sentence.replace('\n', '')
        return sentence
    except:
        return 'NA'
    
targets = []
for output in tqdm(extractedData):
    output = tokenizer.decode(output)
    targets.append(extractTarget(output))
    #print(output)


sentences = []
for output in tqdm(extractedData):
    output = tokenizer.decode(output)
    sentences.append(extractSentences(output))

workingData['TARGETS'] = targets
workingData['SENTENCE_WHERE_FOUND'] = sentences

print('5. Working data completed')

workingData.to_csv(f'extracted_targets_all_pub_after_{year_cut_off}_GPU_{variant}_gemma.csv')
