from transformers import SpeechT5Processor, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5Model, SpeechT5Config, SpeechT5Tokenizer, SpeechT5HifiGan
from datasets import load_dataset, concatenate_datasets, DatasetDict
import os

if not os.path.exists("./model"):
  os.mkdir("./model")
if not os.path.exists("./model/vocoder"):
  os.mkdir("./model/vocoder")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
vocoder.save_pretrained("./model/vocoder", from_pt=True)

if not os.path.exists("./model/processor"):
  os.mkdir("./model/processor")
processor = AutoProcessor.from_pretrained("KyS/newProcessor")
processor.save_pretrained("./model/processor", from_pt=True)

if not os.path.exists("./model/model"):
  os.mkdir("./model/model")
model = SpeechT5ForTextToSpeech.from_pretrained("KyS/A_01.1")
model.save_pretrained("./model/model", from_pt=True)

#embeddings_dataset = load_dataset("KyS/SpeakerEmbedding", use_auth_token = hf_token)
#embeddings_dataset.save_pretrained("./model", from_pt=True)

dataset1 = load_dataset("KyS/ReadyforTraining01")
#dataset2 = load_dataset("KyS/ReadyforTraining02")

#train_dataset_en = dataset2['train'].train_test_split(test_size=0.5, seed=7)
#train_dataset = concatenate_datasets([dataset1['train'], train_dataset_en['train']])
#train_dataset = train_dataset.shuffle(seed=37)
#test_dataset = dataset1['test']

#ds = DatasetDict({
#    'train': train_dataset,
#    'test': test_dataset,
#    })

ds = dataset1

if not os.path.exists("./dataset"):
  os.mkdir("./dataset")
ds.save_to_disk("./dataset")

