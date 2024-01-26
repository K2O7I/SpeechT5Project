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
model = SpeechT5ForTextToSpeech.from_pretrained("KyS/ST5_enhance_f2")
model.save_pretrained("./model/model", from_pt=True)
