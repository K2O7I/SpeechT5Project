from transformers import SpeechT5Processor, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5Model, SpeechT5Config, SpeechT5Tokenizer, SpeechT5HifiGan
from tokenizers import AddedToken
import torch
import os
from speechbrain.pretrained import EncoderClassifier
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, concatenate_datasets, DatasetDict

import warnings
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
processor = AutoProcessor.from_pretrained("KyS/newProcessor")

processor.feature_extractor.sampling_rate = 22050
#processor.feature_extractor.fmax = 76000
processor.tokenizer.model_max_length=10000
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"


speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name)
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio = example["audio"]

    # feature extraction and tokenization
    example = processor(
        text=example["content"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
data_collator = TTSDataCollatorWithPadding(processor=processor)
model.config.use_cache = False
model.resize_token_embeddings(len(processor.tokenizer))

dataset1_TSS_03 = load_dataset("DataStudio/TTS_03")
dataset1_TSS_02 = load_dataset("DataStudio/TTS_02")
dataset1_TSS_01 = load_dataset("DataStudio/Vietnamese_Audio_v1.0")
datasets_ = concatenate_datasets([dataset1_TSS_03['train'], dataset1_TSS_02['train'], dataset1_TSS_01['train']])
datasets_ = datasets_['train'].train_test_split(test_size=0.1)
dataset1 = datasets_.map(
    prepare_dataset, remove_columns=["audio", "content"],
)

dataset2 = load_dataset("KyS/ReadyforTraining02")
train_dataset_en = dataset2['train'].train_test_split(test_size=0.5, seed=7)
train_dataset = concatenate_datasets([dataset1['train'], train_dataset_en['train']])
train_dataset = train_dataset.shuffle(seed=37)
test_dataset = dataset1['test']

ds = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    })


if not os.path.exists("./dataset"):
  os.mkdir("./dataset")
ds.save_to_disk("./dataset")