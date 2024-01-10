import argparse
from transformers import SpeechT5Processor, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5Model, SpeechT5Config, SpeechT5Tokenizer, SpeechT5HifiGan
from tokenizers import AddedToken
import torch
import os
from speechbrain.pretrained import EncoderClassifier
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
import warnings
warnings.filterwarnings('ignore')
path = os.environ['DOWNLOAD_PATH']
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = SpeechT5HifiGan.from_pretrained("{}/model/vocoder")
processor = AutoProcessor.from_pretrained("{}/model/processor")
model = SpeechT5ForTextToSpeech.from_pretrained("{}/model/model")

processor.feature_extractor.sampling_rate = 22050
#processor.feature_extractor.fmax = 76000
processor.tokenizer.model_max_length=10000
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

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
def training(lr, eps,
            batch_size,
            gradient_accumlation_step,
            save_step,
            eval_step,
            logging_step,
            save_total_limit,
            dataloader_num_workers,
            output_dir):

    #dataset1 = load_dataset("KyS/ReadyforTraining01")
    #dataset2 = load_dataset("KyS/ReadyforTraining02")
    #train_dataset_en = dataset2['train'].train_test_split(test_size=0.5, seed=7)
    #train_dataset = concatenate_datasets([dataset1['train'], train_dataset_en['train']])
    #train_dataset = train_dataset.shuffle(seed=37)
    #test_dataset = dataset1['test']

    dataset = load_from_disk("./dataset")

    data_collator = TTSDataCollatorWithPadding(processor=processor)
    model.config.use_cache = False
    model.resize_token_embeddings(len(processor.tokenizer))
    #dataset1 = load_dataset("KyS/ReadyforTraining01")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        num_train_epochs=eps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumlation_step,
        learning_rate=lr,
        warmup_steps=100,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_eval_batch_size=batch_size,
        save_steps=save_step,
        eval_steps=eval_step,
        logging_steps=logging_step,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        #push_to_hub=True,
        save_total_limit=save_total_limit,
        resume_from_checkpoint=True,
        run_name="ST1",
        warmup_ratio = 0.1,
        prediction_loss_only=True,
        auto_find_batch_size = True,
        dataloader_num_workers=dataloader_num_workers,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
  

def main():
    parser = argparse.ArgumentParser(description='NLP Model Training CLI')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--eps', type=int, default=100, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--gradient_accumlation_step', type=int, default=2, help='Gradient Accumlation step')
    parser.add_argument('--save_step', type=int, default=100, help='Save Step')
    parser.add_argument('--eval_step', type=int, default=100, help='Eval Step')
    parser.add_argument('--logging_step', type=int, default=500, help='Logging after this steps')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Number of weight saving')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Data Loading number')
    parser.add_argument('--output_dir', type=int, default=100, help='Output directory')
    
    args = parser.parse_args()
    
    training(
            args.lr, 
            args.eps,
            args.batch_size,
            args.gradient_accumlation_step,
            args.save_step,
            args.eval_step,
            args.logging_step,
            args.save_total_limit,
            args.dataloader_num_workers,
            args.output_dir
            )

if __name__ == "__main__":
    main()
