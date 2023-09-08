import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

DEFAULT_PARAMS = {
    "epochs": 2,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "warmup_steps": 500,
    "learning_rate": 5e-5,
    "fp16": True,
    "model_ckpt": "distilbert-base-uncased",
    "dataset_channel": "train",
    "mini_size": 10000,
}
# NUM_LABELS = 825
NUM_LABELS = 824
RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.05
TRAIN_VALIDATION_SPLIT_RATIO = 0.05


logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"This is the torch device: {device}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=DEFAULT_PARAMS["epochs"])
    parser.add_argument(
        "--train_batch_size", type=int, default=DEFAULT_PARAMS["train_batch_size"]
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=DEFAULT_PARAMS["eval_batch_size"]
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=DEFAULT_PARAMS["warmup_steps"]
    )
    parser.add_argument(
        "--learning_rate", type=str, default=DEFAULT_PARAMS["learning_rate"]
    )
    parser.add_argument("--fp16", type=bool, default=DEFAULT_PARAMS["fp16"])

    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    parser.add_argument("--model_ckpt", type=str, default=DEFAULT_PARAMS["model_ckpt"])
    parser.add_argument(
        "--dataset_channel", type=str, default=DEFAULT_PARAMS["dataset_channel"]
    )
    parser.add_argument("--mini_size", type=int, default=DEFAULT_PARAMS["mini_size"])

    args, _ = parser.parse_known_args()

    return args


def split_train_validation_test(
    input_dataframe, train_test_split_ratio, train_validation_split_ratio, seed
):
    train_plus_validation_df, test_df = train_test_split(
        input_dataframe,
        test_size=train_test_split_ratio,
        random_state=seed,
        stratify=input_dataframe["label"],
    )
    train_df, validation_df = train_test_split(
        train_plus_validation_df,
        test_size=train_validation_split_ratio,
        random_state=seed,
        stratify=train_plus_validation_df["label"],
    )

    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    log.info(f"Size of training dataframe: {train_df.count()}")
    log.info(f"Size of validation dataframe: {validation_df.count()}")
    log.info(f"Size of test dataframe: {test_df.count()}")

    return train_df, validation_df, test_df


def convert_pandas_to_huggingface(train_df, validation_df, test_df):
    modeling_dataset = DatasetDict()

    train_df_huggingface = Dataset.from_pandas(train_df, preserve_index=None)
    validation_df_huggingface = Dataset.from_pandas(validation_df, preserve_index=None)
    test_df_huggingface = Dataset.from_pandas(test_df, preserve_index=None)

    modeling_dataset["train"] = train_df_huggingface
    modeling_dataset["validation"] = validation_df_huggingface
    modeling_dataset["test"] = test_df_huggingface

    # Huggingface API assumes the name of y column is labels
    modeling_dataset = modeling_dataset.rename_column("label", "labels")

    return modeling_dataset


def select_mini_dataset(huggingface_dataset, size):
    mini_dataset_train = huggingface_dataset["train"].select(range(size))
    mini_dataset_validation = huggingface_dataset["validation"].select(range(size))
    return mini_dataset_train, mini_dataset_validation


def build_dataset(input_df, dataset_channel, mini_size):
    train_df, validation_df, test_df = split_train_validation_test(
        input_df, TRAIN_TEST_SPLIT_RATIO, TRAIN_VALIDATION_SPLIT_RATIO, RANDOM_SEED
    )

    modeling_dataset = convert_pandas_to_huggingface(train_df, validation_df, test_df)

    if dataset_channel == "mini":
        modeling_dataset_train, modeling_dataset_validation = select_mini_dataset(
            modeling_dataset, mini_size
        )
    else:
        modeling_dataset_train = modeling_dataset["train"]
        modeling_dataset_validation = modeling_dataset["validation"]
    return modeling_dataset, modeling_dataset_train, modeling_dataset_validation


def tokenize_dataset(huggingface_dataset, tokenizer):
    def tokenizer_func(batch):
        return tokenizer(batch["feature"], padding=True, truncation=True)

    huggingface_dataset_encoded = huggingface_dataset.map(
        tokenizer_func, batched=True, batch_size=32
    )
    return huggingface_dataset_encoded


def build_tokenizer_and_model(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt, use_fast=True, do_lower_case=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=NUM_LABELS
    ).to(device)
    return tokenizer, model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def get_class_weights(dataset):
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(dataset["train"]["labels"]),
        y=dataset["train"]["labels"],
    )
    return torch.tensor(class_weights, dtype=torch.float)


def train_model(
    training_dir,
    output_data_dir,
    output_dir,
    dataset_channel,
    mini_size,
    model_ckpt,
    epochs,
    warmup_steps,
    fp16,
    train_batch_size,
    eval_batch_size,
    learning_rate,
):
    log.info("Training model...")
    input_df = pd.read_parquet(training_dir)
    full, train, validation = build_dataset(input_df, dataset_channel, mini_size)

    tokenizer, model = build_tokenizer_and_model(model_ckpt)
    train_encoded = tokenize_dataset(train, tokenizer)
    validation_encoded = tokenize_dataset(validation, tokenizer)
    dataset_encoded = DatasetDict(
        {
            "train": train_encoded,
            "validation": validation_encoded,
        }
    )
    log.info(f"This is the tokenized dataset: {dataset_encoded}")

    class_weights = get_class_weights(full)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(device))
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True if get_last_checkpoint(output_dir) is not None else False,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        fp16=fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{output_data_dir}/logs",
        learning_rate=float(learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_encoded,
        eval_dataset=validation_encoded,
        tokenizer=tokenizer,
    )

    if get_last_checkpoint(output_dir) is not None:
        log.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    model_path = os.path.join(os.environ["SM_MODEL_DIR"], "model-artifact")
    trainer.save_model(model_path)
    log.info("Model artifacts saved in: %s", model_path)


def train():
    args = parse_arguments()

    train_model(
        args.training_dir,
        args.output_data_dir,
        args.output_dir,
        args.dataset_channel,
        args.mini_size,
        args.model_ckpt,
        args.epochs,
        args.warmup_steps,
        args.fp16,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
    )


if __name__ == "__main__":
    train()