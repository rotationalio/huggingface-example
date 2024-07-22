import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import (
    DistilBertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from datasets import Dataset

 
class HuggingFaceTrainer:
    """
    The HuggingFaceTrainer class uses the `distilbert-base-uncased-finetuned-sst-2-english`
    model to train on sample Yelp data.
    """
    def __init__(self, train_dir="trained_models", model_dir="final_model"):
        self.train_dir = train_dir
        self.model_dir = model_dir
        self.initialize_directories()
        self.initialize_model_and_metrics()

    def initialize_directories(self):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def initialize_model_and_metrics(self):
        """
        Set up the inputs for the HuggingFace Trainer
        """
        # params for TrainingArguments used by Trainer
        params={
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 3,
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "metric_for_best_model": "f1_true",
            "load_best_model_at_end": True,
            "push_to_hub": False,
            "use_cpu": True,
        }
        # select the tokenizer
        self.tokenizer=DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        # select the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        # set up the TrainingArguments
        self.training_args = TrainingArguments(
            **params,
            output_dir=self.train_dir,
        )

    def preprocess_function(self, instances):
        """
        Preprocess function used to tokenize the text data
        """
        return self.tokenizer(
            instances["text"], truncation=True, padding="max_length"
        )
    
    def compute_metric(self, eval_pred, metric_name):
        """
        Calculate the metric score for the positive class and the negative class
        """
        preds, labels = eval_pred
        predictions = np.argmax(preds, axis=-1)
        metric = evaluate.load(metric_name)
        metric_true_output = metric.compute(
            references=labels, predictions=predictions, pos_label=1
        )
        metric_true_output[f"{metric_name}_true"] = metric_true_output.pop(metric_name)
        metric_false_output = metric.compute(references=labels, predictions=predictions)
        metric_false_output[f"{metric_name}_false"] = metric_false_output.pop(
            metric_name
        )
        return metric_true_output, metric_false_output

    def generate_metrics(self, eval_pred):
        """
        Creates a dictionary object that contains the precision, recall, and f1 scores
        for the positive class and the negative class at the end of each training run.
        """
        recall_true_output, recall_false_output = self.compute_metric(
            eval_pred, "recall"
        )
        precision_true_output, precision_false_output = self.compute_metric(
            eval_pred, "precision"
        )
        f1_true_output, f1_false_output = self.compute_metric(eval_pred, "f1")
        metrics_dict = {
            **recall_true_output,
            **recall_false_output,
            **precision_true_output,
            **precision_false_output,
            **f1_true_output,
            **f1_false_output,
        }
        return metrics_dict

    def train(self):
        """
        Train Hugging Face model
        At the end of each epoch the model configurations and weights are stored in
        the `trained_models` directory.
        After training the best model configurations and weights are stored
        in the `final_model` directory
        """
        yelp_df = pd.read_csv(os.path.join("data", "yelp_train.csv"))
        train_df, test_df = train_test_split(yelp_df, test_size=0.2)

        # convert from pandas dataframe to a Dataset object
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Use the preprocess_function to tokenize the data
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)

        # set up the Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            compute_metrics=self.generate_metrics,
        )

        # train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)

        # save the best model based on evaluation metrics
        trainer.save_model(self.model_dir)


if __name__ == "__main__":
    hf_trainer = HuggingFaceTrainer()
    hf_trainer.train()