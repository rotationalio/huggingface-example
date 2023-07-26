import os
import sys
import asyncio
import json
from datetime import datetime

import pandas as pd

from pyensign.events import Event
from pyensign.ensign import Ensign
from pyensign.api.v1beta1.ensign_pb2 import Nack

from transformers import (
    DistilBertTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


async def print_ack(ack):
    """
    Enable the Ensign server to notify the Publisher the event has been acknowledged
    """
    ts = datetime.fromtimestamp(ack.committed.seconds + ack.committed.nanos / 1e9)
    print(ts)

async def print_nack(nack):
    """
    Enable the Ensign server to notify the Publisher the event has NOT been
    acknowledged
    """
    print(f"Could not commit event {nack.id} with error {nack.code}: {nack.error}")

        
class ScoreDataPublisher:
    """
    Read data from the yelp_score.csv file and publish to yelp_data topic.
    This can be replaced by a real time streaming source
    Check out https://github.com/rotationalio/data-playground for examples
    """
    def __init__(self, topic="yelp_data", interval=1):
        self.topic = topic
        self.ensign = Ensign()
        self.interval = interval

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.publish())

    async def publish(self):
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        score_df = pd.read_csv(os.path.join("data", "yelp_score.csv"))
        score_dict = score_df.to_dict("records")
        for record in score_dict:
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=print_ack, on_nack=print_nack)
        
        await asyncio.sleep(self.interval)


class HuggingFaceScorer:
    """
    HuggingFaceScore listens to the yelp_data topic and generates
    predictions using the model located in the final_model directory.
    """

    def __init__(self, topic="yelp_data", model_dir="final_model"):
        self.topic = topic
        self.model_dir = model_dir
        self.ensign = Ensign()
        self.load_model()
    
    def run(self):
        """
        Run the subscriber forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    def load_model(self):
        """
        Select the tokenizer and model and set up the classifier
        """
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    async def decode(self, event):
         """
         Decode and ack the event.
         """
         try:
             data = json.loads(event.data)
         except json.JSONDecodeError:
             print("Received invalid JSON in event payload:", event.data)
             await event.nack(Nack.Code.UNKNOWN_TYPE)
             return

         await event.ack()
         return data

    async def generate_predictions(self, data):
        text_list = []
        text = data["text"]
        text_list.append(text)
        # the classifier takes in a list of text data to make a prediction
        pred_info = self.classifier(text_list)
        pred = 0 if "NEGATIVE" in pred_info[0]["label"] else 1
        pred_score = pred_info[0]["score"]
        label = data["labels"]
        print(text)
        print(f"prediction: {pred}, prediction_score: {pred_score}, label: {label}")
          
    async def subscribe(self):
        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.topic)
        async for event in self.ensign.subscribe(self.topic):
            data = await self.decode(event)
            await self.generate_predictions(data)

if __name__ == "__main__":
    # Run the publisher or subscriber depending on the command line arguments.
    if len(sys.argv) > 1:
        if sys.argv[1] == "score_data":
            publisher = ScoreDataPublisher()
            publisher.run()
        elif sys.argv[1] == "score":
            subscriber = HuggingFaceScorer()
            subscriber.run()
        else:
            print("Usage: python huggingface_scorer.py [score_data|score]")
    else:
        print("Usage: python huggingface_scorer.py [score_data|score]")