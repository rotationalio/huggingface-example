# huggingface-example
Sample NLP streaming workflow using an LLM from Hugging Face and PyEnsign

This is an example of a sentiment analysis application using sample yelp ratings data from [Kaggle](https://www.kaggle.com) using [Hugging Face](https://huggingface.co) and [PyEnsign](https://github.com/rotationalio/pyensign).

To use PyEnsign, create a free account on [rotational.app](https://rotational.app/).  You will need to do the following once you create an account:

- [Create a project.](https://youtu.be/VskNgAVMORQ)
- Add the following topic to the project: `yelp_data`.  Check out this [video](https://youtu.be/1XuVPl_Ki4U) on how to add a topic.  You can choose your own names for the topic but make sure that you update the code accordingly.
- [Generate API keys for your project.](https://youtu.be/KMejrUIouMw)

You will need to create and source the following environment variables prior to running the example:

```
export ENSIGN_CLIENT_ID="your client id here"
export ENSIGN_CLIENT_SECRET="your client secret here"
```

This application consists of three components:
- `Trainer` reads data from the `yelp_train.csv` file and builds a model using the pretrained `DistilBERT` LLM from Hugging Face. The best model gets written to the `final_model` directory.
- `ScoreDataPublisher` reads data from the `yelp_score.csv` file publishes to the `yelp_data` topic.
- `Scorer` listens for new messages in the `yelp_data` topic.  When it receives a new message, it uses the trained Hugging Face model in the `final_model` directory to make predictions.

## Steps to run the application

### Create a virtual environment

```
$ virtualenv venv
```

### Activate the virtual environment

```
$ source venv/bin/activate
```

### Install the required packages

```
$ pip install -r requirements.txt
```

### Open three terminal windows

#### Run the Trainer in the first window (make sure to activate the virtual environment first).  This will create three checkpoint directories under the `trained_models` directory and the final model configurations and weights in the `final_model` directory.
```
$ source venv/bin/activate
```

```
$ python huggingface_trainer.py
```

#### Once the training is complete, run the Scorer in the second window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python huggingface_scorer.py score
```

#### Run the ScoreDataPublisher in the third window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python huggingface_scorer.py score_data
```


