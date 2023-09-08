import time
from sagemaker.huggingface import HuggingFace


# define job name and role
JOB_NAME = f'fwang-research-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
ROLE = "arn:aws:iam::111039177464:role/gdmlml-custom-us-west-2-cerbo-role"
S3_INPUT_PATH = "s3://gd-gdmlml-stage-sagemaker-us-west-2/fwang-dataset/2023_Q3_IC_Jetpack_retraining/20230901/data/part-00000-0b16aaf5-7c88-469a-9e51-818cc52e3b08-c000.snappy.parquet"
IMAGE_URL_AND_TAG = "111039177464.dkr.ecr.us-west-2.amazonaws.com/fwang-research:latest"


# hyperparameters, which are passed into the training job
hyperparameters = {
    "epochs": 10,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "fp16": True,
    "dataset_channel": "train",
    "mini_size": 10000,  # will be used if dataset_channel = 'mini'
    "model_ckpt": "distilbert-base-uncased",
}


# huggingface estimator configuration
estimator_config = {
    "instance_type": "ml.p3.8xlarge",
    "instance_count": 1,
    # "use_spot_instances": True,
    # "max_wait": 360000,
    # "max_run": 100000,
    "metric_definitions": [
        {"Name": "train_runtime", "Regex": "'train_runtime': ([0-9]+(.)[0-9]+),?"},
        {"Name": "eval_accuracy", "Regex": "'eval_accuracy': ([0-9]+(.)[0-9]+),?"},
        {"Name": "eval_loss", "Regex": "'eval_loss': ([0-9]+(.)[0-9]+),?"},
    ],
}


def main():
    huggingface_estimator = HuggingFace(
        entry_point="train.py",
        base_job_name=JOB_NAME,
        role=ROLE,
        py_version="py38",
        instance_type=estimator_config["instance_type"],
        instance_count=estimator_config["instance_count"],
        # use_spot_instances=estimator_config["use_spot_instances"],
        # max_wait=estimator_config["max_wait"],
        # max_run=estimator_config["max_run"],
        metric_definitions=estimator_config["metric_definitions"],
        hyperparameters=hyperparameters,
        image_uri=IMAGE_URL_AND_TAG,
    )
    huggingface_estimator.fit(
        {'train': S3_INPUT_PATH,
         'test': S3_INPUT_PATH}
    )


if __name__ == "__main__":
    main()
