# Breast Cancer Classifier

* [Project Overview](#project-overview)
    * [Dataset](#dataset)
    * [Technologies](#technologies)
    * [Workflow](#workflow)
* [How To Run](#how-to-run)
    * [Initial setup](#initial-setup)
    * [Training pipeline](#training-pipeline)
    * [Inference pipeline](#inference-pipeline)
* [Project Details](#project-details)


# Project Overview

Breast cancer is one of the most prevalent cancers worldwide, with estimates suggesting that 1 in 8 women will be affected during their lifetime. While advancements in treatment have reduced mortality rates, the timing of diagnosis plays a critical role in prognosis. Early detection significantly improves treatment efficacy, leading to better survival rates and quality of life for patients, as well as reducing the burden on healthcare systems.

Ultrasound imaging is commonly used for breast cancer screening and diagnosis due to its non-invasive nature and accessibility. However, it has limitations. Differentiating between normal tissue, benign masses, and malignant tumors can be challenging, potentially leading to missed or false diagnoses. This creates a need for more advanced diagnostic tools. In this context, AI tools can offer assistance to radiologists, potentially leading to earlier interventions.

In this project, I developed a deep learning-based approach&mdash;a Convolutional Neural Network (CNN) followed by fully connected neural layers&mdash;to classify breast ultrasound images into normal, benign or malignant.

### Dataset:

I used the **Breast Ultrasound Images Dataset**, available at: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/ [1]. This dataset includes breast ultrasound images, each associated with a mask image, categorized into three classes: normal, benign and malignant.

[1] Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. (2020) Dataset of breast ultrasound images. *Data in Brief*. 28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

### Technologies:

* Programming language: **Python (numpy, matplotlib)**
* Deep Learning: **TensorFlow and Keras**
* Virtual environment: **venv**
* Containerization: **Docker**
* Deployment: **AWS Lambda and API Gateway**
* Version control: **Git and GitHub**

### Workflow:

![](imgs/proj_overview.png)


# How To Run

This project includes (i) a [Training pipeline](#training-pipeline) and (ii) an [Inference pipeline](#inference-pipeline) to predict the life expectancy of a given country/population.

## Initial setup

#### 1. Clone the project repository:

```bash
git clone https://github.com/LaboraTORIbio/breast_cancer_classifier.git
cd breast_cancer_classifier
```

#### 2. Dowload and setup the dataset:

Download the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/) and place it in `./data/full`, inside the main project directory. The downloaded dataset includes three directories, one for each class, containing both ultrasound and mask images (see [Project Details](#project-details) for an explanation on masks).

This project only uses ultrasound images for classification. Run the following code in a terminal, from the main project directory, to create a new directory containing only ultrasound images (`./data/us_only`):

```bash
!if [ ! -d "./data/us_only" ]; then \
    echo "Creating directory ./data/us_only"; \
    cp -r ./data/full ./data/us_only && \
    rm -rf ./data/us_only/*/*mask*; \
else \
    echo "Directory ./data/us_only already exists"; \
fi
```

#### 3. Set up the virtual environment:

To create and activate a virtual environment, run the following commands from the main project directory:

```bash
python3 -m venv breast-cancer-classifier
source ./breast-cancer-classifier/bin/activate
```

To deactivate the virtual environment, simply run `deactivate`.

#### 4. Install required dependencies:

```bash
pip install requirements.txt
```

## Training pipeline

The training pipeline can be run from the main project directory:

```bash
# With default arguments:
python train.py
# Alternatively, the input (ultrasound images) and output (exported model in tflite format) file paths can be specified:
python train.py -i ./data/us_only -o ./breast_cancer_classifier.tflite
```

The pipeline will (1) split the dataset into training, validation and test, (2) load and preprocess the ultrasound images, (3) train a fully connected neural network on top of the frozen ResNet50 CNN, (4) fine-tune the outter layers of the ResNet50 CNN with the traning data, (5) evaluate the model, and (6) output the best performing model in terms of validation accuracy, in .keras and .tflite formats (please, note that a seed was not set, so different results will be obtained everytime the pipeline is run).

## Inference pipeline

The inference pipeline can be containerized to deploy it either locally or as a web service.

#### 1. Build the Docker image:

```bash
docker build -t breast-cancer-classifier .
```

The image incorporates all necessary libraries, the model and the lambda function, which includes code for image preprocessing and inference. 

#### 2. Make predictions:

Predictions&mdash;both if the model is deployed locally or as a web serive&mdash;can be made by running the `test.py`. This script can accept an image from an URL (line 45) or located locally (line 50), and it converts the image to its base64 string to send as request. When the `lambda_function.py` recieves the request, it first converts the base64 string back as an image, processes it according to the ResNet50 requirements, and makes and returns the predictions. The URL and local image provided by default in `test.py` correspond to malignant tumors.

* **Locally:**

    Run the container of the inference pipeline:

    ```bash
    docker run -it --rm -p 9000:8080 breast-cancer-classifier:latest
    ```

    Once the container is running, you can send requests to classify breast ultrasound images. First, make sure line 38 of `test.py` is uncommented and line 39 is commented, then run:

    ```bash
    python test.py
    ```
    
    ![](imgs/API_testing.png)

* **Web service:**

    The inference pipeline can be served as a web service through the AWS Lambda function. First, create an [AWS account](https://aws.amazon.com/) and an [IAM user](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html), giving the user permission for AWS Lambda and AWS ECR tasks. Then, install and configure AWS CLI, [authenticating using IAM user credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html):

    ```bash
    pip install awscli
    aws configure
    ```
    
    Next, the Docker container must be published to the **AWS Elastic Container Registry (ECR)**, through the AWS CLI:

    ```bash
    # Create a container registry:
    aws ecr create-repository --repository-name breast-cancer-classifier
    # Save the repository URI in a variable, something like:
    REPOSITORY_URI=user.dkr.ecr.region.amazonaws.com/breast-cancer-classifier
    # (note: replace user and region by your own)
    # Generate the REMOTE_URI by concatenating the repository URI to the image tag:
    TAG=breast-cancer-classifier-v1
    REMOTE_URI=${REPOSITORY_URI}:${TAG}
    # Login to the registry:
    $(aws ecr get-login --no-include-email)
    # Tag and push image to ECR:
    docker tag clothing-model:latest ${REMOTE_URI}
    docker push ${REMOTE_URI}
    ```
    
    Now, the **AWS Lambda function** can easily be created from the container image: Lambda > Functions > Create function > Container image > Enter function name (breast-cancer-classifier) and select image in "Container image URI" > Create function. The Lambda function can be tested, using the base64 string of an image:

    ![](imgs/lambda_function.png)

    Finally, the Lambda function can be exposed through **API Gateway** so that requests can be made from a local computer: API Gateway > Create API > REST API (Build) > New API, enter API name (breast-cancer-classifier) > Create API > Create resource > Give Resource name /predict > Create method > Method type: POST, Integration type: Lambda function, select the breast-cancer-classifier Lambda function > Deploy API (new stage, giving the name test).

    ![](imgs/API_resource.png)
    ![](imgs/API_deployed.png)

    Now, requests can be made by running the `train.py` script, commenting line 38 and uncommenting line 39, where the Invoke URL provided by API Gateway should be provided, ended by the resource name /predict:

    ```bash
    python test.py
    ```

    ![](imgs/API_testing.png)


# Project Details

