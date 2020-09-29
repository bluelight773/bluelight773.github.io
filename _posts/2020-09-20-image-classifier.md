# Building an Image Classifier with FastAI

We'll go through all the steps of building an image classifier to distinguish male from female faces. This tutorial assumes one is using Ubuntu 18.04 LTS.

# Virtual Environment setup
Set up a virtual environment.

Ensure venv is installed.
~~~bash
sudo apt-get install python3-venv
~~~

Create a virtual environment in `~/envs`.
~~~bash
mkdir ~/envs
python3 -m venv ~/envs/image_classifier_env
~~~

Activate the environment.
~~~bash
ln -s ~/envs/image_classifier_env/bin/activate
source activate
~~~

Create requirements.txt.
~~~
wheel
jupyter==1.0.0
fastai==2.0.13
~~~

Install requirements.
~~~bash
pip install requirements.txt
~~~

Update `requirements.txt` based on what was installed.
~~~bash
pip freeze > requirements.txt
~~~

# Data
To download the data, we'll use the [Download All Images Extension]{https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm} for Google Chrome. Once installed, go to [Google Images]{https://images.google.com} and search for "Creative Commons License" images with the query `male faces`. Then use the extension to download a zip of the images. Extract the images then do an initial cleanup to remove bad images, such as cartoons or images that are not of male faces altogether.

Repeat this process for `female faces`. Place the images in `data/male` and `data/female`, respectively. Ensure you have at least 100 images per category.

# Quick Model

Let's build a quick model. The following code finetunes a resnet18 model. We can initially run the code in `ipython`.
~~~python
from fastai.vision.all import *
path = Path('data')

# Our x are images, our y is a category. We'll resize all images initially to 128x128. The label for each image can be determined from its parent folder. We'll apply an 80/20 training/validation split.
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = bears.dataloaders(path)

# Finetune an Imagenet-trained resnet18 model where once epoch is run with the last layer unfrozen, then the rest of the network is unfrozen for 4 epochs.
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
~~~

I got around 5% error with this initial effort.

<!--
Apply Data Augmentation
Save model
Load model
MultiCategory
-->
