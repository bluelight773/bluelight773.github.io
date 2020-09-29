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
To download the data, we'll use the [Download All Images Extension]{https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm} for Google Chrome. Once installed, go to [Google Images]{https://images.google.com} and search for "Creative Commons License" images with the query `male faces`. Then use the extension to download a zip of the images.
