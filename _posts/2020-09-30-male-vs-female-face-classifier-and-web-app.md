# Building a Male vs Female Face Classifier and Web App with FastAI

We'll go through all the steps of building and improving an image classifier to distinguish male from female faces using FastAI, and even build a simple web app UI using Voila. This tutorial assumes one is using Ubuntu 18.04 LTS.

1. TOC
{:toc}

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
voila==0.2.3
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
To download the data, we'll use the [Download All Images Extension](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm) for Google Chrome. Once installed, go to [Google Images]{https://images.google.com} and search for "Creative Commons License" images with the query `male faces`. Then use the extension to download a zip of the images. Extract the images then do an initial cleanup to remove bad images, such as cartoons or images that are not of male faces altogether.

Repeat this process for `female faces`. Place the images in `data/male` and `data/female`, respectively. Ensure you have at least 100 images per category.

# Quick Model

Let's build a quick model. We can initially run the code in a Jupyter notebook. To do so, from the repo root, run:
~~~bash
jupyter notebook
~~~

Run the following code to finetune a resnet18 model.
~~~python
from fastai.vision.all import *
path = Path('data')

# Our x are images, our y is a category. We'll resize all images initially to 128x128. The label for each image can be
# determined from its parent folder. We'll apply an 80/20 training/validation split.
faces = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = faces.dataloaders(path)

# Finetune an Imagenet-trained resnet18 model where once epoch is run with the last layer unfrozen, then the rest of the
# network is unfrozen for 4 epochs.
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)
~~~
I got around 94% accuracy rate after all epochs are finished. Not bad for an initial few lines of code.

# Improving the Model with Augmentations

We apply data augmentations - Rotation, flipping, warping, brightness changes, and contrast changes.
~~~python
set_seed(42, True)
faces = faces.new(item_tfms=Resize(128), batch_tfms=aug_transforms())
dls = faces.dataloaders(path)
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)
~~~

I got around 96% accuracy rate after all epochs are run. For more details on how the above code was arrived at after attempting different variations, check the
[Jupyter Notebook](https://github.com/bluelight773/image_classifier/blob/master/image_classifier.ipynb).

# Data Cleaning
Let's delete bad images using the image cleaner. Run the following code from a notebook cell. Mark images to be deleted.

~~~python
from fastai.vision.widgets import *
cleaner = ImageClassifierCleaner(learn)
cleaner
~~~

Once we've marked images to be deleted or moved to another category, run the following code to apply what has been marked.

~~~python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
~~~

# Saving the Model
Now that we've cleaned the data, let's train the model again and save it.
~~~python
set_seed(42, True)
faces = faces.new(item_tfms=Resize(128), batch_tfms=aug_transforms())
dls = faces.dataloaders(path)
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)

# Save model
model_path = Path("models")/"male_vs_female_face_classifier.pkl"
learn.export(model_path)

# Load and use the model
learn_inf = load_learner(model_path)
print("Classes:", learn_inf.dls.vocab)
predicted_class, predicted_class_index, pred_probs = learn_inf.predict("data/male/image.jpeg")
print("Predicted Class:", predicted_class)
~~~

# Building a MultiLabel Classification Model
One issue with the model that we've built is that for any image, it'll always predict either male or female. However, we'd like to be able to take in
images that contain neither a male face nor a female face and predict neither. To achieve this, we can frame the problem as a multi-label classification problem.

~~~python
path = Path('data')

# Ensure reproducibility of results
set_seed(42, True)

# To treat the problem as a multilabel classification problem, we provide y as a list
# indicating all the applicable categories, if any.
def get_y(file):
    return [parent_label(file)]

faces = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=get_y,
    item_tfms=Resize(128), batch_tfms=aug_transforms())

dls = faces.dataloaders(path)

# We use multi-accuracy which computes the accuracy rate across labels
learn = cnn_learner(dls, resnet18, metrics=accuracy_multi)
learn.fine_tune(4)
~~~
We get 94% multi-accuracy rate.

Let's attempt to improve the model using the learning-rate finder.

~~~python
lr_candidates = learn.lr_find()
learn.fine_tune(4, lr_candidates.lr_steep)
~~~
Our accuracy decreased slightly to 93% but we'll stick with this model as we were more methodical in the learning-rate selection.

Let's save this model so that it can be used in a simple classifier web app that we'll build in the next section.

~~~python
# Save model
model_path = Path("models")/"male_vs_female_face_classifier.pkl"
learn.export(model_path)

# Load and use the model
learn_inf = load_learner(model_path)
print("Labels:", learn_inf.dls.vocab)
predicted_labels, prediceted_label_indices, pred_probs = learn_inf.predict("data/male/image.jpeg")
print("Predicted Labels:", predicted_labels)
~~~

# Make a Simple Web App

Create a new notebook, `image_classifier_inference.ipynb` in which a simple UI is provided for uploading and labelling an image.

We will rely on Voila library to create a simple web app right out of the Jupyter notebook.

~~~python
# Restart the Jupyter server and look for a "Voila" button at the top. If you don't see it, run the following line to ensure
# Voila is enabled, then restart the Jupyter server.
# Once Voila has been enabled, comment out the line below. Clicking on the Voila button should load up this notebook, run all the cells
# but only show the output (rather than the code). The result should be to have a simple web app to which we can upload and label images.
!jupyter serverextension enable voila —sys-prefix

from fastai.vision.all import *
from fastai.vision.widgets import *

# Due to an issue with fastai's export, we need to provide the get_y function, get_y_label
# when loading the model to ensure we can use it.
def get_y_labels(file):
    return [parent_label(file)]

# Load model
model_path = Path("models")/"male_vs_female_face_classifier.pkl"
learn_inf = load_learner(model_path)

# Set up UI widgets: Upload button, Label button, Output area to show image, Label to show prediction
btn_upload = widgets.FileUpload()
btn_label = widgets.Button(description="Label")
out_img = widgets.Output()
lbl_prediction = widgets.Label()

# "Label" button callback
def on_click_label(change):
    img = PILImage.create(btn_upload.data[-1])
    out_img.clear_output()
    with out_img: display(img.to_thumb(128,128))
    labels, pred_labels_mask, pred_probs = learn_inf.predict(img)
    lbl_prediction.value = f'Prediction: {labels}, Probabilities: {pred_probs}'

btn_label.on_click(on_click_label)

# Show the vertically-aligned UI
VBox([widgets.Label('Select an image'), 
      btn_upload, btn_label, out_img, lbl_prediction])
~~~

<!--
mybinder
Flask
Reduce false positives perhaps by having a misc/landscape category.
-->
