# HuggingFace and Deep Learning guided tour for Macs with Apple Silicon (Version 2)

A guided tour on how to install optimized `tensorflow` and `pytorch` on Apple Silicon Mac and how to use `HuggingFace` large language models for your own experiments.

We will perform the following steps:

- Install `homebrew` 
- Install tensorflow with and Apple's metal pluggable metal driver  optimizations
- Install pytorch with MPS (metal performance shaders) support 
- Install `jupyter lab` to run notebooks
- Install `HuggingFace` and run some pre-trained language models using `transformers` and just a few lines of code within jupyter lab.

> **Note:** Previous versions of this guide used conda and specific conda chanels to install custom version of pytorch and tensorflow and its support software. This kind of special versions are _no longer required_! The recommendation is to uninstall conda and use Python's `venv` to install the required software. See below at the end of this readme for uninstallation instructions for `conda`.

## 1. Preparations

### 1.1 Install homebrew

If you haven't done so, go to <https://brew.sh/> and follow the instructions to install homebrew.
Once done, open a terminal and type `brew --version` to check that it is installed correctly.

Now use `brew` to install more recent versions of `python` and `git`:

```bash
brew install python git
```

To update at a later time:

```bash
brew upgrade
```

### 1.2 Test project

Now clone this project as a test project:

```bash
git clone https://github.com/domschl/HuggingFaceGuidedTourForMac
```

Now create a Python environment for this project and activate it:

```bash
python -m venv HuggingFaceGuidedTourForMac
```
This added the files required (python binaries, libraries, configs) for the virtual python environment to the project we just cloned.

```bash
cd HuggingFaceGuidedTourForMac
source bin/activate
```

Now the directory `HuggingFaceGuidedTourForMac` contains the content of the github repository (e.g. `00-SystemCheck.ipynb`) _and_ the the files for the virtual env (e.g. `bin/`):

![Folder content](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/Resources/ProjectFolder.png)

### 1.3 When you done with your project

Do deactivate this virtual environment, simply use:

```bash
deactivate
```

To re-activate it, enter the directory `HuggingFaceGuidedTourForMac` and use:

```bash
source bin/activate
```

> **Note:** See <https://docs.python.org/3/tutorial/venv.html> for more information about Python virtual environments.

## 2. Install `tensorflow`

> **Note:** Since `tensorflow` version 2.13, installation has significantly simplified since the standard tensorflow can be used with addition of the (pluggable) metal driver.

Make sure that your virtual environment is active with `pip -V` (uppercase V), this should show a path for `pip` within your project:

`<your-path>/HuggingFaceGuidedTourForMac/lib/python3.11/site-packages/pip (python 3.11)`

Following <https://developer.apple.com/metal/tensorflow-plugin/>, we will install `tensorflow` with `pip` within our `venv`:

```bash
pip install -U tensorflow tensorflow-metal
```

#### 2.1 Quick-test

To test that `tensorflow` is installed correctly, open a terminal, type `python` and within the python shell, enter:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

You should see something like:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3 Install `pytorch`

Following `https://pytorch.org`, we will install Pytorch with `pip`. The current version 2 in order to get MPS (Metal Performance Shaders) support within pytorch, which offers significant performance advantage on Apple Silicon.

To install `pytorch` into the `venv`:

```bash
pip install -U torch torchvision torchaudio
```

#### 3.1 Quick-test

To test that `pytorch` is installed correctly, and MPS metal performance shaders are available, open a terminal, type `python` and within the python shell, enter:

```python
import torch
# check if MPS is available:
torch.backends.mps.is_available()
```

This should return `True`.

### 4 Jupyter lab

At this point, your Apple Silicon Mac should be ready to run `tensorflow` and `pytorch` with hardware acceleration support, using the Apple Metal framework.

To test this, you can use `jupyter lab` to run some notebooks. To install `jupyter lab`, type:

```bash
pip install -U jupyterlab ipywidgets
```

To start Jupyter lab, type:

```bash
jupyter lab
```

This should open a browser window with `jupyter lab` running. You can then create a new python notebook and run some code to test that `tensorflow` and `pytorch` are working correctly:

![](Resources/jupyterlab.png)

```python
import tensorflow as tf
import torch

print("Tensorflow version:", tf.__version__)
print("Pytorch version:", torch.__version__)
```

If this completed successful, your Mac is now ready for Deep Learning experiments.

## 5 HuggingFace

HuggingFace is a great resource for NLP and Deep Learning experiments. It provides a large number of pre-trained language models and a simple API to use them. It will allow us to quickly get started with Deep Learning experiments.

### 5.1 Install `transformers`


From the [huggingface installation instructions](https://huggingface.co/docs/transformers/installation), we use `pip` to install `transformers`:

```bash
pip install -U transformers
```

> **Note:** When experimenting with HuggingFace, you will download large models that will be stored in your home directory at: `~/.cache/huggingface/hub`. 
> You can remove these models at any time by deleting this directory or parts of it's content.

## 6 Experiments

### 6.1 Simple sentiment analysis

Within the directory `HuggingFaceGuidedTourForMac` start `jupyter lab` and load the `00-SystemCheck.ipynb` notebook. Use `<Shift>-Enter` to run the notebook's cells.

> **Note:** If you started Jupyter Lab before installing Huggingface, you either need to restart the python kernel in Jupyter or simply restart Jupyter Lab, otherwise it won't find the Transformers library.

Your should see something like this:

![](Resources/huggingface-transformers.png)

If you've received a label classification of `POSITIVE` with a score of `0.99`, then you are ready to start experimenting with HuggingFace!

> **Note:** You'll see that the `HuggingFace` libraries are downloading all sorts of large binary blobs containing the trained model data. That data is stored in your home directory at: `~/.cache/huggingface/hub`. You can remove these models at any time by deleting this directory or parts of it's content.

#### Trouble-shooting

- If self-tests fail ('xyz not found!'), make sure that tensorflow (optional), pytorch, jupyter, and huggingface are all installed into the same, active Python virtual environment, otherwise the components won't 'see' each other!

### 6.2 Minimal chat-bot

You can open the notebook [`01-ChatBot.ipynb`](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/01-ChatBot.ipynb) to try out a very simple chatbot on your Mac.

The python code used is:

```python
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# Disable warnings about padding_side that cannot be rectified with current software:
logging.set_verbosity_error()

model_names = ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
use_model_index = 1  # Change 0: small model, 1: medium, 2: large model (requires most resources!)
model_name = model_names[use_model_index]
          
tokenizer = AutoTokenizer.from_pretrained(model_name) # , padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

# The chat function: received a user input and chat-history and returns the model's reply and chat-history:
def reply(input_text, history=None):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), chat_history_ids

history = None
while True:
    input_text = input("> ")
    if input_text in ["", "bye", "quit", "exit"]:
        break
    reply_text, history_new = reply(input_text, history)
    history=history_new
    if history.shape[1]>80:
        old_shape = history.shape
        history = history[:,-80:]
        print(f"History cut from {old_shape} to {history.shape}")
    # history_text = tokenizer.decode(history[0])
    # print(f"Current history: {history_text}")
    print(f"D_GPT: {reply_text}")
```

This shows a (quite limited and repetitive) chatbot using Microsoft's [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Mariama%21+How+are+you%3F) models.

Things to try:

- By changing `use_model_index` between `0..2`, you can select either a small, medium or large language model.
- To see the history that the model maintains you can uncomment the two `history_text` related lines above.
- To get rid of the downloaded models, clean up `~/.cache/huggingface/hub`. Missing stuff is automatically re-downloaded when needed.

## Next steps

- Your Mac can run large language models that rival the performance of commercial solutions. An excellent example is the [`llama.cpp` project](https://github.com/ggerganov/llama.cpp) that implements the inference code necessary to run LLMs in highly optimized C++ code, supporting the Mac's Metal acceleration.<br>A step-by-step guide to compile and run Llama 2 chat can be found here:<br>[Llama.cpp chat using the Llama 2 model](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/NextSteps/llama.cpp.md)

## Conda uninstallation notes

> **Note:** This paragraph is to uninstall conda that was used in older versions of this guide:

`brew uninstall miniconda`

Additional modifications are (all of them are inactive, once miniconda is removed):

- `~/.condarc` (list of channels), and `~/.conda\`.
- `~/.zshrc` (or `.bashrc`) for the setup of path and environment.
- After using hugginface models, large model binary blobs may reside at: `~/.cache/huggingface/hub`. Simply remove the directory.

## Changes

- 2023-10-30: Restested with macOS 14.1 Sonoma, Tensorflow 2.14, Pytorch 2.1.
- 2023-09-25: (Guide version 2.0) Switched from `conda` to `pip` and `venv` for latest versions of tensorflow 2.13, Pytorch 2, macOS Sonoma, installation is now much simpler.
- 2023-03-16: Since `pytorch` v2.0 is now released, the channel `pytorch-nightly` can now be replaced by `pytorch` in the installation instructions. The `pytorch-nightly` channel is no longer needed for MPS support.
