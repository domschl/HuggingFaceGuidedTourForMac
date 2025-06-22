![Version](http://img.shields.io/badge/Version-3-blue.svg?style=flat)

# HuggingFace and Deep Learning guided tour for Macs with Apple Silicon

A guided tour on how to install optimized `pytorch` and optionally Apple's new `MLX` and/or Google's `tensorflow` or `JAX` on Apple Silicon Macs and how to use `HuggingFace` large language models for your own experiments. Recent Mac show good performance for machine learning tasks.

We will perform the following steps:

- Install `homebrew` 
- Install `pytorch` with MPS (metal performance shaders) support using Apple Silicon GPUs
- Install Apple's new `mlx` framework ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)
- Install `JAX` with Apple's metal drivers (experimental is this point in time (2024-07), and not always up-to-date.) ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)
- Install `tensorflow` with and Apple's metal pluggable metal driver optimizations ![Optional](http://img.shields.io/badge/legacy-optional-brightgreen.svg?style=flat)
- Install `jupyter lab` to run notebooks
- Install `huggingface` and run some pre-trained language models using `transformers` and just a few lines of code within jupyter lab.

Then we provide additional HowTos for:

- Running large language models (LLMs) that rival commercial projects: Llama 2 or Llama 3 with llama.cpp (s.b.) using Mac Metal acceleration.

## Additional overview notes ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

(skip to **1. Preparations** if you know which framework you are going to use)

### What is Tensorflow vs. JAX vs. Pytorch vs. MLX and how relates Huggingface to it all?

Tensorflow, JAX, Pytorch, and MLX are deep-learning frameworks that provide the required libraries to perform optimized tensor operations used in training and inference. On high level, the functionality of all four is equivalent. Huggingface builds on top of any of the those frameworks and provides a large library of pretrained models for many different use-cases, ready to use or to customize plus a number of convenience libraries and sample code for easy getting-started.

- **Pytorch** is the most general and currently most widely used deep learning framework. In case of doubt, use Pytorch. It supports many different hardware platforms (including Apple Silicon optimizations).
- **JAX** is a newer Google framework that is considered especially by researchers as the better alternative to Tensorflow. It support GPUs, TPUs, and Apple's Metal framework (still experimental) and is more 'low-level', especially when used without complementary neural network-layers such as [flax](https://github.com/google/flax). JAX on Apple Silicon is still 'exotic', hence for production projects, use Pytorch, and for research projects, both JAX and MLX are interesting: MLX has more dynamic development (at this point in time), JAX supports more hardware framework (GPUs and TPUs) besides Apple Silicon, but development of the `jax-metal` drivers is not always up-to-date with the latest versions of `JAX`.
- **MLX** is Apple's new kid on the block, and thus overall support and documentation is (currently) much more limited than for the other main frameworks. It is beautiful and well designed (they took lessons learned for torch and tensorflow), yet it is closely tied to Apple Silicon. It's currently best for students that have Apple hardware and want to learn or experiment with deep learning. Things you learn with MLX easily transfer to Pytorch, yet be aware that conversion of models and porting of training and inference code is needed in order to deploy whatever you developed into the non-Apple universe.
- **corenet** is Apple's [newly released training library](https://github.com/apple/corenet) that utilizes PyTorch and the HuggingFace infrastructure, and additionally contains examples how to migrate models to MLX. See the example: [OpenElm (MLX)](https://github.com/apple/corenet/blob/main/mlx_examples/open_elm).
- **Tensorflow** is the 'COBOL' of deep learning and it's practically silently EoL'ed by Google. Google themselves publishes new models for PyTorch and JAX/Flax, and not for Tensorflow. If you are not forced to use Tensorflow, because your organisation already uses it, ignore it. If your organziation uses TF, make a migration plan! Look at Pytorch for production and JAX for research.

HuggingFace publishes an [Overview of model-support](https://huggingface.co/docs/transformers/index#supported-frameworks) for each framework. Currently, Pytorch is the defacto standard, if you want to make use of existing models.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) For the (probably too simplified) answer to the question "What's the fastest?" have a look at the Jupyter notebook [02-Benchmarks](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/02-Benchmarks.ipynb), and once you've completed the installation, you can test your own environment. The notebook allows to compare the speed of matrix multiplications for different frameworks. However, the difference between frameworks when performing 'standard' model training or inference tasks will most likely be less pronounced.

## 1. Preparations

### 1.1 Install homebrew

If you haven't done so, go to <https://brew.sh/> and follow the instructions to install homebrew.
Once done, open a terminal and type `brew --version` to check that it is installed correctly.

Now use `brew` to install more recent versions of `python` and `git`. The recommendation is to use Homebrew's default Python 3.12, if you are not planning to use Tensorflow with Metal optimization (still requires 3.11 (at 2024-04)).

#### Current Python for Huggingface, Pytorch, JAX, and MLX, Python 3.12, Homebrew default

```bash
brew install python@3.12 git
```

#### Legacy installations (Tensorflow), Python 3.11 ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

```bash
brew install python@3.11 git
```

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat)  you can install both versions of Python and then create a virtual environment using the specific python version you need for each case.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) If you plan to also use Linux, be aware that Python version support sometimes differs between Mac and Linux version of frameworks.

#### Make homebrew's Python the system-default ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) Apple does not put too much energy into keeping MacOS's python up-to-date. If you want to use an up-to-date default python, it makes sense to make homebrew's python the default system python.
So, if, you want to use homebrew's Python 3.11 or 3.12 system-globally, the easiest way
way to do so (after `brew install python@3.12` or `3.11`):

Edit `~/.zshrc` and insert:

```bash
# This is OPTIONAL and only required if you want to make homebrew's Python 3.12 as the global version:
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"                     
export PATH=/opt/homebrew/opt/python@3.12/libexec/bin:$PATH
```

Change all references of `3.12` to `3.11` when wanting to make homebrew's Python 3.11 system-standard python.

(Restart your terminal to activate the path changes, or enter `source ~/.zshrc` in your current terminal session.)

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) Regardless of the system python in use, when creating a virtual environment, you can always select the specific python version you want to use in the `venv` by creating the `venv` with exactly that python. E.g. `/usr/bin/python3 -m venv my_venv_name` creates a virtual environment using Apple's macOS python (which at the time of this writing, 2024-07, is still stuck at 3.9.6). See below, **Virtual environments**, for more details.

### 1.2 Test project

Now clone this project as a test project:

```bash
git clone https://github.com/domschl/HuggingFaceGuidedTourForMac
```

This clones the test-project into a directory `HuggingFaceGuidedTourForMac`

#### Virtual environment

Now create a Python 3.12 environment for this project and activate it:

(Again: replace with `3.11`, if you need)

```bash
python3.12 -m venv HuggingFaceGuidedTourForMac
```

Creating a venv adds the files required (python binaries, libraries, configs) for the virtual python environment to the project folder we just cloned, using again the same directory `HuggingFaceGuidedTourForMac`. Enter the directory and activate the virtual environment:

```bash
cd HuggingFaceGuidedTourForMac
source bin/activate
```

Now the directory `HuggingFaceGuidedTourForMac` contains the content of the github repository (e.g. `00-SystemCheck.ipynb`) _and_ the the files for the virtual env (e.g. `bin`, `lib`, `etc`, `include`, `share`, `pyvenv.cfg`):

![Folder content](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/Resources/ProjectFolder.png)

**Alternatives:** If you have many different python versions installed, you can create an environment that uses a specific version by specifying the path of the python that is used to create the `venv`, e.g.: 

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv my_new_312_env
```

uses homebrew's python explicitly to create a new `venv`, whereas

```bash
/usr/bin/python3 -m venv my_old_system_venv
```

would use Apple's macOS python version for the new environment.

### 1.3 When you done with your project

Do deactivate this virtual environment, simply use:

```bash
deactivate
```

To re-activate it, enter the directory that contains the `venv`, here: `HuggingFaceGuidedTourForMac` and use:

```bash
source bin/activate
```

### Additional notes on `venv` ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)
> ![Warning](http://img.shields.io/badge/⚠️-Warning:-orange.svg?style=flat) A very **unintuitive property of `venv`** is the fact: while you enter an environment by activating it in the subdirectory of your project (with `source bin/activate`), the `venv` **stays active** when you leave the project folder and start working on something completely different _until you explicitly deactivate_ the `venv` with `deactivate`. 
>
> There are a number of tools that modify the terminal system prompt to display the currently active `venv`, which is very helpful thing. Check out [starship](https://github.com/starship/starship) (recommended), or, if you like embellishment [`Oh My Zsh`](https://ohmyz.sh/).

![No venv active](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/Resources/no_venv.png)
_Example with `powerlevel10k` installed. The left side of the system prompt shows the current directory, the right side would show the name of the `venv`. Currently, no `venv` is active._

After activating a `venv` in `HuggingFaceGuidedTourForMac`:

![venv is still active](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/Resources/venv_remind.png)
_Even is the working directoy is changed (here to `home`), since the `venv` is still active, it's name is displayed on the right side by `powerlevel10k`. Very handy._

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) See <https://docs.python.org/3/tutorial/venv.html> for more information about Python virtual environments.

### 2 Install `pytorch`

Make sure that your virtual environment is active with `pip -V` (uppercase V), this should show a path for `pip` within your project:

`<your-path>/HuggingFaceGuidedTourForMac/lib/python3.12/site-packages/pip (python 3.12)`

Following `https://pytorch.org`, we will install Pytorch with `pip`. You need at least version 2.x (default since 2023) in order to get MPS (Metal Performance Shaders) support within pytorch, which offers significant performance advantage on Apple Silicon.

To install `pytorch` into the `venv`:

```bash
pip install -U torch numpy torchvision torchaudio
```

#### 2.1 Quick-test pytorch

To test that `pytorch` is installed correctly, and MPS metal performance shaders are available, open a terminal, type `python` and within the python shell, enter:

```python
import torch
# check if MPS is available:
torch.backends.mps.is_available()
```

This should return `True`.

### 3 Install `MLX` ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

```bash
pip install -U mlx
```

#### 3.1 Quick-test MLX

Again, start `python` and enter:

```python
import mlx.core as mx
print(mx.__version__)
```

This should print a version, such as `0.16.1` (2024-07)

- Visit the Apple [MLX project](https://github.com/ml-explore/) and especially [mlx-examples](https://github.com/ml-explore/mlx-examples)!
- There is a vibrant MLX community on Huggingface that has ported many nets to MLX: [Huggingface MLX-Community](https://huggingface.co/mlx-community)
- Apple's new [corenet](https://github.com/apple/corenet) utilizes PyTorch and the HuggingFace infrastructure, and additionally contains examples how to migrate models to MLX. See the example: [OpenElm (MLX)](https://github.com/apple/corenet/blob/main/mlx_examples/open_elm).

## 4.1 Install `JAX` ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

JAX is an excellent choice, if low-level optimization of algorithms and research beyond the boundaries of established deep-learning algorithms is your focus. Modelled after `numpy`, it supports [automatic differentiation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html) of 'everything' (for optimization problems) and supports [vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html) and [parallelization](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html) of python algorithms beyond mere deep learning. To get functionality that is expected from other deep learning frameworks (layers, training-loop functions and similar 'high-level'), consider installing additional neural network library such as: [`flax`](https://github.com/google/flax). 

### Check supported versions

Unfortunately, the `JAX` metal drivers have started to lag behind JAX releases, and therefore you need to check the [compatibility table](https://pypi.org/project/jax-metal/) for the supported versions of `JAX` that match the available `jax-metal` drivers.

To install a specific version of `JAX` and the latest `jax-metal` with `pip` into the active environment:

```bash
# The version 0.4.26 is taken from the compatibility table mentioned above. Update as required.
pip install -U jax==0.4.26 jaxlib==0.4.26 jax-metal
```

#### 4.2 Quick-test JAX

Start `python` (3.12 is supported) and enter:

```python
import jax
print(jax.devices()[0])
```

This should display (on first run only):

```
Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1721975334.430133   43061 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
Metal device set to: Apple M2 Max

systemMemory: 32.00 GB
maxCacheSize: 10.67 GB

I0000 00:00:1721975334.446739   43061 service.cc:145] XLA service 0x60000031d100 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1721975334.446771   43061 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
I0000 00:00:1721975334.448269   43061 mps_client.cc:406] Using Simple allocator.
I0000 00:00:1721975334.448308   43061 mps_client.cc:384] XLA backend will use up to 22906109952 bytes on device 0 for SimpleAllocator.
[METAL(id=0)]
```

Here `METAL:0` is the device that JAX will use for calculations, and Apple Silicon is supported.

##### Errors

If, instead you see errors like:

```
RuntimeError: Unable to initialize backend 'METAL': INVALID_ARGUMENT: Mismatched PJRT plugin PJRT API version (0.47) and framework PJRT API version 0.54). (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)
```

Your version of `jax` and `jaxlib` are incompatible with `jax-metal`. Check the [compatibility table](https://pypi.org/project/jax-metal/) for `jax-metal` and install the required versions as indicated in the table. 

- [HuggingFace example projects with JAX and Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)
- Apple's rather terse documentation is found at [Apple's JAX documentation](https://developer.apple.com/metal/jax/).

## 4.3 Install `tensorflow` ![Optional](http://img.shields.io/badge/legacy-optional-brightgreen.svg?style=flat)

> ![Warning:](http://img.shields.io/badge/⚠-Note:-orange.svg?style=flat) Tensorflow is losing support fast, and not even Google publishes new models for Tensorflow. A migration plan is recommended, if you plan to use this.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) While Tensorflow supports Python 3.12 since 2.16, the macOS `tensorflow-metal` accelerator has not been updated since 2023-09 (status of 2024-07) and requires Python 3.11:

Make sure that your virtual environment is active with `pip -V` (uppercase V), this should show a path for `pip` within your project:

`<your-path>/HuggingFaceGuidedTourForMac/lib/python3.11/site-packages/pip (python 3.11)`

Following <https://developer.apple.com/metal/tensorflow-plugin/>, we will install `tensorflow` with `pip` within our `venv`:

```bash
pip install -U tensorflow tensorflow-metal
```

#### 4.4 Quick-test Tensorflow

To test that `tensorflow` is installed correctly, open a terminal, type `python` and within the python shell, enter:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

You should see something like:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 5 Jupyter lab

At this point, your Apple Silicon Mac should be ready to run `pytorch` and optionally `MLX` and/or `JAX` or `tensorflow` with hardware acceleration support, using the Apple Metal framework.

To test this, you can use `jupyter lab` to run some notebooks. To install `jupyter lab`, first make sure the virtual environment you want to use is active (`pip -V`), and type:

```bash
pip install -U jupyterlab ipywidgets
```

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) If you have other Jupyter versions installed, the path to the newly installed jupyter version within the `venv` is often not updated correctly, re-activate the environment to make sure that the correct local Jupyter version is used:

```bash
deactivate
source bin/activate
```

To start Jupyter lab, type:

```bash
jupyter lab
```

This should open a browser window with `jupyter lab` running. You can then create a new python notebook and run some code to test that `tensorflow` and `pytorch` are working correctly:

![](Resources/jupyterlab.png)

```python
import torch

print("Pytorch version:", torch.__version__)
```

If this completed successful, your Mac is now ready for Deep Learning experiments.

## 6 HuggingFace

HuggingFace is a great resource for NLP and Deep Learning experiments. It provides a large number of pre-trained language models and a simple API to use them. It will allow us to quickly get started with Deep Learning experiments.

### 6.1 Install `transformers`

From the [huggingface installation instructions](https://huggingface.co/docs/transformers/installation), we use `pip` to install `transformers`:

```bash
pip install -U transformers accelerate "huggingface_hub[cli]"
```

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) When experimenting with HuggingFace, you will download large models that will be stored in your home directory at: `~/.cache/huggingface/hub`. 
> You can remove these models at any time by deleting this directory or parts of it's content.

- `accelerate` is optional, but used to run some large models. Side-effect of installing `accelerate` might be downgrade of some other modules like `numpy`.
- `"huggingface_hub[cli]"` installs the huggingface command line tools that are sometimes required to download (partially proprietarilly licensed) models like Llama 3.

## 7 Experiments

### 7.1 Simple sentiment analysis

Within the directory `HuggingFaceGuidedTourForMac` and active `venv`, start `jupyter lab` and load the `00-SystemCheck.ipynb` notebook.  The notebook will first check all the deep-learning frameworks and give information, if they are correctly installed. Afterward, Pytorch is used for a simple experiment.

Use `<Shift>-Enter` to run the notebook's cells.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) If you started Jupyter Lab before installing Huggingface, you either need to restart the python kernel in Jupyter or simply restart Jupyter Lab, otherwise it won't find the Transformers library.

After the various tests, your should finally see something like this:

![](Resources/huggingface-transformers.png)

If you've received a label classification of `POSITIVE` with a score of `0.99`, then you are ready to start experimenting with HuggingFace!

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) You'll see that the `HuggingFace` libraries are downloading all sorts of large binary blobs containing the trained model data. That data is stored in your home directory at: `~/.cache/huggingface/hub`. You can remove these models at any time by deleting this directory or parts of it's content.

#### Trouble-shooting

- If self-tests fail ('xyz not found!'), make sure that pytorch, jax (optional), MLX (optional), tensorflow (optional), jupyter, and transformers by huggingface are all installed into the same, active Python virtual environment, otherwise the components won't 'see' each other!

### 7.2 Minimal chat-bot

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

- Your Mac can run large language models that rival the performance of commercial solutions. An excellent example is the [`llama.cpp` project](https://github.com/ggerganov/llama.cpp) that implements the inference code necessary to run LLMs in highly optimized C++ code, supporting the Mac's Metal acceleration.<br>A step-by-step guide to compile and run Llama 3 or Llama 2 first for benchmarking and then for chat can be found here:<br>[Llama.cpp chat using the Llama 2 model, with first Llama 3 support](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/NextSteps/llama.cpp.md). Additionally a first version for Llama 3 is provided.

## Learning resources

- One of the (currently) best sources for information about new releases of models on Huggingface is [LocalLLama reddit group](https://old.reddit.com/r/LocalLLaMA/).
- The fast-track to learning how neural network and specifically large languages models actually work, is Andrej Karpathy's course on Youtube: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). If you know some python and how to multiply a matrix with numpy, this is the course that takes you all the way to being able to build your own Large-language model from scratch.

## Conda uninstallation notes

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) This paragraph is to uninstall conda that was used in older versions of this guide:

`brew uninstall miniconda`

Additional modifications are (all of them are inactive, once miniconda is removed):

- `~/.condarc` (list of channels), and `~/.conda\`.
- `~/.zshrc` (or `.bashrc`) for the setup of path and environment.
- After using hugginface models, large model binary blobs may reside at: `~/.cache/huggingface/hub`. Simply remove the directory.

## Changes

- 2024-09-10: Version updates for the platforms.
- 2024-07-26: Version updates for the platforms.
- 2024-04-28: Added JAX installation with Metal support and quick-test.
- 2024-04-26: Apple's corenet
- 2024-04-22: Llama 3.
- 2024-02-24: (Guide version 3.0) Updates for Python 3.12 and Apple MLX framework, Tensorflow is legacy-option.
- 2023-12-14: Pin python version of homebrew to 3.11.
- 2023-10-30: Restested with macOS 14.1 Sonoma, Tensorflow 2.14, Pytorch 2.1. Next steps added for more advanced projects.
- 2023-09-25: (Guide version 2.0) Switched from `conda` to `pip` and `venv` for latest versions of tensorflow 2.13, Pytorch 2, macOS Sonoma, installation is now much simpler.
- 2023-03-16: Since `pytorch` v2.0 is now released, the channel `pytorch-nightly` can now be replaced by `pytorch` in the installation instructions. The `pytorch-nightly` channel is no longer needed for MPS support.
