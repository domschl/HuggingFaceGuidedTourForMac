![Version](http://img.shields.io/badge/Version-4-blue.svg?style=flat)

# HuggingFace and Deep Learning guided tour for Macs with Apple Silicon

A guided tour on how to install optimized `pytorch` and optionally Apple's new `MLX` and Google's `JAX` on Apple Silicon Macs and how to use `HuggingFace` large language models for your own experiments. Apple Silicon Macs show good performance for many machine learning tasks.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) This guide was updated to **Version 4**: main change is the usage of the [`uv`](https://docs.astral.sh/uv/) package manager. The previous version 3 that uses Python's standard `pip` and package manager and explicit `venv` management, is still available [here](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/LegacyGuides/README_v3.md).

We will perform the following steps:

- Install `homebrew` 
- Install `pytorch` with MPS (metal performance shaders) support using Apple Silicon GPUs
- Install Apple's new `mlx` framework ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)
- Install `JAX` with Apple's metal drivers (experimental is this point in time (2025-06), and not up-to-date.) ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)
- Install `tensorflow` with and Apple's metal pluggable metal driver optimizations ![Optional](http://img.shields.io/badge/legacy-optional-brightgreen.svg?style=flat)
- Install `jupyter lab` to run notebooks
- Install `huggingface` and run some pre-trained language models using `transformers` and just a few lines of code within jupyter lab for simple chat bot.

## Additional overview notes ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

(skip to **1. Preparations** if you know which framework you are going to use)

### What is Tensorflow vs. JAX vs. Pytorch vs. MLX and how relates Huggingface to it all?

Tensorflow, JAX, Pytorch, and MLX are deep-learning frameworks that provide the required libraries to perform optimized tensor operations used in training and inference. On high level, the functionality of all four is equivalent. Huggingface builds on top of any of the those frameworks and provides a large library of pretrained models for many different use-cases, ready to use or to customize plus a number of convenience libraries and sample code for easy getting-started.

- **Pytorch** is the most general and currently most widely used deep learning framework. In case of doubt, use Pytorch. It supports many different hardware platforms (including Apple Silicon optimizations).
- **JAX** is a newer Google framework that is considered especially by researchers as the better alternative to Tensorflow. It support GPUs, TPUs, and Apple's Metal framework (still experimental) and is more 'low-level', especially when used without complementary neural network-layers such as [flax](https://github.com/google/flax). JAX on Apple Silicon is still 'exotic', hence for production projects, use Pytorch, and for research projects, both JAX and MLX are interesting: MLX has more dynamic development (at this point in time), JAX supports more hardware framework (GPUs and TPUs) besides Apple Silicon, but development of the Apple `jax-metal` drivers is often not up-to-date with the latest versions of `JAX` and requires the use of old `JAX` versions. (s.b.)
- **MLX** is Apple's new kid on the block, and thus overall support and documentation is (currently) much more limited than for the other main frameworks. It is beautiful and well designed (they took lessons learned for torch and tensorflow), yet it is closely tied to Apple Silicon. It's currently best for students that have Apple hardware and want to learn or experiment with deep learning. Things you learn with MLX easily transfer to Pytorch, yet be aware that conversion of models and porting of training and inference code might be necessary in order to deploy whatever you developed into the non-Apple universe. **Update:** Support for CUDA (and possibly AMD) is [under development](https://github.com/ml-explore/mlx/pull/1983).
- **corenet** is Apple's [training library](https://github.com/apple/corenet) that utilizes PyTorch and the HuggingFace infrastructure, and additionally contains examples how to migrate models to MLX. See the example: [OpenElm (MLX)](https://github.com/apple/corenet/blob/main/mlx_examples/open_elm).
- **Tensorflow** is the 'COBOL' of deep learning and it's practically silently EoL'ed by Google. Google themselves publishes new models for PyTorch and JAX/Flax, and not for Tensorflow. If you are not forced to use Tensorflow, because your organisation already uses it, ignore it. If your organziation uses TF, make a migration plan! Look at Pytorch for production and JAX for research. Another reason to still look into Tensorflow are embedded applications and Tensorflow's C-library.

HuggingFace publishes an [Overview of model-support](https://huggingface.co/docs/transformers/index#supported-frameworks) for each framework. Currently, Pytorch is the defacto standard, if you want to make use of existing models.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) For the (probably too simplified) answer to the question "What's the fastest?" have a look at the Jupyter notebook [02-Benchmarks](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/02-Benchmarks.ipynb), and once you've completed the installation, you can test your own environment. The notebook allows to compare the speed of matrix multiplications for different frameworks. However, the difference between frameworks when performing 'standard' model training or inference tasks will most likely be less pronounced.

## 1. Preparations

### 1.1 Install homebrew

If you haven't done so, go to <https://brew.sh/> and follow the instructions to install homebrew.
Once done, open a terminal and type `brew --version` to check that it is installed correctly.

Now use `brew` to install more recent versions of `python`, `uv`, and `git`. The recommendation is to use Homebrew's Python 3.12, because that is currently the Python version that allows installation of _all_ frameworks. The roadblock for version 3.13 is Tensorflow. Below, we'll install each framework separately using updated Python versions.

#### Current Python for Huggingface, Pytorch, JAX, and MLX, Python 3.13, Homebrew default

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) Use `python@3.12`, if you care about Tensorflow.

```bash
brew install python@3.13 uv git
```

#### Make homebrew's Python the system-default ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) Apple does not put too much energy into keeping MacOS's python up-to-date. If you want to use an up-to-date default python, it makes sense to make homebrew's python the default system python.
So, if, you want to use homebrew's Python 3.12 or 3.13 in Terminal, the easiest way way to do so (after `brew install python@3.13`):

Edit `~/.zshrc` and insert (again use `python@3.12`, if you care about Tensorflow):

```bash
# This is OPTIONAL and only required if you want to make homebrew's Python 3.13 as the global version:
export PATH="/opt/homebrew/opt/python@3.13/bin:$PATH"                     
export PATH=/opt/homebrew/opt/python@3.13/libexec/bin:$PATH
```

(Restart your terminal to activate the path changes, or enter `source ~/.zshrc` in your current terminal session.)

### 1.2 Test project

Now clone this project as a test project:

```bash
git clone https://github.com/domschl/HuggingFaceGuidedTourForMac
```

This clones the test-project into a directory `HuggingFaceGuidedTourForMac`

#### Virtual environment using `uv`

Now execute:

```bash
cd HuggingFaceGuidedTourForMac
uv sync
source .venv/bin/activate
```

This will install a virtual environment at `HuggingFaceGuidedTourForMac/.venv` using the python version defined in the project's [`.python-version`](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/.python-version) file and install the dependencies defined in [`pyproject.toml`](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/pyproject.toml) and finally activate that environment. Have a look at each of those locations and files to get an understanding what `uv sync` installed. Check out `uv` [documentation](https://docs.astral.sh/uv/) for general information on `uv`.

You have now a virtual environment with _all_ of the mentioned deep learning frameworks installed. (Look at [pyproject.toml](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/pyproject.toml) for the installed versions.) This is only useful for a first overview, and below we will install each separately and step-by-step.

### First test

Execute:

```bash
uv run jupyter lab 00-SystemCheck.ipynb
```

This will open a jupyter notebook that will test each of the installed frameworks. Use `shift-enter` to execute each notebook cell and verify that all tests complete successfully.

### Additional notes on `venv` ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

> ![Warning](http://img.shields.io/badge/⚠️-Warning:-orange.svg?style=flat) A very **unintuitive property of virtual environments** is the fact: while you enter an environment by activating it in the subdirectory of your project (with `.venv/bin/activate` or `uv sync`), the `venv` **stays active** when you leave the project folder and start working on something completely different _until you explicitly deactivate_ the `venv` with `deactivate`:

```bash
deactivate
```

> There are a number of tools that modify the terminal system prompt to display the currently active `venv`, which is very helpful. Check out [starship](https://github.com/starship/starship) (recommended). Once `starship` is active, your terminal prompt will show the active Python version and the name of the virtual environment.

### 2. A fresh `pytorch` project

We will now perform a step-by-step installation for a new `pytorch` project. Check out <[https://pytorch.org](https://pytorch.org/get-started/locally/)>, but here, we will install Pytorch with `uv`.

Create a new directory for your test project and install Pytorch using the latest Python version:

```bash
mkdir torch_test
cd torch_test
uv init --python 3.13
uv venv
uv add torch numpy
source .venv/bin/activate
```

This: creates a new project directory, enters it, initializes a new project using Python 3.13 (which is support with Apple Metal acceleration)
and installs a `torch` and `numpy` in a new virtual environment.

We can now start python and enter a short test sequence to verify everything works:

```
python
Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 17.0.0 (clang-1700.0.13.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.backends.mps.is_available()
True
>>>
```

If you see `True` as answer on `torch.backends.mps.is_available()`, Metal acceleration is working ok.

#### 2.1 Alternative: use an Editor, e.g. Visual Studio Code:

Enter `code main.py`. This will open the default 'Hello, world' created by `uv`.

Change the code to:

```python
import torch

def main():
    print("Hello from torch-test!")
    if torch.backends.mps.is_available():
        print("Excellent! MPS backend is available.")
    else:
        print("MPS backend is not available: Something went wrong! Are you running this on a Mac with Apple Silicon chip?")

if __name__ == "__main__":
    main()
```

Save and exit the editor and run with:

```bash
uv run main.py
```

Deactivate this environment with `deactivate`, re-activate it with `source .venv/bin/activate`.

### 3. A fresh `MLX` project ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

In similar fashion, we create now a new MLX project:

Create a new directory for your test project and install pytorch using the latest Python version:

```bash
mkdir mlx_test
cd mlx_test
uv init --python 3.13
uv venv
uv add mlx
source .venv/bin/activate
```

Again, start `python` and enter:

```python
import mlx.core as mx
print(mx.__version__)
```

This should print a version, such as `0.26.1` (2025-06)

- Visit the Apple [MLX project](https://github.com/ml-explore/) and especially [mlx-examples](https://github.com/ml-explore/mlx-examples)!
- There is a vibrant MLX community on Huggingface that has ported many nets to MLX: [Huggingface MLX-Community](https://huggingface.co/mlx-community)
- Apple's new [corenet](https://github.com/apple/corenet) utilizes PyTorch and the HuggingFace infrastructure, and additionally contains examples how to migrate models to MLX. See the example: [OpenElm (MLX)](https://github.com/apple/corenet/blob/main/mlx_examples/open_elm).

Deactivate with `deactivate`.

## 4. A `JAX` project ![Optional](http://img.shields.io/badge/optional-brightgreen.svg?style=flat)

JAX is an excellent choice, if low-level optimization of algorithms and research beyond the boundaries of established deep-learning algorithms is your focus. Modelled after `numpy`, it supports [automatic differentiation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html) of 'everything' (for optimization problems) and supports [vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html) and [parallelization](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html) of python algorithms beyond mere deep learning. To get functionality that is expected from other deep learning frameworks (layers, training-loop functions and similar 'high-level'), consider installing additional neural network library such as: [`Flax NNX`](https://github.com/google/flax). 

Unfortunately, the JAX metal drivers have started to lag behind JAX releases, and therefore you need to check the compatibility table for the supported versions of JAX that match the available jax-metal drivers. Currently, Jax is therefore pinned to the outdated version 0.4.34. Check for new `jax-metal` releases for updates, [compatibility table](https://pypi.org/project/jax-metal/).

```bash
mkdir jax_test
cd jax_test
uv init --python 3.13
uv venv
uv add jax==0.4.34 jax-metal
source .venv/bin/activate
```

Start `python` and enter:

```python
import jax
print(jax.devices()[0])
```

This should output something like:

```
Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 17.0.0 (clang-1700.0.13.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
... print(jax.devices()[0])
...
Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1750594202.639458 5239992 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
Metal device set to: Apple M2 Max

systemMemory: 32.00 GB
maxCacheSize: 10.67 GB

I0000 00:00:1750594202.655851 5239992 service.cc:145] XLA service 0x600002f0c500 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1750594202.655998 5239992 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
I0000 00:00:1750594202.657356 5239992 mps_client.cc:406] Using Simple allocator.
I0000 00:00:1750594202.657365 5239992 mps_client.cc:384] XLA backend will use up to 22906109952 bytes on device 0 for SimpleAllocator.
METAL:0
```

#### 4.1 Check supported versions for JAX and Metal:

Note: `uv` does a good job in resolving version dependencies between JAX and the required metal drivers. If you plan to use `pip`, you will need to manually verify version compliance:

Check the [compatibility table](https://pypi.org/project/jax-metal/) for the supported versions of `JAX` that match the available `jax-metal` drivers.

## 5. A fresh `tensorflow` project ![Optional](http://img.shields.io/badge/legacy-optional-brightgreen.svg?style=flat)

> ![Warning:](http://img.shields.io/badge/⚠-Note:-orange.svg?style=flat) Tensorflow is losing support fast, and not even Google publishes new models for Tensorflow. A migration plan is recommended, if you plan to use this.

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) Tensorflow supports Python 3.12 and the metal drivers are only available for 3.12.

```bash
mkdir tensorflow_test
cd tensorflow_test
uv init --python 3.12
uv venv
uv add tensorflow tensorflow-metal
source .venv/bin/activate
```
 
To test that `tensorflow` is installed correctly, open a terminal, type `python` and within the python shell, enter:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

You should see a GPU-type device:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 6. Putting it all together: Jupyter lab, Huggingface, Pytorch, and a first chat-bot experiment

Let's now create a full project and do a first experiment by implementing a chat-bot in a Jupyter notebook:

```bash
mkdir chat_test
cd chat_test
uv init --python 3.13
uv venv
uv add torch numpy jupyterlab ipywidgets transformers accelerate "huggingface_hub[cli]"
source .venv/bin/activate
```

To start Jupyter lab, type:

```bash
jupyter lab
```

Either copy the notebook `01-ChatBot.ipynb` into your project `chat_test`, or enter the code into a new notebook:

```python
import torch
from transformers import pipeline

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
pipeline = pipeline(task="text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")  # or request device_map="mps"
chat = [
    {"role": "system", "content": "You are a super-intelligent assistant"},
]
first = True
while True:
    if first is True:
        first = False
        print("Please press enter (not SHIFT-enter) after your input, enter 'bye' to end:")
    try:
        input_text = input("> ")
        if input_text in ["", "bye", "quit", "exit"]:
            break
        print()
        chat.append({"role": "user", "content": input_text})
        response = pipeline(chat, max_new_tokens=512)
        print(response[0]["generated_text"][-1]["content"])
        chat = response[0]["generated_text"]
        print()
    except KeyboardInterrupt:
        break
```

That all that is required to build a simple chat-bot with dialog history.

Try to:

- Change the chat model to larger versions: "Qwen/Qwen2.5-3B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"
- Check out <https://huggingface.co/models?pipeline_tag=text-generation&sort=trending> for the latest chat models!

> ![Note:](http://img.shields.io/badge/📝-Note:-green.svg?style=flat) When experimenting with HuggingFace, you will download large models that will be stored in your home directory at: `~/.cache/huggingface/hub`. 
> You can remove these models at any time by deleting this directory or parts of it's content.

- `"huggingface_hub[cli]"` installs the huggingface command line tools that are sometimes required to download (proprietary licensed) models.

## 7. Further Experiments

## Learning resources

- The fast-track to learning how neural network and specifically large languages models actually work, is Andrej Karpathy's course on Youtube: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). If you know some python and how to multiply a matrix with numpy, this is the course that takes you all the way to being able to build your own Large-language model from scratch.

## Changes

- 2025-06-22: (Guide version 4): Version updates and usage of `uv` package manager. Old v3 version using `pip` available at [v3](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/LegacyGuides/README_v3.md).
- 2024-09-10: Version updates for the platforms.
- 2024-07-26: Version updates for the platforms.
- 2024-04-28: Added JAX installation with Metal support and quick-test.
- 2024-04-26: Apple's corenet
- 2024-04-22: Llama 3.
- 2024-02-24: (Guide version 3.0) Updates for Python 3.12 and Apple MLX framework, Tensorflow is legacy-option.
- 2023-12-14: Pin python version of homebrew to 3.11.
- 2023-10-30: Re-tested with macOS 14.1 Sonoma, Tensorflow 2.14, Pytorch 2.1. Next steps added for more advanced projects.
- 2023-09-25: (Guide version 2.0) Switched from `conda` to `pip` and `venv` for latest versions of tensorflow 2.13, Pytorch 2, macOS Sonoma, installation is now much simpler.
- 2023-03-16: Since `pytorch` v2.0 is now released, the channel `pytorch-nightly` can now be replaced by `pytorch` in the installation instructions. The `pytorch-nightly` channel is no longer needed for MPS support.
