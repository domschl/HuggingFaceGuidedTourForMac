## Using Metal acceleration with [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and high quality chat models like llama s

### Preparations

This project is independent of Python, Jupyter, Tensorflow, Pytorch, and Huggingface. We use the `llama.cpp` project to access the Mac's accelerator Hardware directly via metal. First, let's install some tools useful for C++ development:

> **Note:** This installation is optional, since the core-tools (compiler and make) are already installed, if Homebrew is installed, or as part of macOS. This simply installs newer versions and optional compilers and build-tools:

```bash
brew install llvm gcc make cmake
```

In order to download large models from Huggingface, we need larg-file-support (LFS) for GIT:

```bash
brew install git-lfs
git lfs install
```

You should see: 'Git LFS initialized'.

## Compilation of `llama.cpp`

Clone the `llama.cpp` project with:

```bash
git clone https://github.com/ggerganov/llama.cpp
```

Enter the project directory:

```bash
cd llama.cpp
```

and compile the project:

```bash
make
```

> **Notes:** You can speed up the build process by specifiying the number of cores used for compilation `make -j4` would use 4 cores. To retry a build, use `make clean` first to remove existing build artifacts.

## Downloading model data

> **Warning**: The entire situation with copyright and moral right to use large language models that have mostly been created by using vast undisclosed datasources is unclear at best. The underlying training data-collections have been created by a vast number of different authors and have been used without explicit consent in almost all cases. That is the reason why the actual source of the model data and it's origins is obfuscated in many cases, even by large cooperations offering commercial services. 
> Here we try to use the models that at least try to be open about their provenance, but it is important to consider that even those models should be considered as having an undefined state concerning copyright and moral status.

The by far largest repository for large-language-models (LLMs) is Huggingface. There are basically two ways to download a model:

- Clone a Huggingface model repository using Git LFS (as installed above)
- Download a single model file directly

### Llama 2

Llama 2 is a large language model released by Facebook / Meta. A version converted
into the GGUF format (currently) been used by `llama.cpp` can be optained from:

- <https://huggingface.co/TheBloke/Llama-2-7B-GGUF>

Enter the tab 'Files and versions' and look for the file `llama-2-7b.Q4_0.gguf`. Click the download icon to right of the filename:

![Folder content](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/NextSteps/Resources/llama-model.png)

> **Note:** Alternatively, you can download the entire model project with all models using git lfs with: `git clone https://huggingface.co/TheBloke/Llama-2-7B-GGUF`.

From within the directory `llama.cpp`:

```bash
mkdir models/7B
mv ~/Downloads/llama-2-7b.Q4_0.gguf models/7B
```

As a first test, use the benchmark tool (built in directory `llama.cpp`):

```bash
./llama-bench -m models/7B/llama-2-7b.Q4_0.gguf
```

Some results:

| machine | model | size | params | backend | ngl | test | t/s |
| ------- | ----- | ---: | -----: | ------- | --: | ---- | --: |
| Mac mini M1 | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal | 99 | pp 512 | 114.39 ± 1.64 |
| Mac mini M1 | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal | 99 | tg 128 | 14.17 ± 0.08 |
| Macbook Pro M2 Max | llama 7B mostly Q4_0  | 3.56 GiB | 6.74 B | Metal | 99 | pp 512 | 535.90 ± 0.26 |
| Macbook Pro M2 Max | llama 7B mostly Q4_0  | 3.56 GiB | 6.74 B | Metal | 99 | tg 128 | 58.80 ± 3.11 |
