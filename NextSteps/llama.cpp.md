## Using Metal acceleration with [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and high-quality chat models such as Llama 2 and Llama 3

This project is independent of Python, Jupyter, Tensorflow, and Pytorch. We use Huggingface's site as source for models, and the `llama.cpp` project to access the Mac's accelerator Hardware directly via metal. 

## Installation

### via homebrew

Since `llama.cpp` is available via homebrew, and its version is updated regularly, it's no longer necessary to compile llama yourself:

```bash
brew install llama.cpp
```

### Alternative: compile yourself

#### Preparation and tools

First, let's install some tools useful for C++ development:

> **Note:** This installation is optional, since the core-tools (compiler and make) are already installed, if Homebrew is installed, or as part of macOS. This simply installs newer versions and optional compilers and build tools:

```bash
brew install llvm gcc make cmake ccache
```

In order to download large model projects from Huggingface, we need large-file-support (LFS) for GIT:

```bash
brew install git-lfs
git lfs install
```

You should see: 'Git LFS initialized'.

#### Compilation of `llama.cpp`

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

> **Notes:** You can speed up the build process by specifying the number of cores used for compilation `make -j4` would use 4 cores. To retry a build, use `make clean` first to remove existing build artefacts.

## Downloading model data

> **Warning**: The entire situation with copyright and moral right to use large language models that have mostly been created by using vast undisclosed data sources is unclear at best. The underlying training data-collections have been created by a vast number of different authors and have been used without explicit consent in almost all cases. That is the reason why the actual source of the model data and its origins is obfuscated in many cases, even by large corporations offering commercial services. 
> Here we try to use the models that at least try to be open about their provenance, but it is important to consider that even those models should be considered as having an undefined state concerning copyright and moral status.

The by far largest repository for large-language models (LLMs) is Huggingface. There are basically two ways to download a model:

- Clone a Huggingface model repository using Git LFS (as installed above)
- Download a single model file directly

### Llama 3

(**Note:** (2024-04-22) this is rather recent, so optimal model and instructions might change rapidly)

Update the `llama.cpp` project, code must be newer than 2024-04-22 for Llama 3 support, then compile again and:

- Download model: from <https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF>, the recommended model is <https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf>, simply download this file.
- Copy the model file into the `models` directory
- Run with:

```bash
./main -m models/Meta-Llama-3-8B-Instruct-Q6_K.gguf -p "How to build a simple website that displays 'Hello, world!'?"
```

Output is something like:

----------

> How to build a simple website that displays 'Hello, world!'?](https://stackoverflow.com/questions/1434473/how-to-build-a-simple-web-site-that-displays-hello-world)
>
> Here is a simple example using HTML, CSS, and JavaScript:

> **index.html**
```
<!DOCTYPE html>
<html>
<head>
  <title>Hello, World!</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1 id="hello-world">Hello, World!</h1>
  <script src="script.js"></script>
</body>
</html>
```

> **script.js**
```
document.getElementById("hello-world").innerHTML = "Hello, World!";
```

> **How it works:**
>
1. We create an `index.html` file with a basic HTML structure.
2. We add a `<style>` block to set the font family and text alignment for the page.
3. We add an `<h1>` element with an ID of "hello-world" to display the message.
4. We link to a `script.js` file using the `<script>` tag.
5. In the `script.js` file, we use the `document.getElementById` method to get a reference to the `<h1>` element with the ID "hello-world".
6. We set the `innerHTML` property of the element to the string "Hello, World!".
>
> **Run it:**
> 
1. Save both files in the same directory.
2. Open `index.html` in a web browser.
3. The page should display "Hello, World!" in the center of the page.
>
> That's it! This is a very basic example, but it should give you an idea of how to create a simple web page that displays a message. Of course, in a real-world scenario, you would likely want to add more functionality and styling to your page.<|eot_id|> [end of text]

----------

### Llama 2

Llama 2 is a large language model released by Facebook / Meta. A version converted
into the GGUF format (currently) being used by `llama.cpp` can be obtained from:

- <https://huggingface.co/TheBloke/Llama-2-7B-GGUF>

Enter the tab 'Files and versions' and look for the file `llama-2-7b.Q4_0.gguf`. Click the download icon to the right of the filename:

![Huggingface model download](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/NextSteps/Resources/llama-model.png)

> **Note:** Alternatively, you can download the entire model project with all models using git lfs with: `git clone https://huggingface.co/TheBloke/Llama-2-7B-GGUF`.

From within the directory `llama.cpp`:

```bash
mkdir models/7B
mv ~/Downloads/llama-2-7b.Q4_0.gguf models/7B
```

#### Some benchmarks

As a first test, use the [benchmark tool](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench) (built-in directory `llama.cpp`):

```bash
./llama-bench -m models/7B/llama-2-7b.Q4_0.gguf
```

Some results:

| machine                                      | model                |     size | params | backend | ngl | test   |           t/s |
|----------------------------------------------|----------------------|---------:|-------:|---------|----:|--------|--------------:|
| Mac mini M1, 16 GB, 8 GPU                    | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal   |  99 | pp 512 | 114.39 ± 1.64 |
| Mac mini M1                                  | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal   |  99 | tg 128 |  14.17 ± 0.08 |
|                                              |                      |          |        |         |     |        |               |
| Macbook Pro M2 Max, 32 GB, 8/4 cores, 30 GPU | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal   |  99 | pp 512 | 535.90 ± 0.26 |
| Macbook Pro M2 Max, 32 GB, 8/4 cores, 30 GPU | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | Metal   |  99 | tg 128 |  58.80 ± 3.11 |
|                                              |                      |          |        |         |     |        |               |
| Intel® Core™ i5-13500                        | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | CPU     |  10 | pp 512 |  16.23 ± 0.06 |
| Intel® Core™ i5-13500                        | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | CPU     |  10 | tg 128 |  13.54 ± 0.02 |
|                                              |                      |          |        |         |     |        |               |
| Nvidia GTX 1080 Ti                           | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | CUDA    |  99 | pp 512 | 371.01 ± 2.15 |
| Nvidia GTX 1080 Ti                           | llama 7B mostly Q4_0 | 3.56 GiB | 6.74 B | CUDA    |  99 | tg 128 |  39.14 ± 0.06 |

Note that it is difficult to compare benchmarks with [output from the project](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench), since the exact provenance of the used model is undefined.

#### Chat with Llama 2

The official documentation for the `main` program used for chatting is at:

- <https://github.com/ggerganov/llama.cpp/tree/master/examples/main>

The most important part, of getting a model into 'chat-mode', is an appropriate prompt. We follow the project's example and insert our model-file for Llama 2:

In directory `llama.cpp` execute:

```bash
./main -m models/7B/llama-2-7b.Q4_0.gguf  -n -1 --color -r "User:" --in-prefix " " -i -p \
'User: Hi
AI: Hello. I am an AI chatbot. Would you like to talk?
User: Sure!
AI: What would you like to talk about?
User:'
```

A typical dialogue looks like this:

![Huggingface model download](https://github.com/domschl/HuggingFaceGuidedTourForMac/blob/main/NextSteps/Resources/llama2-chat.png)

## References

- The [`llama.cpp`](https://github.com/ggerganov/llama.cpp) project
- A large collection of models by [TheBloke on Huggingface](https://huggingface.co/TheBloke). This contains ready-to-run models for the llama.cpp and other formats, and is usually up-to-date with the latest and greatest models.
