<h2>1. Introduction</h2>
In my previous articles, we saw [how to create a simple Telegram bot][1] and [how to build a responsive assistant][2]: in both of these cases, we built something that relied on pre-defined responses, no matter how natural language-like they might have seemed. 

The question is now: can we create something even more **customizable** and **helpful**, that does not contain standardized answers, but that builds its replies on the **context** given by the user and upon their **prompts**?

In this tutorial we will address this question, building an **AI-powered** and **context-aware** Telegram bot. Before we start, nevertheless, we need to define some important terms that we will be using in this tutorial, and that are not-so-common in everyday programming.

<h2>2. Definitions</h2>
<h3>2a. LLM</h3>
 LLM stands for **Large Language Model**: it is an Artificial Intelligence-based model able to understand, process and produce natural language, performing also complex tasks. Example of modern LLMs are GPTs (*Generative Pre-trained Transformers*), the underlying technology behind ChatGPT, which have been proving as the most capable of modelling human language. For this tutorial, we will be using **[Phi-3-128K instruct][3]**, which is one of the most recent and powerful models, released by Microsoft.

I wrote about LLMs architecture in [a post on my personal blog][4]
<h3>2b. Vector Databases</h3>
A vector database is a **non-traditional** data storage facility and can be used to represent **complex data** (with lots of features) based on a set of **multi-dimensional numerical objects** (*vectors*). In this sense, we can represent the information contained in long texts, images, videos or other data with numbers, without actually losing too much and, at the same time, easing the access to these data. For this example, we will be using **[Qdrant][5]** as vector database provider.

I wrote an educational article on vector databases [on my personal blog][6]

<h3>2c. RAG</h3>
RAG is the acronym for **Retrieval-Augmented Generation**: this is a technique employed to get **more reliable responses** from LLMs. It is not unusual that language models *hallucinate*, providing false or misleading information: you can build a vector database with **all the relevant information you want your model to know** and query the database right before feeding your request to the LLM, providing the results from your vector search as a context… This will **remarkably improve** the quality of the AI-generated answers, making the LLM **context-aware**. 

<h3>2d API</h3>
API stands for **Application Programming Interface**: this represents a set of protocols and rules that allow that two applications interact with each other by sending and receiving information. For this tutorial, we will be using 

  [1]: https://coderlegion.com/252/create-a-python-telegram-bot-plain-simple-and-production-ready
  [2]: https://coderlegion.com/261/learn-how-to-build-a-user-friendly-conversational-telegram-bot-with-python
  [3]: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
  [4]: https://astrabert.github.io/hophop-science/Transformers-architecture-for-everyone/
  [5]: https://qdrant.tech/
  [6]: https://astrabert.github.io/hophop-science/Vector-databases-explained/