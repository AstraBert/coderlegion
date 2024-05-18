<h1 align="center">supabase-ai-chat</h1>
<h2 align="center">A journalist that nows lots of news about AI!</h2>

<h3 align="center">1. Introduction</h3>
**supabase-ai-chat** is natively developed as a Gradio-backed HF space and can be found [here](https://huggingface.co/spaces/as-cle-bert/supabase-ai-chat): it was developed to serve as a knowledgeable assistant in the field of AI news.

>*There is also a version built on top of Flowise but it is only a test deployment and, due to security reasons, its access is limited: you can request the access writing an email [to this address](mailto:astra.bertelli01@universitadipavia.it) in which you explain who you are and why you would want to use the application.*

It is an application that exploits Supabase ([the open-source alternative to Firebase](https://github.com/supabase/supabase)) as a vector store to search for context information and HF Spaces API to query and retrieve responses from Phi-3-128K.

supabase-ai-chat was built selecting 5000 AI-related news articles from [ai-tech-articles](https://huggingface.co/datasets/siavava/ai-tech-articles) dataset on HF Hub, with the help of Supabase as a vector store (if you don't know what vectore stores are, here is an article that I wrote about them: ["Vector databases explained"](https://astrabert.github.io/hophop-science/Vector-databases-explained/)). Let's break dow the steps we need to take to build this application.

<h3 align="center">2. Pre-requisites</h3>

<h4 align="center">2a. Supabase</h4>

Before we start building our application, we will need **to access Supabase**: you can head over to [their website](https://supabase.com/) and register through your GitHub account (which is definitely the simplest way to get access to their services).

Once your account has been created, you can go to *"Dashboard"* and select *"New Project"*. This will take you to the actual project you can build and it will require you to name it and save a password: **do not loose the password**, because it will be useful later on. 

After Supabase manages all the dependencies of your project, you will notice that there will be both an **API Key** and a **Project URL** that pop up as soon as you enter the page: they won't be useful to us for this project, but they can definitely be in the future.

Now make sure to go to **"Project Settings > Database"** and copy the *Connection String*: store it somewhere, for now, as it will be fundamental to build the application.

<h4 align="center">2b. HuggingFace</h4>

In order to get  an [Hugging Face](https://huggingface.co/) account:

- Go to Hugging Face from the link above and click on the **Sign up** bottom. Alternatively, you can directly follow [this link](https://huggingface.co/join).
- Provide e-mail address and password
- Follow the instructions as you are prompted by the registration procedure

<h3 align="center">3. Folder Setup</h3>

You will need to have a specific folder setup in order to build this application. You can achieve that setup by cloning the [GitHub repository](https://github.com/AstraBert/supabase-ai-chat):

```bash
git clone https://github.com/AstraBert/supabase-ai-chat.git
cd supabase-ai-chat
```

Let's now take a look at the repository:

```
./
├── README.md
├── app.py
├── .env.example
├── data
│   └── ainews_5k.csv.gz
├── requirements.txt
├── supabase_upsert.py
└── utils.py
```
We have:

* **app.py**, which is the script we will be using to design our application
* **utils.py**, which is the script where we will be storing useful classes and functions
* **supabase_upsert.py**, which is the script that we will upload our AI news to our Supabase project with
* **requirements.txt**, which contains the needed dependencies for this project
* **data/ainews_5k.csv.gz**, which is the compressed version of the file that stores all the 5000 news articles about AI
* **.env.example**, which is the example file to store environment variables

We can set our environment up by running:

```bash
python3 -m pip install -r requirements.txt
```

And, on the other hand, we complete the configuration by renaming the **.env.example** file to **.env** and by substituting to the placeholder string associated with *supabase_db* environment variable. An example of that would be:

```bash
supabase_db=postgresql://postgres.fsaosjjgbgpbsiag:averycomplexpassword@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
```

Remember that it is important to write **postgresql** at the beginning of the URL, and not only **postgres**, as the newest versions of *vecs*, the python package the serves as a PostgresSQL client, do not support this last variation of the baseurl. 

<h3 align="center">4. Prepare Supabase Collection</h3>

Now that we have everything set up in terms of environment, we can upload the documents to our Supabase project, configuring them as a collection. We will be doing it using the **supabase_upsert.py** script we downloaded earlier from GitHub.

Let's take a quick look at the code!

We start by importing the needed dependencies and loading the environment variable related to our Supabase database:

```python
import vecs
from dotenv import load_dotenv
import os

load_dotenv()
DB_CONNECTION = os.getenv("supabase_db")
```

Then we move on to connect to Supabase client and create a new collection called *documents*, which will be receiving vectors embedding of a size of 348:

```python
vx = vecs.create_client(DB_CONNECTION)

docs = vx.get_or_create_collection(name="documents", dimension=384)
```

Now we load the word-embedding model and proceed with the upsertion of the AI news to the Supabase collection (we need to read the csv, first):

```python
import pandas as pd
df = pd.read_csv("data/ainews_5k.csv.gz")
idex = list(df["id"])
texts = list(df["text"])
data = [{'id': idex[i], 'content': texts[i], 'embedding': []} for i in range(len(texts))]

from sentence_transformers import SentenceTransformers
encoder = SentenceTransformer("all-MiniLM-L6-v2")

for item in data:
    docs.upsert(records=[(item['id'],encoder.encode(item['content']).tolist(), {"Content": item['content']}),])
```

The last, but not least, portion of the code creates a vector index to ease the search for our retrieval algorithm:

```python
docs.create_index(measure=vecs.IndexMeasure.cosine_distance)
```

<h3 align="center">5. Define Utilities</h3>

After having upserted the AI news to Supabase, we just need to define two classes that will help us managing the queries from the users. 

The first one involves the possibility of translating the user's prompt from the original language to English, and then the response from English back to the original language:

```
from langdetect import detect
from deep_translator import GoogleTranslator

class Translation:
    def __init__(self, text, destination):
        self.text = text
        self.destination = destination
        try:
            self.original = detect(self.text)
        except Exception as e:
            self.original = "auto"
    def translatef(self):
        translator = GoogleTranslator(source=self.original, target=self.destination)
        translation = translator.translate(self.text)
        return translation
```

As you can see, we two python packages, **langdetect** and **deep_translator**, to achieve the goal. 

The second class allows us to search a Supabase collection using a text query (which is transformed into vectors on the fly by the same encoder we used to upload everything to the collection).

```python
class NeuralSearcher:
    def __init__(self, collection, encoder):
        self.collection = collection
        self.encoder = encoder
    def search(self, text):
        results = self.collection.query(
            data=self.encoder.encode(text).tolist(),  # required
            limit=1,                     # number of records to return
            filters={},                  # metadata filters
            measure="cosine_distance",   # distance measure to use
            include_value=True,         # should distance measure values be returned?
            include_metadata=True,      # should record metadata be returned?
        )
        return results
```

For this simple example project, we do not define filters: this can be done for more advanced projects and more complex necessities.

<h3 align="center">6. Build the Application</h3>

Now that we have everything set up, we can actually build our Gradio application that will be hosted in HF Spaces. 

We open **app.py** and start by importing packages and defining constants:

```python
import gradio as gr
from utils import Translation, NeuralSearcher
from gradio_client import Client
import os
import vecs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

collection_name = "documents"
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = os.getenv("supabase_db")
api_client = Client("eswardivi/Phi-3-mini-128k-instruct")
lan = "en"
vx = vecs.create_client(client)
docs = vx.get_or_create_collection(name=collection_name, dimension=384)
```

Now that we have defined our Supabase and Gradio API clients, we can finally build a function that takes the user query, recognizes the language, performs various translation tasks, retrieves some relevant information from Supabase collection, feeds them as context to Phi-3-128K along with the actual message from the user, and then finally gets a response. You can refer to the flowchart here, for a better visualization of the process: 

![Description](https://github.com/AstraBert/supabase-ai-chat/raw/main/data/supabase.drawio.png)


Here is the function:

```python
def reply(message, history):
    global docs
    global encoder
    global api_client
    global lan
    txt = Translation(message, "en")
    print(txt.original, lan)
    if txt.original == "en" and lan == "en":
        txt2txt = NeuralSearcher(docs, encoder)
        results = txt2txt.search(message)
        response = api_client.predict(
            f"Context: {results[0][2]['Content']}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096)
            api_name="/chat"
        )
        return response
    elif txt.original == "en" and lan != "en":
        txt2txt = NeuralSearcher(docs, encoder)
        transl = Translation(message, lan)
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",
            0.4,
            True,
            512,
            api_name="/chat"
        )
        response = Translation(response, txt.original)
        return response.translatef()
    elif txt.original != "en" and lan == "en":
        txt2txt = NeuralSearcher(docs, encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0][2]['Content'], "en")
        translation = transl.translatef()
        response = api_client.predict(
            f"Context: {translation}; Prompt: {message}",
            0.4,
            True,
            512,
            api_name="/chat"
        )
        t = Translation(response, txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcher(docs, encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,
            True,
            512,
            api_name="/chat"
        )
        tr = Translation(response, txt.original)
        ress = tr.translatef()
        return ress
```

Now, in order to build the Chat Interface, Gradio offers solutions that hardly have competitors in terms of simplicity: as a matter of facts, we actually need one line of code to define the interface and one to launch it. 

```python
demo = gr.ChatInterface(fn=reply, title="Supabase AI Journalist")
demo.launch(server_name="0.0.0.0", share=False)
```

Now, if you run:

```bash
python3 app.py
```

From your terminal, in less than 30 seconds you will see that **http://localhost:7860** gets crowded and fills with life: it's our chatbot! We can now query it by just writing a prompt and patiently waiting for the results to come!

We can also upload our repository to Spaces, or simply create a space [here](https://huggingface.co/spaces) and paste the code we just write in files that are named as our local version of them.

I strongly suggest, for safety reasons, **not to expose your .env file**: you can set secret keys in the Space settings, and just import the keys by simply including:

```python
client = os.getenv("your-key-name-here")
```

<h3 align="center">7. Conclusion</h3>

We now created our first Gradio application: this is the first time we build a fully **front-end and back-end** defined tool, without relying on third-party interfaces such as Telegram or Discord. 

Moreover, we natively integrated in our project a **vector database** (Supabase) for text retrieval and a **Large Language Model** (Phi-3-128K) for text generation: we did everything with **pure python** and, what's even better, in **less than 150 lines of code**!

Stay tuned for the next article, in which we will conclude our bot series by talking about **Discord bots**, and do not forget to live a little star in the repository!