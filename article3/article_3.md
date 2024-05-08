<h2>1. Introduction</h2>
In my previous articles, we saw [how to create a simple Telegram bot][1] and [how to build a responsive assistant][2]: in both of these cases, we built something that relied on pre-defined responses, no matter how natural language-like they might have seemed. 

In this tutorial we will build an **AI-powered** and **context-aware** Telegram bot that reads information from **PDF files** and responds in multiple languages. Before we start, nevertheless, we need to define some important terms that we will be using in this tutorial, and that are not-so-common in everyday programming.
<h2>2. Definitions</h2>
<h3>2a. LLM</h3>
 LLM stands for **Large Language Model**: it is an Artificial Intelligence-based model able to understand, process and produce natural language, performing also complex tasks. For this tutorial, we will be using **[Phi-3-128K instruct][3]**, which is one of the most recent and powerful models, released by Microsoft.

I wrote about LLMs architecture in [a post on my personal blog][4]
<h3>2b. Vector Databases</h3>
A vector database is a **non-traditional** data storage facility and can be used to represent **complex data** (with lots of features) based on a set of **multi-dimensional numerical objects** (*vectors*). For this example, we will be using **[Qdrant][5]** as vector database provider.

I wrote an educational article on vector databases [on my personal blog][6]
<h3>2c. RAG</h3>
RAG is the acronym for **Retrieval-Augmented Generation**: you build a vector database with **all the relevant information you want your model to know** and query the database right before feeding your request to the LLM, providing the results from your search as a context… This will **remarkably improve** the quality of the AI-generated answers, making the LLM **context-aware**. 
<h3>2e. Docker</h3>
You have to imagine Docker as a big registry where lots of applications (**images**) are stored: you can download (**pull**) them and make them run in a virtual environment (the **container**). A Docker application contains everything that is needed for it to run in the container. A Docker container may also come with some data storage space (a **volume**), which should be mounted on your local file system. 
<h2>3. Setup</h2>
<h3>3a. Folder Structure</h3>
As usual, we start by setting up our local folder for the tutorial: you can refer to the structure found [on my GitHub repo](https://github.com/AstraBert/coderlegion/tree/main/article3), which is here represented:

    .
    ├── bot.py
    ├── requirements.txt
    ├── utils.py
    ├── qdrant_storage/
Let's break down the files:

- **bot.py** will be our main script, containing the Telegram bot
- **requirements.txt** will be the file where we write all the needed dependencies, in order to install them
- **utils.py** will be the script where we define useful functions and classes for our bot
- **qdrant_storage/** will be the local folder where the databases will be stored.

<h3>3b. Install Dependencies</h3>
Open *requirements.txt* and paste the following text:

    gradio_client==0.15.0
    pypdf==3.17.4
    sentence_transformers==2.2.2
    transformers==4.39.3
    langdetect==1.0.9
    deep-translator==1.11.4
    qdrant_client==1.9.0
    langchain-community==0.0.13 
    langchain==0.1.1 

These are all the python packages that we need to build the bot and its back-end architecture: 

- **gradio_client** allows to interact with Gradio API
- **pypdf** manages PDF files
- **sentence_transformers** helps with transforming data into vectors
- **transformers** is an HuggingFace library commonly used for LLMs
- **langdetect** and **deep-translator** are two packages that manage language detection and translation, to add multilingual support to our bot
- **qdrant_client** allows to interact with Qdrant running in the background
- **langchain** and **langchain-community** are useful to preprocess PDFs and turn them into plain-text data.

Now we save *requirements.txt*, head over to our terminal and run:

    python3 -m pip install -r requirements.txt
<h3>3c. Install and Run Qdrant with Docker</h3>
We will need **Qdrant running in a Docker container** on our machine in order to build the vector database. 
To do so, first of all we need to **download it from Docker Hub**:

    docker pull qdrant/qdrant
And then we can make it run, mounting as a volume (`-v` option) our local *qdrant_storage* folder:

     docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant_storage:z qdrant/qdrant
Now, if you want to see Qdrant WebUI, just type`localhost:6333` on your browser and press `enter`.
<h2>4. Define Useful Functions and Classes</h2>
In this section, we will be editing *utils.py*
<h3>4a. Import Necessary Dependencies</h3>
To make our script work, we need to import all the following packages, classes and/or functions:

    from langdetect import detect
    from deep_translator import GoogleTranslator
    from pypdf import PdfMerger
    from qdrant_client import models
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    import os

<h3>4b. Handle PDFs</h3>
We first define a function to **merge multiple PDFs** (the Telegram bot does not take multiple documents as input, but this can be always useful): 

    def merge_pdfs(pdfs: list):
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(f"{pdfs[-1].split('.')[0]}_results.pdf")
        merger.close()
        return f"{pdfs[-1].split('.')[0]}_results.pdf"

And another function that **removes blank lines from a list** (we will use it when subdividing our PDFs into smaller chuncks of plain text):

    def remove_items(test_list, item): 
        res = [i for i in test_list if i != item] 
        return res 
Now we can initialize a class that is able to **turn PDFs into Qdrant collections** (i.e. vector databases):

    class PDFdatabase:
        def __init__(self, pdfs, encoder, client):
            self.finalpdf = merge_pdfs(pdfs)
            self.collection_name = os.path.basename(self.finalpdf).split(".")[0].lower()
            self.encoder = encoder
            self.client = client
The `client` is the interface between our script and Qdrant, while the `encoder` is the sentence-transformer model that turns our text data into vectors.
Now we define, inside the class, three functions to **preprocess** (turn into text and subdivide in batches), **organize** and **vectorize** our PDF files:

       def preprocess(self):
            loader = PyPDFLoader(self.finalpdf)
            documents = loader.load()
            ### Split the documents into smaller chunks for processing
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            self.pages = text_splitter.split_documents(documents)
        def collect_data(self):
            self.documents = []
            for text in self.pages:
                contents = text.page_content.split("\n")
                contents = remove_items(contents, "")
                for content in contents:
                    self.documents.append({"text": content, "source": text.metadata["source"], "page": str(text.metadata["page"])})
            return self.collection_name
        def qdrant_collection_and_upload(self):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                    distance=models.Distance.COSINE,
                ),
            )
            self.client.upload_points(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=idx, vector=self.encoder.encode(doc["text"]).tolist(), payload=doc
                    )
                    for idx, doc in enumerate(self.documents)
                ],
            )
<h3>4c. Search the Vector Database</h3>
We now have to build a class that **searches** the vector database using **Qdrant client and the name of the collection** to search as initial inputs, and the **query as search term**:

    class NeuralSearcher:
        def __init__(self, collection_name, client, model):
            self.collection_name = collection_name
            # Initialize encoder model
            self.model = model
            # initialize Qdrant client
            self.qdrant_client = client
        def search(self, text: str):
            # Convert text query into vector
            vector = self.model.encode(text).tolist()
    
            # Use `vector` for search for closest vectors in the collection
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=None,  # If you don't want any filters for now
                limit=1,  # 5 the most closest results is enough
            )
            # `search_result` contains found vector ids with similarity scores along with the stored payload
            # In this function you are interested in payload only
            payloads = [hit.payload for hit in search_result]
            return payloads
<h3>Translation</h3>
We now build a class that is able to **detect 55 languages** (the ones supported by Google Translator) and to **translate** the provided text from the **original** (automatically-detected) language to the **target** one:

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
<h2>5. The Bot</h2>
We already know how to get our Telegram API Token: after having obtained it, we open *bot.py* and paste it as follows:

    TOKEN = ""
We can then import everything we need:

    from gradio_client import Client
    from utils import *
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from telegram.ext import *
And set some pre-defined variables:

    client = QdrantClient("localhost:6333")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    collection_name = ""
    api_client = Client("eswardivi/Phi-3-mini-128k-instruct")
    lan = "en"
<h3>5a. Start the Conversation</h3>
First of all, we need to start the conversation with the user and understand in what language will they provide their PDFs:

    LAN = 1 # conversation handler index
    async def start_command(update, context):
        user = update.message.from_user
        await update.message.reply_text(f"Hi {user.first_name} {user.last_name}, and thank you so much for having chosen RAGBOT as your assistant today!\nI'm here to help you chatting with your pdfs, so let's start with an ice-breaking question: what language is your pdf written in? Reply with the command '/lan' followed by the ISO code of the language, that you can find here: https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes. For example, if your pdfs are in Italian, reply: '/lan it' (without quotation marks)")
        return LAN
    async def handle_lan(update, context):
        global lan
        message = update.message.text
        message = message.replace("/lan ", "")
        lan = message
        txt = Translation(f"Now the language has been changed to {lan}", lan)
        await update.message.reply_text(txt.translatef())
        return ConversationHandler.END

<div class="div-blue">
<span class="alert-body">NOTE:</span>
<span class="alert-header"> we need to provide the language with the ISO code (English becomes "en", Italian becomes "it" and so on...)</span>
</div>
<h3>5b. Upload the PDF Document</h3>
Now that we have started the conversation and set the language, we can pass our first PDF document, that will be handled by our `handle_PDFs` function:

    async def handle_pdfs(update, context):
        global collection_name
        global lan
        if update.message.document:
            doc = update.message.document
            inf = "downloaded_from_user.pdf"
            fid = doc.file_id
            print(fid)
            # Download the file from Telegram to the local directory
            await (await context.bot.get_file(fid)).download_to_drive(custom_path=inf)
            pdfdb = PDFdatabase([inf], encoder, client)
            pdfdb.preprocess()
            collection_name = pdfdb.collect_data()
            pdfdb.qdrant_collection_and_upload()
            txt = Translation("Your document has been succesfully uploaded to a Qdrant collection!", lan)
            await update.message.reply_text(txt.translatef())
This function is able to **download the PDF document** from Telegram to our local folder **preprocess** it and turn it **into a Qdrant collection in a short time**: when it is done, we receive a message (in our favorite language) that tells us that the operation has been completed successfully.
<h3>5c. Talk to the LLM about Your Document</h3>
Now we have our last function before actually building the application: `reply` will take the user's **query**, adapt it to the **language of the PDF**, search in the PDF for **context information**, build a **prompt (in English)** that contains both the **context** and the user's **question**. This last prompt is then fed to Phi-3, that responds:

    async def reply(update, context):
        global collection_name
        global client
        global encoder
        global api_client
        global lan
        message = update.message.text
        txt = Translation(message, "en")
        print(txt.original, lan)
        if txt.original == "en" and lan == "en":
            txt2txt = NeuralSearcher(collection_name, client, encoder)
            results = txt2txt.search(message)
            response = api_client.predict(
                f"Context: {results[0]['text']}; Prompt: {message}",# str  in 'Message' Textbox component
                0.4,# float (numeric value between 0 and 1) in 'Temperature' Slider component
                True,# bool  in 'Sampling' Checkbox component
                512,# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
                api_name="/chat"
            )
            await update.message.reply_text(response)
        elif txt.original == "en" and lan != "en":
            txt2txt = NeuralSearcher(collection_name, client, encoder)
            transl = Translation(message, lan)
            message = transl.translatef()
            results = txt2txt.search(message)
            t = Translation(results[0]["text"], txt.original)
            res = t.translatef()
            response = api_client.predict(
                f"Context: {res}; Prompt: {message}",# str  in 'Message' Textbox component
                0.4,# float (numeric value between 0 and 1) in 'Temperature' Slider component
                True,# bool  in 'Sampling' Checkbox component
                512,# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
                api_name="/chat"
            )
            response = Translation(response, txt.original)
            await update.message.reply_text(response.translatef())
        elif txt.original != "en" and lan == "en":
            txt2txt = NeuralSearcher(collection_name, client, encoder)
            results = txt2txt.search(message)
            transl = Translation(results[0]["text"], "en")
            translation = transl.translatef()
            response = api_client.predict(
                f"Context: {translation}; Prompt: {message}",# str  in 'Message' Textbox component
                0.4,# float (numeric value between 0 and 1) in 'Temperature' Slider component
                True,# bool  in 'Sampling' Checkbox component
                512,# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
                api_name="/chat"
            )
            t = Translation(response, txt.original)
            res = t.translatef()
            await update.message.reply_text(res)
        else:
            txt2txt = NeuralSearcher(collection_name, client, encoder)
            transl = Translation(message, lan.replace("\\","").replace("'",""))
            message = transl.translatef()
            results = txt2txt.search(message)
            t = Translation(results[0]["text"], txt.original)
            res = t.translatef()
            response = api_client.predict(
                f"Context: {res}; Prompt: {message}",# str  in 'Message' Textbox component
                0.4,# float (numeric value between 0 and 1) in 'Temperature' Slider component
                True,# bool  in 'Sampling' Checkbox component
                512,# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
                api_name="/chat"
            )
            tr = Translation(response, txt.original)
            ress = tr.translatef()
            await update.message.reply_text(ress)
<h3>5d. Build the Application and Make It Run</h3>
We have all the utilities and functions to build our Telegram bot, now. 
To do so, we first create the application:

    if __name__ == "__main__":
        print("Bot is up and running")
        application = Application.builder().token(TOKEN).build()
And the conversation handler:

    conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', start_command)],
            states={
                LAN: [CommandHandler('lan', handle_lan)]
            },
            fallbacks=[]
        )
Then we add all the handlers to our application and set it up to run:

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdfs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    application.run_polling(1.0)
<div class="div-blue">
<span class="alert-body">NOTE:</span>
<span class="alert-header"> we use a particular filter to recognize messages with uploaded pdfs: can you spot it?</span>
</div>
Now we can finally give our bot a try!
    python3 bot.py

And here's an example chat:
![chat_example][9]
The uploaded PDF file contains information about penguins (generated with Llama-3 70B).

<h3>6. Conclusion</h3>
We built our first **AI-powered** bot, which is inclusive on the side of the language (**English is not necessarily needed**) and can expand its knowledge based on *our PDF files*, fully implementing the concept of RAG.

Our bot is **useful** to anyone that wants to **learn**, **query** their documents **rapidly** and **access large amount of information**: the most interesting thing is that what we built in this tutorial is **accessible to everyone**, completely **open-source** and it can be generated **without a big amount of prior knowledge**. In the end, it doesn't take that much to become part of the **AI revolution**!

  [1]: https://coderlegion.com/252/create-a-python-telegram-bot-plain-simple-and-production-ready
  [2]: https://coderlegion.com/261/learn-how-to-build-a-user-friendly-conversational-telegram-bot-with-python
  [3]: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
  [4]: https://astrabert.github.io/hophop-science/Transformers-architecture-for-everyone/
  [5]: https://qdrant.tech/
  [6]: https://astrabert.github.io/hophop-science/Vector-databases-explained/
  [7]: https://www.gradio.app/
  [8]: https://huggingface.co/spaces/eswardivi/Phi-3-mini-128k-instruct
  [9]: https://coderlegion.com/?qa=blob&qa_blobid=8528884168806013813