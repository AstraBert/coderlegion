<h1>Build a Discord python assistant with Langchain</h1>

<h2>1. Introduction</h2>

Discord is one of the most widespread instant-messaging services, especially when it comes to developers: its structure and internal organization into servers and channels made it easy to use with large community and extremely popular all across the world.

But Discord **is more than just the blue-ish icon on your desktop**: it indeed comes with lots of integrations for developers, such as applications and bots. In today's tutorial, we will see how we can build an AI-powered bot, in pure and simple **python**, with the help of **Langchain**, a widespread and appreciated library to build scalable AI products through **composability**.

<h2>2. Setup</h2>
<h3>2a. Folder</h3>

For the setup, let's begin with structuring our base folder. To do so, I strongly suggest you navigate to my [GitHub repository](https://github.com/AstraBert/coderlegion) and either take a look to [article5](https://github.com/AstraBert/coderlegion/tree/main/article5/) directory or you **clone it** with:

git clone https://github.com/AstraBert/coderlegion.git
cd coderlegion/article5

Here you will see something like this:

.
├── .env.example
├── bot.py
├── inference.py
├── requirements.txt
└── utils.py

Let's take a look to what these file do: 

- `.env.example` (make sure to rename it to `.env`) is thd file where you safely store and secure private **environmental variables**. 
- `bot.py` is the script which **builds the bot and makes it run**
- `inference.py` contains the actual **Langchain** structure to power the bot with **AI**
- `requirements.txt` contains the **necessary packages** to be installed
- `utils.py`defines a class to **search the vector database** we will build in the next steps (more about vector databases at [_Vector databases explained_](https://astrabert.github.io/hophop-science/Vector-databases-explained/)).

<h3>2b. Dependencies</h3>

You will have to install the needed dependencies, and the easiest way to do it is to copy `requirements.txt` to your local folder and run:

python3 -m pip install -r requirements.txt

Let's see what packages we need:

- **transformers** and **sentence-transformers** are used to tokenize and encode (which means represent as multi-dimensional numerical arrays) users' messages to query the vector database
- **langchain-related** packages are needed to build the AI architecture, interact with Cohere and give the bot some conversational memory.
- **discord.py** is the Discord python SDK
- **vecs** is the python integration for Supabase (that we already used in the last tutorial)
- **pydotenv** is used to load environmental variables from the `.env` file.

<h3>2c. Vector Database</h3>

We will build our vector database with the help of Supabase and with the Hugging Face dataset `dipesh/python-code-ds-mini`, by Dipesh Paul, which encompasses **2200 python code instances**.

In order to do it, we will follow [this GitHub Gist](https://gist.github.com/AstraBert/455badcbf24fac2ff65115d12374d301).

In order to make the Gist code run properly, you just need to: 

- [Register to Supabase](https://supabase.com/dashboard/sign-up) (you can quickly sign up with GitHub)
- Create your first project
- Navigate to `Project Settings` and then `Database`, where you should copy the `Connection string`, making sure to replace the `postgres://` heading with `postgresql://`.
- Open the Gist in Google Colab
- Copy the `Connection string` into a Colab Secret (under the key sign) named "TINY_CODES_DB"

Now just run all the blocks in Colab and you'll have the **vector database built in 10 minutes' time**!

Make sure also to save the `Connection string` under `supabase_database` variable in your `.env` file!

<h3>2d. Discord Bot</h3>

Follow these simple steps to build the Discord bot container:

- Go to [Discord](https://discord.com/) and create an account (or log into yours, if you already have one)
- Create a new server by clicking on "Add a server" (a big green "+" button) and name it as you want
- Go to [Discord developers portal](https://discord.com/developers/applications) and click on "New application"
- Name your application, then save the changes and click on "Bot", on the left
- There you will get the chance to name your bot and copy its token
- After that, go on "OAuth2" > "URL generator" and generate a URL that you will paste in your navigation bar, to add the bot to your newly created server.

Now paste the bot token and your server's channel ID (the last numeric combination you see in the URL of your channel) to your `.env` file, as `discord_token` and `discord_channel`.

The bot is created, but it is **unresponsive**; this is because we only generated an external facade, with no backbone to sustain it: it's now time to fill it with life thanks some python code.

<h3>2e. Cohere API key</h3>

Getting a Cohere free API key is really simple: you need to register [on their website](https://dashboard.cohere.com/welcome/register) (you can exploit signing up with GitHub also here) and then head over to the `API keys` section, where you could generate a free key that has a 100 calls/min limit.

Copy the key and save it to your `.env` file, as `cohere_api_key`.

<h2>3. Build with Langchain</h2>

In this section, we're gonna modify **inference.py**

<h3>3a. Build the Cohere model</h3>

In order to connect with Cohere, you need to import the **needed packages**:

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

And then we load the environmental variables, get Cohere API key and load the chat model:

load_dotenv()
os.environ["COHERE_API_KEY"]  = os.getenv("cohere_api_key") 
model = ChatCohere()

We now need to actually build it into some sort of conversational chain, so we first define a system template (variables in brackets will be passed as arguments to the following block, so they get re-defined at every call):

system_template = "You are an helpful coding assistant that can rely on this code: {context} and on the previous message history as context, and from that you build a context and history-aware reply to this (DO NOT mention the fact that you are starting from a code snippet):"

<div class="div-green">
<span class="alert-header">TIP:</span>
<span class="alert-body"> This portion is critical for getting the best out of the chat model: customize it to your needs!</span>
</div>

And then a prompt template and the chain:

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]
)
chain = prompt_template | model

<h3>3b. Chat Memory</h3>

Chat memory is crucial for the bot to **remember the previous messages** during a conversation, so we need to add it with:

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory1.db")
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

`session_id` is an alphanumeric code associated with each session of activity of the bot: each session is independent and there is **no common knowledge across them**.

<h3>3c. Inference</h3>

Now that we have all set, we can define a inference function with which we query Cohere's model and we retrieve the answers:

def infer_reply(context, text, sessionid):
    global chain
    r = runnable_with_history.invoke(
        {"context": context, "input": text},
        config={"configurable": {"session_id": sessionid}}
    )
    return r.content

<h2>4. Vector Database Search</h2>

To search the vector database, we define a **NeuralSearcher** class inside **utils.py**, which is the same as the one we used in last tutorial:

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

<h2>5. Discord Bot</h2>

We have everything in place now, so we can actually import the needed packages in **bot.py** and define some global variables and the Discord client:

from discord import Client, Intents
from utils import NeuralSearcher
import os
import vecs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from inference import infer_reply
import time
import random as r

load_dotenv()
collection_name = "documents"
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = os.getenv("supabase_database")
vx = vecs.create_client(client)
docs = vx.get_or_create_collection(name=collection_name, dimension=384)
rand_sess_id = str(r.randint(1,100)) + r.choice(["a", "e", "i", "o", "u"])
CHANNEL_ID = int(os.getenv("discord_channel"))
TOKEN = os.getenv("discord_token")
intents = Intents.default()
intents.messages = True
bot = Client(intents=intents)

<div class="div-blue">
<span class="alert-header">NOTE:</span>
<span class="alert-body"> A unique session_id is set here at the beginning of the code, and will be re-set every time we start the bot</span>
</div>

Now let's tell the bot what to do when it connects to our channel:

@bot.event
async def on_ready():
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        # Send the welcome message
        await channel.send(f"Hi, I'm here to assist you!")
    else:
        print("Unable to find the specified channel ID. Make sure the ID is correct and the bot has the necessary permissions.")

<div class="div-blue">
<span class="alert-header">NOTE:</span>
<span class="alert-body"> The expression starting with '@' is known as a decorator</span>
</div>

Last but not least, we define a function that produces the context for Cohere chat model (which discards contexts with less than 25% similarity with the query), a function to reply to the users' query and we make the bot run:

def get_context(results):
    if results[0][1] > 0.25:
        return results[0][2]["Content"]
    else:
        return "There is no specific context for this query"

def reply(message):
    global docs
    global encoder
    txt2txt = NeuralSearcher(docs, encoder)
    results = txt2txt.search(message)
    print(results)
    context = get_context(results)
    response = infer_reply(context, message, rand_sess_id)
    return response

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  
    elif message.content:
        response = reply(message.content)
        await message.channel.send(
            response
        )
bot.run(TOKEN)

Ok, everything is ready! We go to the terminal and launch the bot with:

python3 bot.py

Then we wait for the connection to be established and for the welcome message by the bot.

<h2>6. Conclusion</h2>

In this tutorial, we learnt how to build a **free**, personal **coding assistant** that is able to help you **fix python snippets**, all accessible through a popular messaging platform like **Discord**.

The bot was built using **plain Langchain** and some other not-too-difficult python code, so it is **beginner friendly**!

This article **concludes the series dedicated to python-backed bots**: hope you enjoyed it!