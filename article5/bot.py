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


CHANNEL_ID = int(os.getenv("discord_channel"))
TOKEN = os.getenv("discord_token")

intents = Intents.default()
intents.messages = True


bot = Client(intents=intents)


@bot.event
async def on_ready():
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        # Send the welcome message
        await channel.send(f"The bot was activated at: {time.time()}")
    else:
        print("Unable to find the specified channel ID. Make sure the ID is correct and the bot has the necessary permissions.")

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