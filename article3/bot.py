from gradio_client import Client
from utils import *
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from telegram.ext import *

client = QdrantClient("localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = ""
api_client = Client("eswardivi/Phi-3-mini-128k-instruct")
lan = "en"
LAN = 1

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
            f"Context: {results[0]['text']}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
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
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
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
            f"Context: {translation}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
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
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        tr = Translation(response, txt.original)
        ress = tr.translatef()
        await update.message.reply_text(ress)

async def unrecognized_command(update, context):
    text = update.message.text
    text.replace("/", "")
    if (
        text.startswith("start") == False
        or text.startswith("lan") == False
    ):
        await update.message.reply_text(
            f'I cannot understand the message:\n"{text}"\nAs my programmer did not insert it among the command I am set to respond (that are /price, /menu and /location): please check for misspelling/errors or contact the programmer if you feel anything is wrong/missing'
        )
    else:
        pass


async def error_handler(update, context: CallbackContext) -> None:
    print(f"An error occurred: {context.error}")
    await update.message.reply_text("Sorry, something went wrong.")

TOKEN = ""

if __name__ == "__main__":
    print("Bot is up and running")
    application = Application.builder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start_command)],
        states={
            LAN: [CommandHandler('lan', handle_lan)]
        },
        fallbacks=[]
    )
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdfs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    application.add_handler(MessageHandler(filters.TEXT, unrecognized_command))
    application.add_error_handler(error_handler)
    application.run_polling(1.0)