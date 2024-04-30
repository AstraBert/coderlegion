from telegram.ext import *

TOKEN = ""

async def start_command(update, context):
    await update.message.reply_text("Hi, I'm NeapolitanPizzaBot, how may I assist you today?")

async def price_command(update, context):
    await update.message.reply_text("- Margherita: 8$\n- Marinara: 5$\n- Pepperoni: 9$")

async def location_command(update, context):
    await update.message.reply_venue(40.82694507320722, 14.426713826679295, "Italy", "Pizzeria NeapolitanPizzaBot")

async def menu_command(update, context):
    await update.message.reply_photo("menu.jpg")

async def unrecognized_command(update, context):
    text = update.message.text
    if (
        text.startswith("/start") == False
        or text.startswith("/menu") == False
        or text.startswith("/price") == False
        or text.startswith("/location") == False
    ):
        await update.message.reply_text(
            f'I cannot understand the message:\n"{text}"\nAs my programmer did not insert it among the command I am set to respond (that are /price, /menu and /location): please check for misspelling/errors or contact the programmer if you feel anything is wrong/missing'
        )
    else:
        pass


async def error_handler(update, context: CallbackContext) -> None:
    await update.message.reply_text("Sorry, something went wrong.")

if __name__ == "__main__":
    print("Bot is high and running")
    application = Application.builder().token(TOKEN).build()
    # Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("location", location_command))
    application.add_handler(MessageHandler(filters.COMMAND, unrecognized_command))
    application.add_handler(MessageHandler(filters.TEXT, unrecognized_command))
    application.add_error_handler(error_handler)
    # Run bot
    application.run_polling(1.0)
