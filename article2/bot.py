from telegram.ext import *
import pandas as pd
import joblib

MODEL = joblib.load("model.joblib")

TOKEN = ""

CSV = "users.csv"

USR_DF = {"Month": 0,"Duration(days)": 0,"Age": 0,"Target": 0,"Accommodation": 0,"Acc_cost": 0,"Transportation": 0,"Transp_cost": 0}

NUM2MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

TARGET2NUM = {"Alone": 0, "Couple": 1,  "Group": 2, "Family": 3}
ACC2NUM = {'Hostel': 0, 'Hotel': 1, 'Airbnb': 2, 'Resort': 3, 'Vacation rental': 4, 'Guesthouse': 5, 'Villa': 6}
TRANSP2NUM = {'Bus': 0, 'Car': 1, 'Train': 2, 'Plane': 3}

AGE, MONTH, DURATION, TARGET, ACC, ACCOST, TRANSP, TRANSPCOST = range(len(USR_DF))

async def start_command(update, context):
    user = update.message.from_user
    await update.message.reply_text(f"Hi {user.first_name} {user.last_name}, and thank you so much for having chosen ItalianTravelBot as your assistant today!\nAs you may already know, we offer several travel options, of which three are available right now: Rome, Florence, Courmayeur... I'm here to help you choose the most suitable for your needs and desires, so let's start with an ice-breaking question: How old are you? (Reply with a number)")
    return AGE

async def age(update, context):
    global USR_DF
    age = int(update.message.text)
    USR_DF["Age"] = [age]
    await update.message.reply_text(f"Wow, {age} years old: that's nice! And in which month are you planning to go on vacation? (Reply with the month number)")
    return MONTH


async def month(update, context):
    global USR_DF
    mon = int(update.message.text)
    USR_DF["Month"] = [mon]
    await update.message.reply_text(f"{NUM2MONTHS[mon]}, awesome! And how are you going to travel? Reply with one of this categories: Alone, Couple, Group, Family")
    return TARGET

async def target(update, context):
    global USR_DF
    tar = update.message.text
    USR_DF["Target"] = [TARGET2NUM[tar.capitalize()]]
    await update.message.reply_text(f"{tar}, that's just perfect! And how are you planning to go there? We offer several options: Bus, Car, Plane, Train")
    return TRANSP

async def transp(update, context):
    global USR_DF
    tra = update.message.text
    USR_DF["Transportation"] = [TRANSP2NUM[tra.capitalize()]]
    await update.message.reply_text(f"You wanna go by {tra.capitalize()}, got it! And how much do you plan to spend on transportation? Reply with a number")
    return TRANSPCOST

async def transpcost(update, context):
    global USR_DF
    mon = int(update.message.text)
    USR_DF["Transp_cost"] = [mon]
    await update.message.reply_text(f"{mon}, that's a perfect match! Where are you planning to stay? We can offer: Hostel, Hotel, Airbnb, Resort, Vacation rental, Guesthouse, Villa. Reply exactly with one of these")
    return ACC


async def acc(update, context):
    global USR_DF
    tar = update.message.text
    USR_DF["Accommodation"] = [ACC2NUM[tar.capitalize()]]
    await update.message.reply_text(f"{tar.capitalize()}, we got this one!  And how much do you plan to spend on accommodation? Reply with a number")
    return ACCOST

async def accost(update, context):
    global USR_DF
    mon = int(update.message.text)
    USR_DF["Acc_cost"] = [mon]
    await update.message.reply_text(f"{mon}, seems like everything is fitting perfectly! Last question: how long do you wanna stay? Reply with the number of days")
    return DURATION

async def duration(update, context):
    global USR_DF, CSV
    mon = int(update.message.text)
    USR_DF["Duration(days)"] = [mon]
    print(USR_DF)
    await update.message.reply_text(f"Ok, got it! Just wait a few seconds and I'll tell you what is the best match for you!")
    usrdf = pd.DataFrame.from_dict(USR_DF)
    df = pd.read_csv(CSV)
    merged = pd.concat([df, usrdf], axis=0, ignore_index=True)
    merged.to_csv(CSV, index=False)
    pred = MODEL.predict(usrdf)
    await update.message.reply_text(f"The perfect destination for you is: {pred[0]}! Hope this was useful: in the next message you'll find the flyer with our offers for that city! :)")
    flyer = f"{pred[0]}.pdf"
    await update.message.reply_document(document=open(flyer, 'rb'))
    return ConversationHandler.END

if __name__ == "__main__":
    print("Bot is up and running")
    application = Application.builder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start_command)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            MONTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, month)],
            TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, target)],
            TRANSP: [MessageHandler(filters.TEXT & ~filters.COMMAND, transp)],
            TRANSPCOST: [MessageHandler(filters.TEXT & ~filters.COMMAND, transpcost)],
            ACC: [MessageHandler(filters.TEXT & ~filters.COMMAND, acc)],
            ACCOST: [MessageHandler(filters.TEXT & ~filters.COMMAND, accost)],
            DURATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, duration)]
        },
        fallbacks=[]
    )
    application.add_handler(conv_handler)
    application.run_polling(1.0)




