import io
import cv2
import numpy as np
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from predict import ModelPredict

model_predict = ModelPredict()

USE_WEBHOOK = True
TEXT = "Загрузите изображение (bbox) и бот определит лейбл класса, к которому принадлежит это изображение."

BBOX_UPLOAD = 0

def start(update, context):
    update.message.reply_text(TEXT)
    return BBOX_UPLOAD

def bbox_upload(update, context):
    user = update.message.from_user
    image = io.BytesIO()
    if len(update.message.photo) > 0:
        photo_file = update.message.photo[-1].get_file()
        photo_file.download(out=image)
    elif update.message.document:
        update.message.document.get_file().download(out=image)
    else:
        update.message.reply_text("Файл не является изображением.")
        return MessageHandler.START
    image.seek(0)
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    label, class_ = model_predict.predict(img)
    update.message.reply_text('Класс: %s, Label: %s' % (class_, label))
    return BBOX_UPLOAD

def other(update, context):
    update.message.reply_text("Изображение не выбранно.")
    update.message.reply_text(TEXT)
    return BBOX_UPLOAD

def error(update, context):
    print ('Update "%s" caused error "%s"', update, context.error)

def main():
    TOKEN = os.environ.get("TOKEN", None)
    if not TOKEN:
        print ("No TOKEN provided!")
        exit(-1)
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],

        states={
            BBOX_UPLOAD: [MessageHandler(Filters.photo | Filters.document.category("image"), bbox_upload), 
                          MessageHandler(Filters.all, other)
                          ],
        },

        fallbacks=[]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)
    if USE_WEBHOOK:
        PORT = int(os.environ.get("PORT", "8443"))
        HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")
        updater.start_webhook(listen="0.0.0.0",
                              port=PORT,
                              url_path=TOKEN)
        updater.bot.set_webhook("https://{}.herokuapp.com/{}".format(HEROKU_APP_NAME, TOKEN))
    else:
        updater.start_polling()
    print ("Running...")
    updater.idle()


if __name__ == '__main__':
    main()
