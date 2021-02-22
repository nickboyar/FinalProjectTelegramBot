import torch
from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.types import InlineKeyboardButton

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


from torchvision.transforms import transforms

from nst.single.SingleStyleTransfer import SingleStyleTransfer
from nst.twice.TwiceStyleTransfer import TwiceStyleTransfer
from nst.universe.UniverseStyleTransfer import UniverseStyleTransfer


API_TOKEN = '1670309388:AAEQ5tHLziTI8beOZey0mfUeK8UTtdQ0bgM'

bot = Bot(token=API_TOKEN)

mb = Dispatcher(bot)

userInfo = {}


class Settings:
    def __init__(self):
        self.settings = {'num_epochs': 300,
                         'imsize': 256}
        self.photos = []

    def set_standard_settings(self):
        self.settings = {'num_epochs': 300,
                         'imsize': 256}

    def set_high_settings(self):
        self.settings = {'num_epochs': 300,
                         'imsize': 512}



button1 = KeyboardButton('Выбрать стиль')
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(button1)



#start
@mb.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply(f"Привет, {message.from_user.first_name}!\n", reply_markup=greet_kb)
    await bot.send_message(message.chat.id,
                           "Я могу перенести стиль с одной фотографии на другую. " +
                           "Есть несколько вариантов для переноса твоего стиля. " +
                           "Приступим ?\n", reply_markup=choice_kb)
    userInfo[message.chat.id] = Settings()



@mb.message_handler(content_types=['text'])
async def get_example(message):
    if str(message["text"])=='Выбрать стиль':
        await bot.send_message(message.chat.id,
                           "Что выберешь ?\n", reply_markup=choice_kb)



choice_kb = InlineKeyboardMarkup()
choice_kb.add(InlineKeyboardButton("Использовать свой стиль", callback_data='my'))



bot_kb = InlineKeyboardMarkup()
bot_kb.add(InlineKeyboardButton("Одну", callback_data='single'))
bot_kb.add(InlineKeyboardButton("Две", callback_data='two'))



mix_kb = InlineKeyboardMarkup()
mix_kb.add(InlineKeyboardButton("Смешанный", callback_data='universe'))
mix_kb.add(InlineKeyboardButton("По частям", callback_data='twice'))


# my
@mb.callback_query_handler(lambda c: c.data == 'my')
async def my(callback_query):
    await bot.answer_callback_query(callback_query.id)

    await callback_query.message.edit_text(
        "Сколько фотографий для стиля будем использовать ?")
    await callback_query.message.edit_reply_markup(reply_markup=bot_kb)

# your
@mb.callback_query_handler(lambda c: c.data == 'two')
async def two(callback_query):
    await bot.answer_callback_query(callback_query.id)

    await callback_query.message.edit_text(
        "Сделаем смешанный стиль или применим по частям ? ")
    await callback_query.message.edit_reply_markup(reply_markup=mix_kb)


#setting to define resolution
settings_kb = InlineKeyboardMarkup()
settings_kb.add(InlineKeyboardButton('Стандартное',
                                      callback_data='standard'))
settings_kb.add(InlineKeyboardButton('Высокое',
                                      callback_data='high'))


# style transfer one style
@mb.callback_query_handler(lambda c: c.data == 'single')
async def single_style(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Для обработки картинки может понадобиться некоторое время. " +
                                           "Предлагаю выбрать качество финальной картинки. Имей ввиду " +
                                           "чем выше качество - тем дольше ждать обработку ")
    await callback_query.message.edit_reply_markup(reply_markup=settings_kb)

    if callback_query.from_user.id not in userInfo:
        userInfo[callback_query.from_user.id] = Settings()
    userInfo[callback_query.from_user.id].st_type = 'single'


# style transfer universe style
@mb.callback_query_handler(lambda c: c.data == 'universe')
async def universe_style(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Для обработки картинки может понадобиться некоторое время. " +
                                           "Предлагаю выбрать качество финальной картинки. Имей ввиду " +
                                           "чем выше качество - тем дольше ждать обработку ")
    await callback_query.message.edit_reply_markup(reply_markup=settings_kb)

    if callback_query.from_user.id not in userInfo:
        userInfo[callback_query.from_user.id] = Settings()
    userInfo[callback_query.from_user.id].st_type = 'universe'

# style transfer twice styles
@mb.callback_query_handler(lambda c: c.data == 'twice')
async def twice_style(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Для обработки картинки может понадобиться некоторое время. " +
                                           "Предлагаю выбрать качество финальной картинки. Имей ввиду " +
                                           "чем выше качество - тем дольше ждать обработку ")
    await callback_query.message.edit_reply_markup(reply_markup=settings_kb)
    if callback_query.from_user.id not in userInfo:
        userInfo[callback_query.from_user.id] = Settings()
    userInfo[callback_query.from_user.id].st_type = 'twice'




# standard settings
@mb.callback_query_handler(lambda c: c.data == 'standard')
async def standard_set(callback_query):
    await bot.answer_callback_query(callback_query.id)

    if userInfo[callback_query.from_user.id].st_type == 'single':
        await callback_query.message.edit_text(
                                               "Пришли мне картинку, стиль с которой нужно перенести ")
        userInfo[callback_query.from_user.id].need_photos = 2

    elif userInfo[callback_query.from_user.id].st_type == 'twice':
        await callback_query.message.edit_text(
                                               "Пришли мне картинку, стиль с которой нужно перенести на левую часть изображения ")
        userInfo[callback_query.from_user.id].need_photos = 3

    elif userInfo[callback_query.from_user.id].st_type == 'universe':
        await callback_query.message.edit_text(
                                               "Хорошо, теперь пришли мне картинку для первого стиля ")
        userInfo[callback_query.from_user.id].need_photos = 3

    userInfo[callback_query.from_user.id].set_standard_settings()


# high settings
@mb.callback_query_handler(lambda c: c.data == 'high')
async def high_set(callback_query):
    await bot.answer_callback_query(callback_query.id)

    if userInfo[callback_query.from_user.id].st_type == 'single':
        await callback_query.message.edit_text(
            "Пришли мне картинку, стиль с которой нужно перенести. ")
        userInfo[callback_query.from_user.id].need_photos = 2

    elif userInfo[callback_query.from_user.id].st_type == 'twice':
        await callback_query.message.edit_text(
            "Пришли мне картинку, стиль с которой нужно перенести на левую часть изображения. ")
        userInfo[callback_query.from_user.id].need_photos = 3

    elif userInfo[callback_query.from_user.id].st_type == 'universe':
        await callback_query.message.edit_text(
            "Хорошо, теперь пришли мне картинку для первого стиля. ")
        userInfo[callback_query.from_user.id].need_photos = 3
    userInfo[callback_query.from_user.id].set_high_settings()


# preparing image
@mb.message_handler(content_types=['photo', 'document'])
async def get_image(message):
    if message.content_type == 'photo':
        img = message.photo[-1]

    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)

    userInfo[message.chat.id].photos.append(photo)

    # single style transfer
    if userInfo[message.chat.id].st_type == 'single':
        if userInfo[message.chat.id].need_photos == 2:
            userInfo[message.chat.id].need_photos = 1

            await bot.send_message(message.chat.id,
                                   "Отлично, теперь пришли мне картинку, на которую нужно перенести этот стиль. ")

        elif userInfo[message.chat.id].need_photos == 1:
            await bot.send_message(message.chat.id, "Запускаю процесс переноса стиля")

            output = await style_transfer(SingleStyleTransfer, userInfo[message.chat.id],
                                              *userInfo[message.chat.id].photos)
            await bot.send_photo(message.chat.id, output)


    # double style transfer
    elif userInfo[message.chat.id].st_type == 'twice':
        if userInfo[message.chat.id].need_photos == 3:
            userInfo[message.chat.id].need_photos = 2
            await bot.send_message(message.chat.id,
                                   "Отлично, теперь пришли мне картинку, стиль с которой " +
                                   "нужно перенести на правую часть изображения. ")
        elif userInfo[message.chat.id].need_photos == 2:
            userInfo[message.chat.id].need_photos = 1
            await bot.send_message(message.chat.id,
                                   "Отлично, теперь пришли мне картинку, на которую нужно перенести эти стили. ")
                                   
        elif userInfo[message.chat.id].need_photos == 1:
            await bot.send_message(message.chat.id, "Запускаю процесс переноса стиля")
            output = await style_transfer(TwiceStyleTransfer, userInfo[message.chat.id],
                                              *userInfo[message.chat.id].photos)
            await bot.send_photo(message.chat.id, output)

    # union style transfer
    elif userInfo[message.chat.id].st_type == 'universe':
        if userInfo[message.chat.id].need_photos == 3:
            userInfo[message.chat.id].need_photos = 2

            await bot.send_message(message.chat.id,
                                   "Супер, теперь жду вторую ")

        elif userInfo[message.chat.id].need_photos == 2:
            userInfo[message.chat.id].need_photos = 1

            await bot.send_message(message.chat.id,
                                   "Отлично, осталось теперь загрузить картинку, на которую перенесем эти стили. ")

        elif userInfo[message.chat.id].need_photos == 1:
            await bot.send_message(message.chat.id, "Запускаю процесс переноса стиля")

            output = await style_transfer(UniverseStyleTransfer, userInfo[message.chat.id],
                                              *userInfo[message.chat.id].photos)
            await bot.send_photo(message.chat.id, output)



async def style_transfer(st_class, user, *imgs):
    st = st_class(*imgs,
                  imsize=user.settings['imsize'],
                  num_steps=user.settings['num_epochs'],
                  style_weight=100000, content_weight=1)
    output = await st.transfer()
    return tensorimg(output)


def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensorimg(t):
    output = np.rollaxis(t.cpu().detach().numpy()[0], 0, 3)
    output = Image.fromarray(np.uint8(output * 255))
    bio = BytesIO()
    bio.name = 'result.jpeg'
    output.save(bio, 'JPEG')
    bio.seek(0)
    return bio

def draw_img(img):
    plt.imshow(np.rollaxis(img.cpu().detach()[0].numpy(), 0, 3))
    plt.show()

def draw_photo(*photos):
    for photo in photos:
        img = np.array(Image.open(photo))
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    executor.start_polling(mb, skip_updates=True)
