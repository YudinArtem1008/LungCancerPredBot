from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message
from utils import model

import text
import os
import torch
import numpy as np

from PIL import Image

router = Router()


@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet.format(name=msg.from_user.full_name))


@router.message(F.photo)
async def predict_cancer(msg: Message):
    i = len([name for name in os.listdir('./photos') if os.path.isfile(name)])
    await msg.bot.download(file=msg.photo[-1].file_id, destination=f'./photos/photo{i + 1}.jpg')

    img = torch.tensor(np.asarray(Image.open(f'./photos/photo{i + 1}.jpg').resize((256, 256))), dtype=torch.float32).permute(2, 0, 1)
    img /= 255
    # Так, теперь хуйня с предсказаниями. Я не знаю, как это фиксить. Походу, надо обучать нейронку заново
    # upd: Походу, я знаю, как решить эту проблему без обучения нейронки заново
    img = torch.stack((img, img, img, img, img, img, img, img,
                       img, img, img, img, img, img, img, img,
                       img, img, img, img, img, img, img, img,
                       img, img, img, img, img, img, img, img))
    pred_list = list(model(img).max(1).indices.cpu().detach())
    frequency = [0, 0, 0]
    for el in pred_list:
        frequency[el] += 1
    pred = frequency.index(max(frequency))
    if pred == 0:
        await msg.answer(text.bengin_cancer)
    elif pred == 1:
        await msg.answer(text.malignant_cancer)
    else:
        await msg.answer(text.no_cancer)
