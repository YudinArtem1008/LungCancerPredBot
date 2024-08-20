from aiogram import F, Router, types
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
    await msg.photo[-1].download(f'./photos/photo{i + 1}.jpg')

    img = torch.tensor(np.asarray(Image.open(f'./photos/photo{i + 1}.jpg').resize((256, 256)))).permute(2, 0, 1)
    pred = model(img)
    await msg.answer(pred)
