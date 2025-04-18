import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv


async def echo(message: Message) -> None:
    await message.answer(message.text)


async def my_command(message: Message) -> None:
    await message.answer("my command handler")


async def main() -> None:
    # переименуй файл .env.dist в .env и подставь соотвествующие данные
    load_dotenv()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    dp = Dispatcher()
    dp.message.register(my_command, Command("my_command"))
    dp.message.register(echo, F.text)

    bot = Bot(token=bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
