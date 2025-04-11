import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)

model = sdk.models.text_classifiers("yandexgpt").configure(
    task_description="определи тип сообщения",
    labels=[
        "спам",
        "не спам",
    ],
)


result = model.run("Продам гараж, недорого, в лс")

print(max(result, key=lambda x: x.confidence).label)
