import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)

model = sdk.models.text_classifiers("yandexgpt").configure(
    task_description="извлеки несколько labels из текста",
    labels=[
        "спорт",
        "наука",
        "IT",
        "c#"
    ],
)


predictions = model.run("""
В данной статье мы рассмотрим почему c# лучшем чем python.
Это статья не о спорте и не о науке, а о самом лучше ЯП в мире - то есть о c# и платформе .NET!
""")

tags = []
for predict in predictions:
    if predict.confidence > 0.05:
        tags.append(predict.label)

print(tags)
