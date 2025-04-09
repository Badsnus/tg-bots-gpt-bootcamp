import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    TextSearchIndexType
)

load_dotenv()


def main():
    sdk = YCloudML(
        folder_id=os.getenv("YANDEX_FOLDER_ID"),
        auth=os.getenv("YANDEX_API_KEY"),
    )

    file = sdk.files.upload("./data/grant.md")

    operation = sdk.search_indexes.create_deferred(
        [file],
        index_type=TextSearchIndexType(),
    )
    text_index = operation.wait()
    text_tool = sdk.tools.search_index(text_index)
    model = sdk.models.completions("yandexgpt", model_version="rc")
    assistant = sdk.assistants.create(model, tools=[text_tool])
    text_index_thread = sdk.threads.create()

    query = "НАПИШИ СЮДА ЧТО-НИБУДЬ"
    text_index_thread.write(query)
    run = assistant.run(text_index_thread)
    result = run.wait().message
    for part in result.parts:
        print(part)


if __name__ == "__main__":
    main()
