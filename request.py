import requests
import csv
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
REPORT_FILE = "inference_report.csv"

prompts = [
    "Расскажи про игру dying light от компании techland в паре предложений",
    "Расскажи про движок Unity",
    "В чем кардинальное отличие в Unity и Unreal Engine. Назови 3 главных отличия",
    "Как ты относишься к LLM?",
    "В каком году вышел Fallout 4?",
    "Как ты относишься к c++?",
    "Что такое Python?",
    "Назови все поисковые сайты",
    "Как приручить дракона?",
    "Что такое огурец?",
]


def send_prompt(prompt: str, model: str = MODEL_NAME) -> str:
    #Отправляет промпт в Ollama API и возвращает текстовый ответ модели
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def run_inference(prompts_list: list[str]) -> list[dict]:
    #Последовательно отправляет список промптов в модель и собирает результаты
    results = []
    for i, prompt in enumerate(prompts_list, start=1):
        answer = send_prompt(prompt)
        results.append({"prompt": prompt, "response": answer})
    return results


def save_report(results: list[dict], filename: str = REPORT_FILE) -> None:
    #Сохраняет результаты инференса в CSV-файл
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Запрос к LLM", "Вывод LLM"])
        for row in results:
            writer.writerow([row["prompt"], row["response"]])


def main():
    #Точка входа: запускает инференс по списку промптов и сохраняет отчёт в CSV
    print(f"Модель: {MODEL_NAME}")
    print(f"Сервер: {OLLAMA_URL}")
    print(f"Количество запросов: {len(prompts)}")

    results = run_inference(prompts)
    save_report(results)

    print("Конец")


if __name__ == "__main__":
    main()
