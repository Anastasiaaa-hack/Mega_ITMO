from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral 
from urllib.parse import urljoin
import re
from typing import Optional

app = FastAPI()

# Инициализация клиента Mistral API
mistral_client = Mistral(api_key="i0nKOaw2v8VXp3PVzxnOauTAmY7Dl0d7")

class PredictionRequest(BaseModel):
    query: str
    id: int

class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int] = None
    reasoning: str
    sources: List[HttpUrl]

@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        query = request.query
        request_id = request.id

        # Генерация ответа с помощью Mistral
        response = mistral_client.chat.complete(
            model="mistral-tiny",
            messages=[{"role": "user", "content": query}]
        )


        # Проверяем, есть ли ответы в choices
        if not response.choices:
            raise HTTPException(status_code=500, detail="Mistral API вернул пустой ответ.")

        reasoning = response.choices[0].message.content  # Корректный доступ к тексту ответа

        # Поиск ссылок в интернете
        search_results = search_web(query)
        sources = search_results[:3] 

        # Парсинг новостей
        news_links = get_latest_news()
        sources.extend(news_links)

        # Формирование ответа
        answer = extract_answer_from_options(query, reasoning)

        return PredictionResponse(
            id=request_id,
            answer=answer,
            reasoning="Ответ сгенерирован с помощью модели Mistral AI. "+reasoning,
            sources=sources[:3],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

def search_web(query: str) -> List[str]:
    return ["https://itmo.ru/ru/", "https://abit.itmo.ru/"]

def get_latest_news() -> List[str]:
    base_url = "https://news.itmo.ru"
    url = f"{base_url}/ru/"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Проверяем HTTP-статус
        soup = BeautifulSoup(response.text, 'html.parser')
        # Используем urljoin для обработки относительных ссылок
        news_links = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True) if 'news' in a['href']]
        return news_links[:3] 
    except requests.RequestException:
        return []

def extract_answer_from_options(query: str, reasoning: str) -> Optional[int]:
    """
    Универсальная функция для извлечения номера правильного ответа из текста reasoning.
    Работает как с числами, так и с текстовыми вариантами.
    """
    # Извлекаем варианты ответов из query 
    options = re.findall(r'(\d+)\.\s*(.*)', query)  # Захватываем номера и текст или числа

    # Приводим reasoning к нижнему регистру для упрощения поиска
    reasoning_lower = reasoning.lower()

    # Проверяем, есть ли явное указание номера ответа в reasoning 
    explicit_answer_match = re.search(r'(\d+)\.\s*(.*)', reasoning)
    if explicit_answer_match:
        explicit_number = int(explicit_answer_match.group(1))
        # Проверяем, что номер ответа существует в вариантах
        if any(number == str(explicit_number) for number, _ in options):
            return explicit_number

    # Если явного указания нет, ищем точные совпадения (полные фразы)
    for number, option in options:
        option_cleaned = option.strip().lower()
        if option_cleaned in reasoning_lower:
            return int(number)

    # Если точных совпадений нет, ищем числовые значения (например, год)
    match = re.search(r'(\d{4})', reasoning_lower)  
    if match:
        reasoning_year = match.group(1) 
        for number, option in options:
            option_cleaned = option.strip().lower()
            if reasoning_year in option_cleaned:
                return int(number)

    # Если числовых совпадений нет, ищем частичные совпадения (ключевые слова)
    for number, option in options:
        option_cleaned = option.strip().lower()
        # Разбиваем вариант на слова и проверяем, есть ли хотя бы одно слово в reasoning
        if any(word in reasoning_lower for word in option_cleaned.split()):
            return int(number)

    return None  # Если не нашли правильный ответ, возвращаем None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
