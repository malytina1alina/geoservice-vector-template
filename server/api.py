# Серверная часть векторного сервиса: программный интерфейс (API) для расчёта по выбранному городу в двух режимах.
# Отдельный запуск: uvicorn server.api:app --port 8100 (обычно серверная часть работает внутри страницы main.py).
import json, time, traceback
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shapely.geometry import mapping
from server.osm_features import CITIES, place, MAX_AREA_KM2
from server.analysis import connect, load_model, analyze, train_on_the_fly, add_addresses
from extensions import extension

app = FastAPI(title="Геоинформационный сервис оценки потенциала размещения объектов")  # Название в документации API.
connect()  # Earth Engine – один раз при запуске.
MODEL, CARD = load_model()  # Модель и карточка из папки models.


class AreaRequest(BaseModel):  # Запрос расчёта по своей области.
    city: str  # Название города или пользовательской области для отчёта.
    geometry: dict | None = None  # Полигон или мультиполигон GeoJSON в WGS 84.


class TrainRequest(AreaRequest):  # Запрос обучения на лету: та же область и теги.
    target: list[dict]  # Объекты темы: варианты тегов, в варианте – все условия.
    extra_env: list[dict] = []  # Дополнительные факторы окружения: каждый словарь – признак.


@app.get("/health")  # Состояние сервиса.
def health():  # Тема, модель, метрики и пределы запроса.
    # Ключи – часть ответа API; limits – предел площади города не из справочника.
    return {"status": "ok", "topic": CARD["topic"], "model": CARD["model"], "approbation": CARD.get("approbation", []),
            "target": CARD["target"], "extra_env": CARD["extra_env"], "train_cities": CARD["train_cities"],
            "metrics": CARD.get("metrics", {}), "limits": {"max_area_km2": MAX_AREA_KM2}}


@app.get("/cities")  # Список городов.
def cities():  # Для выбора города на странице.
    return sorted(CITIES)  # Города справочника по алфавиту.


@app.get("/place")  # Граница города.
@lru_cache(maxsize=64)  # Границы 64 последних городов – из памяти.
def city_boundary(city: str):
    """Граница города для карты клиентской части: из справочника или поиском по названию в OpenStreetMap."""
    try:
        return {"city": city, "geometry": mapping(place(norm(city))[1])}  # Граница в GeoJSON.
    except ValueError as error:  # Город не найден или слишком велик.
        raise HTTPException(400, str(error))  # 400 – название нужно уточнить.


def to_json(gdf):  # Слой – в словарь GeoJSON для ответа.
    return json.loads(gdf.reset_index().to_json(drop_id=True))  # GeoJSON-словарь, индекс – свойством объекта.


def response(cells, candidates, targets, summary):
    """Ответ сервиса: сводка, слои карты и результат собственной функции."""
    try:
        ext = extension(cells, targets, summary["город"])  # Дополнительный расчёт по готовым данным.
    except Exception as error:  # Ошибка собственной функции не ломает основной расчёт.
        ext = {"name": "Собственная функция", "text": f"Ошибка: {type(error).__name__}: {error}"}  # Причина – вместо результата.
    # Объекты темы с подсказками: название, часы, адрес.
    marks = targets.reset_index()[["geometry"] + [c for c in ("name", "opening_hours", "addr:street", "addr:housenumber")
                                                  if c in targets]]
    return {"summary": summary,  # Сводка и слои карты; здания темы – точкой внутри контура.
            "cells": to_json(cells[["объектов темы", "ожидаемое", "потенциал", "вне диапазона", "население", "geometry"]]),
            "candidates": to_json(add_addresses(candidates)[["адрес", "ожидаемое", "объектов темы", "потенциал", "geometry"]]),
            "targets": to_json(marks.assign(geometry=marks.geometry.representative_point())),
            "extension": {"name": ext.get("name"), "text": ext.get("text"), "column": ext.get("column"),
                          "layer": None if ext.get("layer") is None else to_json(ext["layer"]),
                          "table": None if ext.get("table") is None else ext["table"].to_dict(orient="records")}}


def norm(city):  # Название города: без пробелов по краям; неразрывный дефис («Санкт‑Петербург» из документов) – обычный.
    return city.strip().replace("‑", "-").replace("‐", "-")


def guarded(compute, *args):
    """Расчёт с ответом HTTP 400 на понятную ошибку и 500 – на непредвиденную."""
    try:
        return compute(*args)  # Ответ сервиса.
    except ValueError as error:  # Город не найден, слишком велик или мало объектов темы.
        raise HTTPException(400, str(error))  # 400 – запрос нужно изменить.
    except Exception as error:  # Причина и место ошибки – в тексте ответа.
        # 500 – с последними строками следа ошибки.
        traceback.print_exc()  # Подробности – в журнал серверной части, не в ответ.
        raise HTTPException(500, f"{type(error).__name__}: {error}")  # Тип и текст ошибки.


def key(value):  # Ключ кэша из словаря запроса.
    # lru_cache принимает только неизменяемые значения.
    return None if value is None else json.dumps(value, sort_keys=True, ensure_ascii=False)


def parsed(text):  # Обратное преобразование ключа кэша.
    return None if text is None else json.loads(text)  # Словарь из строки кэша.


def timed(compute, *args, **kwargs):
    """Ответ сервиса со временем расчёта в сводке."""
    start = time.time()  # Начало расчёта.
    answer = response(*compute(*args, **kwargs))  # Расчёт и сборка ответа.
    answer["summary"]["время расчёта, с"] = round(time.time() - start)  # Повторный запрос из памяти – то же значение.
    return answer  # Хранится в кэше ответов.


@lru_cache(maxsize=8)  # Последние 8 ответов – из памяти.
def run_model(city, geometry=None):  # Расчёт своей моделью.
    return timed(analyze, city, MODEL, CARD, geometry=parsed(geometry))  # Своя модель: город или область.


@lru_cache(maxsize=8)  # Последние 8 ответов – из памяти.
def run_training(city, target, extra_env, geometry=None):  # Расчёт с обучением на лету.
    return timed(train_on_the_fly, city, parsed(target), parsed(extra_env), geometry=parsed(geometry))  # Обучение на лету.


@app.get("/analyze")  # Город по названию.
def analyze_city(city: str):
    """Режим «своя модель»: обученная модель из models/ применяется к любому городу. Повторный запрос берётся из памяти."""
    return guarded(run_model, norm(city))  # Город из справочника или поиском по названию.


@app.post("/analyze")  # Своя область GeoJSON.
def analyze_area(req: AreaRequest):  # Название и граница области.
    return guarded(run_model, norm(req.city), key(req.geometry))  # Своя область из GeoJSON.


@app.post("/train")  # Обучение на лету.
def train_city(req: TrainRequest):
    """Режим «обучение на лету»: модель обучается по выбранному городу и тегам, прогноз – вне обучения."""
    return guarded(run_training, norm(req.city), key(req.target), key(req.extra_env), key(req.geometry))  # Город или область.
