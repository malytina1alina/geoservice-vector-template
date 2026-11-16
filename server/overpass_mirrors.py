# Выбор самого быстрого зеркала Overpass API и переход на запасные зеркала при сбое или перегрузке для запросов osmnx.
import time
import threading
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, as_completed, wait

import requests
import osmnx as ox

# Бесплатные публичные серверы Overpass API с данными по всему миру (значит, и по России); адрес – без "/interpreter".
# Список сверен с https://wiki.openstreetmap.org/wiki/Overpass_API (раздел «Public Overpass API instances»).
# Зеркала: VK Карты (Россия, без лимита запросов), два сервера FOSSGIS, Private.coffee (бывший overpass.kumi.systems).
OVERPASS_MIRRORS = [  # Самое быстрое выбирается проверкой.
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://lz4.overpass-api.de/api",
    "https://z.overpass-api.de/api",
    "https://overpass.private.coffee/api",
]
# Общий адрес https://overpass-api.de/api не добавляем: он перенаправляет на один из двух серверов выше,
# и лишний одновременный запрос только занял бы ещё один слот из лимита FOSSGIS на наш IP-адрес.

BUSY_CODES = (429, 502, 503, 504)  # «Сервер занят»: запрос стоит отправить на другое зеркало.
ALIVE_MIRRORS = []  # Зеркала со свежими данными в порядке ответа на проверку.
ALL_MIRRORS = []  # Все зеркала, которые можно взять запасными.
HEDGE = {"delay": 10.0, "max_age_days": 7}  # Через сколько секунд дублировать запрос; какие данные свежие.
_POOL = ThreadPoolExecutor(max_workers=16)  # Потоки для одновременных запросов к зеркалам.


def _data_age_days(answer):
    """Возраст данных зеркала в днях по метке времени в ответе Overpass API (None, если метки нет)."""
    try:
        stamp = dt.datetime.strptime(answer["osm3s"]["timestamp_osm_base"], "%Y-%m-%dT%H:%M:%SZ")  # Время данных, UTC.
    except (KeyError, ValueError):  # Метки нет или она в другом виде.
        return None  # Возраст неизвестен.
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)  # Текущее время UTC без пояса, как у метки.
    return (now - stamp).total_seconds() / 86400  # Секунды в сутки.


def _probe_mirror(base_url, query, headers, deadline, stop):
    """Проверочный запрос к зеркалу; пока сервер занят – повтор до deadline или до выбора другого зеркала (stop)."""
    result = {"url": base_url, "ok": False, "reason": "не ответило вовремя"}  # Итог проверки зеркала.
    start = time.perf_counter()  # Отсчёт времени ответа.
    while not stop.is_set() and time.perf_counter() < deadline:  # Зеркало ещё не выбрано, время есть.
        try:
            # Проверочный запрос; ждём не дольше остатка времени.
            response = requests.post(base_url + "/interpreter", data={"data": query}, headers=headers,
                                     timeout=max(deadline - time.perf_counter(), 1))
        except requests.Timeout:  # Не ответило вовремя.
            return result  # Причина – прежняя.
        except requests.RequestException:
            return result | {"reason": "нет соединения"}  # Сервер недоступен.
        if response.status_code in BUSY_CODES:  # Сервер занят: пауза и повтор.
            result["reason"] = f"сервер занят (HTTP {response.status_code})"  # Причина, если время выйдет.
            time.sleep(2)  # Пауза перед повтором.
            continue  # Повтор запроса.
        if response.status_code != 200:  # Другая ошибка сервера.
            return result | {"reason": f"HTTP {response.status_code}"}  # Код ошибки.
        try:
            answer = response.json()  # Ответ Overpass в JSON.
        except ValueError:
            return result | {"reason": "ответ не в формате JSON"}  # Пришла страница вместо данных.
        if "error" in (answer.get("remark") or "").lower():  # Ошибка выполнения приходит с кодом 200 в поле "remark".
            return result | {"reason": "ошибка сервера: " + answer["remark"][:80]}  # Начало текста ошибки.
        elements = answer.get("elements", [])  # Ответ на «out count» – один элемент с числом точек.
        result["objects"] = int(elements[0]["tags"]["total"]) if elements else 0  # Точек в области.
        if result["objects"] == 0:  # Региональное зеркало без данных по области.
            return result | {"reason": "нет данных по области"}  # Зеркало не подходит.
        return result | {"age_days": _data_age_days(answer), "time": time.perf_counter() - start, "ok": True}  # Пригодно.
    return result  # Время вышло или выбрано другое зеркало.


def _post_overpass(url, data, answered):
    """Запрос OSMnx к одному зеркалу: ответ или None. Если другое уже ответило (answered), о неудаче молчим."""
    def fail(reason):  # Печать причины; функция вернёт None.
        if not answered.is_set():  # Сообщаем, только пока ответа нет.
            print(f"  {url}: {reason}")  # Причина неудачи зеркала.

    try:
        response = requests.post(url + "/interpreter", data=data, timeout=ox.settings.requests_timeout,  # Тот же запрос, что делает OSMnx.
                                 headers=ox._http._get_http_headers(), **ox.settings.requests_kwargs)
        if response.status_code in BUSY_CODES:  # Зеркало занято.
            return fail(f"занято (HTTP {response.status_code})")  # None: пробуем другое.
        answer = ox._http._parse_response(response)  # Разбор ответа средствами OSMnx.
    # Сбой соединения или ответа.
    except (requests.RequestException, ox._errors.InsufficientResponseError, ox._errors.ResponseStatusCodeError) as error:
        return fail(type(error).__name__)  # Имя ошибки.
    if "runtime error" in (answer.get("remark") or ""):  # Сервер прервал запрос по времени или памяти: ответ неполный.
        return fail("запрос прерван сервером")  # Неполный ответ не принимаем.
    age = _data_age_days(answer)  # Устаревшие данные или без метки времени не принимаем.
    if age is None or age > HEDGE["max_age_days"]:  # Свежесть не подтверждена.
        return fail("свежесть данных не подтверждена")  # Ждём ответа другого зеркала.
    return answer  # Полный ответ со свежими данными.


# Стандартная функция OSMnx при ответе «сервер занят» ждёт 55 с и повторяет запрос на том же сервере, а неполный.
# ответ сохраняет в кэш как нормальный. Замена отправляет запрос самому быстрому зеркалу; если оно занято или молчит.
# HEDGE["delay"] с, запрос дублируется на следующее (сначала ответившие на проверку, затем остальные),
# и принимается первый полный ответ со свежими данными.
def _overpass_request_with_failover(data, max_wait=900):
    """Замена внутренней функции OSMnx для запроса к Overpass API: переход на запасные зеркала."""
    # Ключ кэша не зависит от зеркала: при повторном запуске ответ берётся из кэша, какое бы зеркало его ни дало.
    cache_key = requests.Request("GET", "https://overpass-api.de/api/interpreter", params=data).prepare().url
    cached = ox._http._retrieve_from_cache(cache_key)  # Ответ из кэша OSMnx.
    if isinstance(cached, dict):  # Есть сохранённый ответ.
        return cached  # Без запроса к серверу.
    order = list(ALIVE_MIRRORS) or [ox.settings.overpass_url]  # Сначала ответившие на проверку.
    order += [url for url in ALL_MIRRORS if url not in order]  # Затем остальные.
    queue = list(order)  # Зеркала, которым запрос ещё не отправлен.
    running = {}  # Запрос -> зеркало.
    answered = threading.Event()  # Ответ получен.
    deadline = time.perf_counter() + max_wait  # Общий предел ожидания.
    while time.perf_counter() < deadline:  # Пока не вышло время.
        if queue:  # Подключаем следующее зеркало.
            url = queue.pop(0)  # Первое из очереди.
            running[_POOL.submit(_post_overpass, url, data, answered)] = url  # Запрос в отдельном потоке.
        if not running:  # Все зеркала ответили «занято».
            print("  все зеркала заняты, повторный запрос через 30 с")  # Сообщение в журнал.
            time.sleep(30)  # Так просит политика overpass-api.de.
            queue = list(order)  # Снова по всем зеркалам.
            continue  # Новый круг.
        # Пока есть запасные зеркала, ждём ответа не дольше HEDGE["delay"] с, затем подключаем следующее.
        done, _ = wait(running, timeout=HEDGE["delay"] if queue else 30, return_when=FIRST_COMPLETED)
        for future in done:  # Завершившиеся запросы.
            running.pop(future)  # Больше не ждём.
            if future.result() is not None:  # Первый полный ответ принимается.
                answered.set()  # Остальным – не сообщать о неудаче.
                ox._http._save_to_cache(cache_key, future.result(), True)  # Сохраняем в кэш OSMnx.
                return future.result()  # Ответ для OSMnx.
        if not queue and not done:  # Оставшиеся молчат 30 с: снова пробуем ответившие «занято».
            queue = [url for url in order if url not in running.values()]  # Кроме зависших.
    answered.set()  # Остальные неудачи больше не печатаем.
    raise RuntimeError("Зеркала Overpass API перегружены. Повторите запуск ячейки через несколько минут")  # Все попытки исчерпаны.


def select_overpass_mirror(point=(55.7520, 37.6175), wait_seconds=120, max_age_days=7,
                           requests_timeout=300, max_query_area_km2=10_000):
    """Зеркало, первым вернувшее корректный ответ по окрестности point (широта, долгота); остальные – запасные."""
    # Запрос уходит всем зеркалам сразу, занятые опрашиваются повторно, общее ожидание не дольше wait_seconds.
    mirrors = list(OVERPASS_MIRRORS)  # Публичные зеркала.
    # Проверочный запрос: число точек OSM в квадрате ~400 м вокруг point – у зеркала есть данные по территории.
    # Малые объявленные лимиты (5 с, 1 МБ) помогают быстрее получить слот в очереди сервера.
    lat, lon = point  # Центр квадрата.
    d = 0.002  # Полсторона квадрата в градусах.
    query = f"[out:json][timeout:5][maxsize:1048576];node({lat - d},{lon - d},{lat + d},{lon + d});out count;"  # Число точек.
    headers = ox._http._get_http_headers()  # Те же заголовки, с которыми потом обращается osmnx.

    def remember_alive(future):  # Зеркало со свежими данными – в запасные.
        r = future.result()  # Итог проверки зеркала.
        if r["ok"] and r["age_days"] is not None and r["age_days"] <= max_age_days:  # Свежие данные.
            ALIVE_MIRRORS.append(r["url"])  # В запасные в порядке ответа.

    print(f"Опрос зеркал Overpass API: всего {len(mirrors)}, ожидание ответа не дольше {wait_seconds} с...")  # Начало опроса.
    ALIVE_MIRRORS.clear()  # Новый опрос – новый порядок.
    ALL_MIRRORS[:] = mirrors  # Список меняется на месте: его читают другие функции.
    HEDGE["max_age_days"] = max_age_days  # Порог свежести и для запросов OSMnx.
    deadline = time.perf_counter() + wait_seconds  # Общий предел ожидания.
    stop = threading.Event()  # Сигнал проверкам: зеркало выбрано.
    futures = [_POOL.submit(_probe_mirror, url, query, headers, deadline, stop) for url in mirrors]  # Все сразу.
    for future in futures:  # Запасные пополняются в порядке ответа.
        future.add_done_callback(remember_alive)  # Вызов по завершении проверки.

    best = None  # Первое зеркало с корректным ответом – самое быстрое.
    for future in as_completed(futures):  # Проверки в порядке завершения.
        r = future.result()  # Итог проверки.
        if r["ok"]:  # Зеркало ответило правильно.
            days = "неизвестен" if r["age_days"] is None else f"{r['age_days']:.1f} дн.".replace(".", ",", 1)  # Возраст.
            seconds = f"{r['time']:.1f}".replace(".", ",")  # Десятичная запятая.
            print(f"  {r['url']} – ответ за {seconds} с, точек в области: {r['objects']}, возраст данных {days}")  # Итог.
            best = r  # Выбранное зеркало.
            break  # Остальных не ждём.
        print(f"  {r['url']} – {r['reason']}")  # Причина отказа зеркала.

    stop.set()  # Занятые больше не опрашиваем; отправленные проверки пополнят запасные.
    if best is None:  # Ни одно зеркало не ответило.
        # Без ответа зеркал данные OSM не загрузить.
        raise RuntimeError(f"За {wait_seconds} с ни одно зеркало Overpass API не ответило: все публичные серверы "
                           "сейчас перегружены. Повторите запуск ячейки через несколько минут")
    if best["age_days"] is None or best["age_days"] > max_age_days:  # Зеркало выбрано, но данные могут быть старыми.
        # Предупреждение, расчёт продолжается.
        print(f"Свежесть данных выбранного зеркала не подтверждена (старше {max_age_days} дн. или неизвестна)")
    HEDGE["delay"] = max(10.0, 3 * best["time"])  # Подстраховка – после трёх «обычных» времён ответа, не раньше 10 с.

    ox.settings.overpass_url = best["url"]  # Сервер запросов.
    ox.settings.requests_timeout = requests_timeout  # Предел ожидания ответа.
    ox.settings.overpass_rate_limit = False  # Без ожидания слотов /status: занятость обрабатывает переход ниже.
    # По умолчанию OSMnx режет области больше 2500 км² на квадраты и запрашивает их по очереди. Для выборки по тегам.
    # это лишние запросы: например, вся Москва (выпуклая оболочка ~4900 км²) одним запросом скачивается втрое быстрее.
    ox.settings.max_query_area_size = max_query_area_km2 * 1e6
    ox._overpass._overpass_request = _overpass_request_with_failover  # Подмена функции запроса: переход на запасные.
    print(f"\nВыбрано зеркало: {best['url']}")  # Итог выбора.
    return best["url"]  # Адрес выбранного зеркала.
