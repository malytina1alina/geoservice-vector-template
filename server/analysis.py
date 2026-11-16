# Расчёт в двух режимах: «своя модель» – обученная модель из папки models применяется к выбранному городу;
# «обучение на лету» – модель обучается по городу и тегам, ожидаемое число – прогноз вне обучения по
# пространственным блокам. Население ячеек – GHS-POP в Earth Engine, адреса мест-кандидатов – Nominatim.
import json, os, threading, time
from functools import lru_cache
import ee, joblib, numpy as np, pandas as pd, h3, requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from server.osm_features import city_cells, polygon_cells


def connect():
    """Ключ сервисного аккаунта берётся из секрета EE_SERVICE_ACCOUNT (содержимое JSON-ключа), проект – из EE_PROJECT."""
    key = os.environ.get("EE_SERVICE_ACCOUNT")  # Содержимое JSON-ключа или ничего.
    if key:  # Запуск на сервере с сервисным аккаунтом.
        info = json.loads(key)  # Почта аккаунта и проект по умолчанию.
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=key)  # Вход по ключу.
        ee.Initialize(credentials, project=os.environ.get("EE_PROJECT", info["project_id"]))  # Проект из EE_PROJECT или ключа.
    else:
        ee.Initialize(project=os.environ.get("EE_PROJECT"))  # Локальный запуск: сохранённая авторизация пользователя.


def population(cells, year=2020, chunk=2000):
    """Сумма GHS-POP в каждой ячейке (в Earth Engine частями, чтобы не превысить размер запроса)."""
    ghs = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()  # Снимок года.
    values = {}  # Ячейка: население.
    for start in range(0, len(cells), chunk):  # Части по chunk ячеек.
        part = cells.iloc[start:start + chunk].reset_index()[["cell", "geometry"]]  # Только номер и граница.
        features = json.loads(part.to_json(drop_id=True))  # Ячейки в GeoJSON.
        for feature in features["features"]:  # Каждая ячейка.
            feature["geometry"]["geodesic"] = True  # Стороны – линии по поверхности Земли, а не по плоской проекции.
        result = ghs.reduceRegions(ee.FeatureCollection(features), ee.Reducer.sum(), 100).getInfo()  # Сумма на сетке 100 м.
        values |= {f["properties"]["cell"]: f["properties"].get("sum", 0) for f in result["features"]}  # Нет суммы – 0.
    return cells.index.map(values).fillna(0).round(0)  # В порядке ячеек, целые люди.


MODEL_DIR = os.environ.get("MODEL_DIR", "models/vector_model")  # Папка модели и её карточки.
MIN_TARGETS = 20  # Меньше объектов темы в городе – модель на лету не обучается.
CV_FOLDS = 5  # Число частей проверки по пространственным блокам.


def load_model(model_dir=MODEL_DIR):
    """Модель сервиса и её карточка из папки models."""
    import sklearn
    with open(os.path.join(model_dir, "card.json"), encoding="utf-8") as file:  # Карточка: тема, теги, признаки.
        card = json.load(file)  # Словарь карточки.
    if card.get("scikit_learn") and card["scikit_learn"] != sklearn.__version__:  # Модель сохранена другой версией scikit-learn.
        # Предупреждение в журнал сервера: другая версия может читать модель иначе.
        print(f"Внимание: модель сохранена scikit-learn {card['scikit_learn']}, установлена {sklearn.__version__} – "
              "установите версию из requirements.txt")
    return joblib.load(os.path.join(model_dir, "model.joblib")), card  # Модель и карточка.


def finish(cells, city, n_candidates):
    """Потенциал, места-кандидаты и сводка – общие для обоих режимов. Сумма и отбор – по неокруглённым значениям."""
    cells["потенциал"] = cells["ожидаемое"] - cells["объектов темы"]  # Сколько объектов ещё может появиться.
    candidates = cells[~cells["вне диапазона"]].nlargest(n_candidates, "потенциал").copy()  # Лучшие ячейки с надёжным прогнозом.
    summary = {"город": city, "ячеек": len(cells), "объектов темы": int(cells["объектов темы"].sum()),  # Сводка по городу или области.
               "ожидаемое число": round(float(cells["ожидаемое"].sum()), 1),
               "ячеек вне диапазона обучения, %": round(100 * float(cells["вне диапазона"].mean()), 1)}
    for table in (cells, candidates):  # Округление только для показа.
        table[["ожидаемое", "потенциал"]] = table[["ожидаемое", "потенциал"]].round(3)  # Три знака после запятой.
    return cells, candidates, summary  # Ячейки, кандидаты и сводка.


def analyze(city, model, card, n_candidates=10, *, geometry=None):
    """Режим «своя модель»: ячейки города с ожидаемым числом, потенциалом и признаком выхода за диапазон обучения."""
    cells, targets = (city_cells(city, card, population) if geometry is None  # Город из списка или своя область.
                      else polygon_cells(geometry, card, population))
    X = cells[card["features"]]  # Признаки в порядке обучения.
    cells["ожидаемое"] = np.clip(model.predict(X.values), 0, None)  # Число объектов не бывает отрицательным.
    low, high = pd.Series(card["feature_min"]), pd.Series(card["feature_max"])  # Диапазоны признаков в городах обучения.
    cells["вне диапазона"] = ((X < low[card["features"]]) | (X > high[card["features"]])).any(axis=1)  # Хотя бы один признак вне.
    cells, candidates, summary = finish(cells, city, n_candidates)  # Потенциал, кандидаты, сводка.
    summary |= {"режим": "своя модель", "модель": card["model"], "города обучения": ", ".join(card["train_cities"])}  # Чья модель.
    return cells, candidates, targets, summary  # Слои карты и сводка.


def train_on_the_fly(city, target, extra_env, n_candidates=10, *, geometry=None):
    """Режим «обучение на лету»: признаки по базовым категориям и своим факторам, две модели, проверка по блокам."""
    card = {"target": target, "extra_env": extra_env, "h3_resolution": 9, "features": []}  # Карточка без обученной модели.
    cells, targets = (city_cells(city, card, population) if geometry is None  # Блоки своей области – до запросов.
                      else polygon_cells(geometry, card, population, folds=CV_FOLDS))
    found = int(cells["объектов темы"].sum())  # Объектов темы в области.
    if found < MIN_TARGETS:  # Слишком мало для обучения.
        raise ValueError(f"В области «{city}» объектов с такими тегами {found}, а для обучения "  # Сообщение с числом найденных объектов.
                         f"нужно не меньше {MIN_TARGETS}. Выберите другие теги или режим «своя модель».")
    features = [c for c in cells.columns if c not in ("geometry", "объектов темы")]  # Все признаки окружения и население.
    X, y = cells[features].values, cells["объектов темы"].values  # Признаки и фактическое число.
    blocks = [h3.cell_to_parent(c, 7) for c in cells.index]  # Блок проверки – ячейка H3 разрешения 7.
    if len(set(blocks)) < CV_FOLDS:  # Блоков меньше, чем частей проверки.
        raise ValueError(f"Для обучения нужно не меньше {CV_FOLDS} пространственных блоков H3. "  # Область слишком мала для проверки.
                         "Увеличьте область или выберите режим «Своя модель».")
    splits = list(GroupKFold(CV_FOLDS).split(X, y, groups=blocks))  # Одни и те же части для обеих моделей.
    if any(y[train].sum() == 0 for train, _ in splits):  # В обучающей части нет ни одного объекта темы.
        raise ValueError("Объекты темы сосредоточены в одной части проверки. "  # Без объектов в обучающей части модель не обучится.
                         "Выберите область с объектами темы в нескольких пространственных блоках.")
    # Регуляризация бустинга: объектов мало, без неё прогноз вне обучения занижен в разы.
    models = {"Регрессия Пуассона": make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                                  PoissonRegressor(alpha=1e-3, max_iter=1000)),
              "Градиентный бустинг": HistGradientBoostingRegressor(
                  loss="poisson", learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=42)}
    metrics, predictions = {}, {}  # Метрики и прогнозы вне обучения по моделям.
    for name, model in models.items():  # Каждая модель – на тех же частях.
        p = np.clip(cross_val_predict(model, X, y, cv=splits), 1e-6, None)  # Прогноз для ячеек, которых модель не видела.
        top = y[np.argsort(-p)[:len(y) // 10]].sum() / y.sum() * 100  # Доля объектов в 10 % лучших по прогнозу ячеек.
        metrics[name] = {"отклонение Пуассона": round(mean_poisson_deviance(y, p), 3),  # Качество вне обучения.
                         "MAE": round(mean_absolute_error(y, p), 3), "объектов в 10 % ячеек, %": round(top, 1)}
        predictions[name] = p  # Прогноз модели.
    best = min(metrics, key=lambda n: metrics[n]["отклонение Пуассона"])  # Лучшая – с наименьшим отклонением.
    cells["ожидаемое"] = predictions[best]  # Прогноз вне обучения: ячейку не видела модель.
    cells["вне диапазона"] = False  # Модель обучена на этом же городе.
    cells, candidates, summary = finish(cells, city, n_candidates)  # Потенциал, кандидаты, сводка.
    summary |= {"режим": "обучение на лету", "модель": best, "города обучения": city, "metrics": metrics}  # Метрики обеих моделей.
    return cells, candidates, targets, summary  # Слои карты и сводка.


NOMINATIM = "https://nominatim.openstreetmap.org/reverse"  # Адрес по координатам.
PROJECT = "https://github.com/tester170/geoservice-vector-template"  # Адрес проекта для подписи запросов.
# Подпись с адресом проекта: без неё Nominatim отклоняет запросы с облачных серверов.
HEADERS = {"User-Agent": f"GeoMarketing Analysis ({PROJECT})", "Referer": PROJECT}
LAST, LOCK = [0.0], threading.Lock()  # Время прошлого запроса; запросы по очереди.


@lru_cache(maxsize=4096)  # Повтор точки – без нового запроса.
def address(lat, lon):
    """Адрес точки: улица и дом, район; пустая строка, если адрес не найден."""
    with LOCK:  # Правило сервиса Nominatim: не больше одного запроса в секунду.
        time.sleep(max(0.0, LAST[0] + 1.1 - time.monotonic()))  # Ждать только остаток до 1,1 с.
        LAST[0] = time.monotonic()  # Начало этого запроса.
    try:
        # Адрес с точностью до дома, на русском языке.
        found = requests.get(NOMINATIM, timeout=20, headers=HEADERS,  # Подпись приложения.
                             params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "accept-language": "ru"}).json()
    except (requests.RequestException, ValueError):  # Нет связи или ответ не JSON.
        return ""  # Адрес не найден.
    a = found.get("address", {})  # Части адреса.
    street = a.get("road") or a.get("pedestrian") or a.get("neighbourhood") or ""  # Улица или квартал.
    house = a.get("house_number", "")  # Номер дома.
    district = a.get("city_district") or a.get("suburb") or ""  # Район города.
    text = ", ".join(x for x in [street, house] if x)  # «Улица, дом» без пустых частей.
    return f"{text} ({district})" if text and district else text or district  # Район – в скобках.


def add_addresses(gdf):
    """Столбец «адрес» по центрам ячеек (координаты округлены до 5 знаков – около 1 м)."""
    return gdf.assign(адрес=[address(round(p.y, 5), round(p.x, 5)) for p in gdf.geometry.centroid])  # Повтор точки – из памяти.
