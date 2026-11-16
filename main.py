# Клиентская часть векторного сервиса GeoMarketing Analysis: город или своя область, своя модель или обучение на лету
# по тегам OpenStreetMap, карта потенциала и мест-кандидатов, таблицы, файлы и HTML-отчёт.
# Запуск: streamlit run main.py. Серверная часть работает внутри приложения; если она развёрнута
# отдельно, её адрес задаётся переменной окружения или секретом API_URL.
import json, os, sys, tempfile  # Настройки, пути и временные папки.
from pathlib import Path  # Чтение файлов рядом с модулем.
import requests, pandas as pd, geopandas as gpd, streamlit as st  # Запросы, таблицы, слои и страница.
from streamlit_folium import st_folium  # Карта folium на странице.
from client.maps_and_report import (base_map, add_boundary, full_map, finish, drawing_tools, drawing_mode, remember_view,  # Карты и слои.
                  result_layer, candidate_layer)
from client.maps_and_report import build_report, tags_text  # HTML-отчёт и запись тегов.
from client.interface import page, fmt, map_height, drawn_area, upload_area  # Оформление страницы, нарисованные и загруженные области.

PRESETS = json.loads((Path(__file__).parent / "client" / "presets.json").read_text(encoding="utf-8"))  # Объекты и факторы тем.
OWN = "Свои теги…"  # Пункт списка для своих тегов.
# Настройки страницы и шапка сервиса.
page("GeoMarketing Analysis", "Оценка потенциала размещения объектов по данным OpenStreetMap", "hex", "GeoMarketing Analysis")


def secret(name):  # Значение из секретов Streamlit; без файла секретов – пусто.
    try:
        return str(st.secrets.get(name, ""))  # Secrets на Streamlit Community Cloud или .streamlit/secrets.toml.
    except Exception:  # Файла секретов нет (локальный запуск).
        return ""  # Значение не задано.


# Адрес отдельно развёрнутой серверной части; пусто – серверная часть работает внутри приложения.
API_URL = (os.environ.get("API_URL") or secret("API_URL")).rstrip("/")


SERVICE = Path(__file__).resolve().parent  # Папка сервиса.


def code_version():  # Время последнего изменения кода серверной части.
    files = [*SERVICE.glob("server/*.py"), SERVICE / "extensions.py"]  # Код серверной части и собственная функция.
    return max(f.stat().st_mtime for f in files)  # Обновили репозиторий – значение изменилось.


@st.cache_resource(show_spinner="Запуск серверной части…", max_entries=1)  # Один раз на версию кода.
def local_api(version):
    """Серверная часть (FastAPI) внутри этого приложения: запросы к ней идут без сети."""
    for name in ("EE_SERVICE_ACCOUNT", "EE_PROJECT"):  # Ключ и проект Earth Engine – из секретов.
        if secret(name) and name not in os.environ:  # Задан в Secrets, но ещё не в окружении.
            os.environ[name] = secret(name)  # Серверная часть читает его из окружения.
    for name in [m for m in sys.modules if m == "extensions" or m.split(".")[0] == "server"]:  # Прежний код.
        del sys.modules[name]  # Обновлённый код импортируется заново.
    from fastapi.testclient import TestClient  # Запросы к приложению FastAPI без сети.
    from server.api import app  # Серверная часть сервиса.
    return TestClient(app, raise_server_exceptions=False)  # Сбой расчёта – ответ 500, как у отдельного сервера.


def call(method, path, **kwargs):  # Запрос к серверной части: по адресу API_URL или внутри приложения.
    if API_URL:  # Серверная часть развёрнута отдельно.
        return requests.request(method, API_URL + path, **kwargs)  # Запрос по сети.
    return local_api(code_version()).request(method, path, **kwargs)  # Тот же запрос без сети.


def answer(response):
    """Данные ответа серверной части или ошибка с её пояснением."""
    if response.status_code >= 400:  # Сервер сообщил об ошибке.
        raise RuntimeError(response.json().get("detail", response.text))  # Текст ошибки из поля detail.
    return response.json()  # Данные ответа.


@st.cache_data(ttl=3600, show_spinner=False)  # Одинаковые запросы – из кэша в течение часа.
def get(path, **params):  # Запрос GET к серверной части.
    return answer(call("GET", path, params=params, timeout=1800))  # Параметры – в адресе.


def post(path, body):  # Запрос POST к серверной части.
    return answer(call("POST", path, json=body, timeout=3600))  # Тело запроса – JSON.


def once(name, build):
    """Карта, отчёт и файлы строятся один раз на расчёт, а не при каждом действии на странице."""
    key = (name, id(state.get("result")))  # Ключ – имя и текущий результат.
    built = state.setdefault("built", {})  # Уже построенное.
    if key not in built:  # Для этого результата ещё не строили.
        # Построенное для прежних результатов больше не нужно.
        state["built"] = built = {k: v for k, v in built.items() if k[1] == key[1]} | {key: build()}
    return built[key]  # Готовый объект.


def parse_tags(text):
    """Строка – вариант объекта или фактор; условия через «;», значения через «|», «*» – любое значение."""
    tags = []  # Варианты тегов.
    for line in text.splitlines():  # Каждая строка – вариант.
        pairs = [part.split("=", 1) for part in line.split(";") if "=" in part]  # Пары «ключ=значения».
        tag = {k.strip(): True if v.strip() == "*" else [x.strip() for x in v.split("|")] for k, v in pairs}  # Условия.
        if tag:  # Строки без условий пропускаются.
            tags.append(tag)  # Вариант в список.
    return tags  # Все варианты.


try:
    info, cities = get("/health"), get("/cities")  # Сведения о модели и города.
except (requests.RequestException, RuntimeError):  # Серверная часть не запущена или недоступна.
    st.error(f"Нет связи с серверной частью {API_URL}. Обновите страницу, когда она запустится.")  # Понятное сообщение вместо трассировки.
    st.stop()  # Без сведений о модели страница не строится.
except Exception as error:  # Серверная часть внутри приложения не запустилась.
    st.error(f"Серверная часть не запустилась: {error}. Проверьте ключ EE_SERVICE_ACCOUNT в Secrets.")  # Причина.
    st.stop()  # Без сведений о модели страница не строится.
state = st.session_state  # Состояние страницы.
state.setdefault("city_input", state.get("city_choice"))  # Город, введённый ранее.


def clear_map():  # Кнопка «Очистить карту».
    for key in ("active_area", "result", "params", "built"):  # Область, результат и файлы.
        state.pop(key, None)  # Ключа может не быть.
    state["city_input"] = state["city_choice"] = None  # Поле города – пустое.
    state["polygon_upload_revision"] = state.get("polygon_upload_revision", 0) + 1  # Новое поле загрузки – без файлов.


def select_area(area):  # Сделать область текущей.
    previous = state.get("active_area")  # Прежняя область.
    if area != previous:  # Выбрана другая область.
        saved_city = state.get("params", {}).get("city")  # Город последнего расчёта.
        if previous or area["method"] != "Город" or area["name"] != saved_city:  # Результат относится к другой области.
            for key in ("result", "params", "built"):  # Результат и файлы.
                state.pop(key, None)  # Ключа может не быть.
        state["active_area"] = area  # Новая область.
    if area["method"] == "Полигон":  # Полигон заменяет город.
        state["city_input"] = None  # Поле города – пустое.
        state.pop("city_choice", None)  # Сбросить выбор в списке.


left, right = st.columns([2.7, 1])  # Карта слева, параметры справа.
with right, st.container(border=True, gap="xsmall"):  # Блок параметров.
    title, clear = st.columns([5, 1], gap="xsmall", vertical_alignment="center")  # Заголовок и кнопка очистки.
    title.subheader("Параметры расчёта")  # Заголовок блока.
    area = state.get("active_area")  # Текущая область.
    with clear, st.container(horizontal=True, horizontal_alignment="right"):  # Кнопка у правого края.
        st.button("Очистить карту", icon=":material/cancel:", width=40, key="clear_map_button",  # Кнопка очистки.
                  disabled=area is None, on_click=clear_map, help="Очистить карту: убрать границу, результаты и файлы.")
    way = st.segmented_control("Способ", ["Город", "Полигон"], default="Город",  # Способ задать область.
                               label_visibility="collapsed") or "Город"
    if way == "Город":  # Область – граница города.
        city_input = state["city_input"]  # Город, введённый ранее.
        options = [*cities, city_input] if city_input and city_input not in cities else cities  # Свой город – в конец списка.
        city = st.selectbox("Город", options, index=options.index(city_input) if city_input else None,  # Выбор города.
                            accept_new_options=True, key="city_choice", placeholder="Город из списка или своё название",
                            label_visibility="collapsed",
                            help="Свой город введите названием. Граница и объекты загрузятся из OpenStreetMap.")
        state["city_input"] = city  # Запомнить город.
        if city and (not area or area["method"] != way or area["name"] != city):  # Выбран новый город.
            try:
                geometry = get("/place", city=city)["geometry"]  # Граница города от серверной части.
            except (requests.RequestException, RuntimeError) as error:  # Сервер недоступен или города нет.
                st.error(str(error))  # Причина под списком городов.
            else:
                select_area({"geometry": geometry, "name": city, "method": way})  # Город – новая область.
                st.rerun()  # Показать границу на карте.
        ready = bool(city and area and area["method"] == way and area["name"] == city)  # Граница города уже на карте.
    else:
        st.caption("Полигон, квадрат и круг рисуются на карте.")  # Подсказка о рисовании.
        imported = upload_area(state.get("polygon_upload_revision", 0))  # Границы из файла.
        if imported and imported != area:  # Применены новые границы.
            select_area(imported)  # Границы – новая область.
            st.rerun()  # Показать границы на карте.
        ready = bool(area and area["method"] == way)  # Полигон уже на карте.
    if area and (way == "Полигон" or not ready):  # Подпись выбранной области.
        label_area = "Выбрано" if ready else "На карте"  # Готова ли область к расчёту.
        st.caption(f"{label_area}: **{area['name']}**",  # Название области.
                   help="В расчёт входят целые ячейки H3, пересекающие границу." if area["method"] == "Полигон" else None)
    mode = st.segmented_control("Модель", ["Своя модель", "Обучение на лету"], default="Своя модель") or "Своя модель"  # Режим.
    if mode == "Своя модель":  # Готовая модель серверной части.
        target, extra, label = info["target"], info["extra_env"], info["topic"]  # Теги, факторы и тема модели.
        st.caption(label, help=f"{info['model']}. Теги OpenStreetMap: {tags_text(target)}. "  # Тема и сведения о модели.
                               f"Города обучения: {', '.join(info['train_cities'])}.")
    else:
        objects = {o["name"]: o for o in PRESETS["objects"]}  # Готовые виды объектов.
        factors = {f["name"]: f["tags"] for f in PRESETS["factors"]} | state.setdefault("own_factors", {})  # Все факторы.
        names = list(objects)  # Названия видов объектов.
        # Вид объектов.
        choice = st.selectbox("Объекты", names + [OWN], index=names.index(info["topic"]) if info["topic"] in objects else 0,
                              help="Выберите категорию объектов или задайте свои теги OpenStreetMap.")
        if choice == OWN:  # Свои теги.
            a, b = st.columns(2)  # Название и теги рядом.
            label = a.text_input("Название", "Мои объекты")  # Название своих объектов.
            target = parse_tags(b.text_input("Теги OpenStreetMap", placeholder="shop=pet",  # Свои теги.
                                             help="Условия через «;», значения через «|», «*» – любое значение"))
        else:
            label, target = choice, objects[choice]["target"]  # Название и теги готового вида.
            st.caption(f"Теги: {tags_text(target)}")  # Показать теги.
        if state.get("factors_for") != choice:  # Новый вид объектов – факторы его темы.
            state["factors_for"], state["picked_factors"] = choice, list(objects.get(choice, {}).get("factors", []))  # Сменить выбор.
        if "pending_factors" in state:  # Выбор после добавления или удаления своего фактора.
            state["picked_factors"] = state.pop("pending_factors")  # Новый выбор факторов.
        picked = st.multiselect("Дополнительные факторы окружения", list(factors), key="picked_factors",  # Выбор факторов.
                                placeholder="Выберите факторы",
                                help="12 базовых категорий окружения, улицы и население учитываются всегда")
        with st.expander("Добавить свой фактор"):  # Свои факторы.
            with st.form("own_factor", border=False):  # Поля отправляются вместе.
                a, b = st.columns(2)  # Название и тег рядом.
                name = a.text_input("Название фактора", placeholder="Ветеринарные клиники")  # Название фактора.
                tags = parse_tags(b.text_input("Тег OpenStreetMap", placeholder="amenity=veterinary"))  # Тег фактора.
                if st.form_submit_button("Добавить", width="stretch") and name and tags:  # Фактор задан полностью.
                    state["own_factors"][name] = tags[0]  # Сохранить свой фактор.
                    state["pending_factors"] = picked + [name]  # И сразу выбрать его.
                    st.rerun()  # Обновить список факторов.
            for name in list(state["own_factors"]):  # Свои факторы с кнопками удаления.
                a, b = st.columns([5, 1], vertical_alignment="center")  # Подпись и кнопка рядом.
                a.caption(f"{name}: {tags_text([state['own_factors'][name]])}")  # Название и тег фактора.
                if b.button("✕", key=f"del_{name}", help="Удалить фактор"):  # Удаление фактора.
                    del state["own_factors"][name]  # Убрать фактор.
                    state["pending_factors"] = [p for p in picked if p != name]  # И из выбора.
                    st.rerun()  # Обновить список факторов.
        extra = [factors[name] for name in picked if name in factors]  # Теги выбранных факторов.
    if st.button("Рассчитать", type="primary", width="stretch", disabled=not (target and ready)):  # Запуск расчёта.
        name = area["name"]  # Название области.
        body = {"city": name}  # Тело запроса.
        if way == "Полигон":  # Для полигона – его геометрия.
            body["geometry"] = area["geometry"]  # Граница области.
        try:
            with st.spinner(f"Идёт расчёт для «{name}»…"):  # Индикатор расчёта.
                if mode == "Своя модель":  # Готовая модель.
                    state["result"] = get("/analyze", city=name) if way == "Город" else post("/analyze", body)  # Расчёт.
                else:
                    state["result"] = post("/train", body | {"target": target, "extra_env": extra})  # Обучение и расчёт.
            state["params"] = {"city": name, "method": way, "mode": mode, "target": target, "extra_env": extra,  # Для отчёта.
                               "label": label, "title": f"GeoMarketing Analysis: {name}"}
            st.rerun()  # Показать результат.
        except (requests.RequestException, RuntimeError) as error:  # Сервер недоступен или вернул ошибку.
            st.error(str(error))  # Причина под кнопкой «Рассчитать».

result, params = state.get("result"), state.get("params", {})  # Результат и его параметры.
shown = result is not None and area is not None  # Есть что показать.
map_height(320 if shown else 225)  # Высота карты под окно.
boundary = None  # Граница области.
if area:  # Область выбрана.
    boundary = gpd.GeoDataFrame.from_features(  # Граница слоем.
        [{"type": "Feature", "properties": {}, "geometry": area["geometry"]}], crs="EPSG:4326")
    state["map_bounds"] = tuple(boundary.total_bounds)  # Охват области.
view = state.get("map_bounds")  # Охват карты.

with left:  # Левая колонка: показатели и карта.
    if shown:  # Есть результат.
        s = result["summary"]  # Сводка расчёта.
        above = once("potential_count", lambda: sum(  # Ячейки с потенциалом больше 0,5.
            (cell["properties"].get("потенциал") or 0) > 0.5 for cell in result["cells"]["features"]
        ))
        cols = st.columns(5, gap="xsmall")  # Пять показателей в ряд.
        cols[0].metric("Объектов в городе" if area["method"] == "Город" else "Объектов в области",  # Объекты темы.
                       fmt(s["объектов темы"], 0),
                       help=f"{s['город']}: {params['label'].lower()} по OpenStreetMap")
        cols[1].metric("Прогноз модели", fmt(s["ожидаемое число"]), help="Сумма прогноза модели по ячейкам области")  # Прогноз.
        cols[2].metric("Ячеек H3", fmt(s["ячеек"], 0), help="Шестиугольники разрешения 9, около 0,1 км² каждый")  # Ячейки.
        cols[3].metric("Вне диапазона", f"{fmt(s['ячеек вне диапазона обучения, %'])} %",  # Доля вне диапазона.
                       help="Доля ячеек, признаки которых выходят за диапазон городов обучения модели")
        cols[4].metric("Потенциал > 0,5", fmt(above, 0), help="Число ячеек с потенциалом строго больше 0,5")  # Перспективные.
        m = full_map(result, boundary, params["label"])  # Каждый раз заново: st_folium изменяет объект карты.
    else:
        m = base_map(view)  # Карта с подложками Esri.
        if boundary is not None:  # Область выбрана.
            add_boundary(m, boundary)  # Граница области.
        finish(m)  # Кнопки управления картой.
    drawing_tools(m)  # Инструменты рисования.
    remember_view(m, view)  # Положение карты при смене слоёв.
    drawn = st_folium(m, height=640, use_container_width=True, returned_objects=["last_active_drawing"],  # Карта на странице.
                      key="geomarketing_map", feature_group_to_add=drawing_mode(way == "Полигон"))
    feature = (drawn or {}).get("last_active_drawing") or {}  # Последняя нарисованная фигура.
    draw_id = feature.get("properties", {}).get("area_draw_id")  # Её номер.
    if draw_id and draw_id != state.get("last_area_drawing"):  # Нарисована новая фигура.
        state["last_area_drawing"] = draw_id  # Запомнить номер.
        if way == "Полигон":  # Рисование доступно для полигона.
            try:
                select_area(drawn_area(feature))  # Фигура – новая область.
                st.rerun()  # Показать её на карте.
            except ValueError as error:  # Некорректная фигура.
                st.error(str(error))  # Например, самопересечение контура.
    symbols = "Номерные метки — места-кандидаты, белые точки — существующие объекты. " if shown else ""  # Условные знаки.
    source = "Объекты" if area and area["method"] == "Полигон" else "Объекты и границы"  # Что взято из OpenStreetMap.
    st.markdown(f'<div class="source">{symbols}{source} — © OpenStreetMap (ODbL), население — '  # Источники данных.
                "GHS-POP (CC BY 4.0), подложка — Esri.</div>",
                unsafe_allow_html=True)

if not shown:  # Результата нет.
    st.stop()  # Дальше показывать нечего.

ext = result["extension"]  # Результат собственной функции.
name = area["name"]  # Название области для файлов.
cells = once("cells", lambda: gpd.GeoDataFrame.from_features(result["cells"], crs="EPSG:4326"))  # Ячейки – раз на расчёт.
candidates = candidate_layer(result)  # Места-кандидаты.
with right, st.container(border=True, gap="xsmall"):  # Блок файлов.
    st.subheader("Файлы")  # Заголовок блока.

    def geopackage():  # Файл GeoPackage из трёх слоёв.
        with tempfile.TemporaryDirectory() as tmp:  # Временная папка.
            path = os.path.join(tmp, "result.gpkg")  # Временный файл.
            cells.to_file(path, layer="cells")  # Слой ячеек.
            candidates.to_file(path, layer="candidates")  # Слой мест-кандидатов.
            result_layer(result["targets"]).to_file(path, layer="targets")  # Слой существующих объектов.
            return Path(path).read_bytes()  # Содержимое файла.

    # Отчёт HTML строится один раз на расчёт.
    st.download_button("Отчёт HTML", once("report", lambda: build_report(result, boundary, params)), f"{name}_отчёт.html",
                       width="stretch", help="Карта со слоями, места-кандидаты и сведения о модели")
    c1, c2, c3 = st.columns(3, gap="xsmall")  # Три кнопки в ряд.
    # Отдельная карта для файла: показанную карту st_folium уже изменил.
    c1.download_button("Карта", once("map_html", lambda: full_map(result, boundary, params["label"]).get_root().render()),
                       f"{name}_карта.html", width="stretch",
                       help="Интерактивная карта HTML")
    c2.download_button("GPKG", once("gpkg", geopackage), f"{name}.gpkg", width="stretch",  # Файл GeoPackage.
                       help="GeoPackage: ячейки, кандидаты, объекты")
    c3.download_button("GeoJSON", once("geojson", cells.to_json), f"{name}.geojson", width="stretch",  # Ячейки.
                       help="Ячейки с прогнозом и потенциалом")
if s["ячеек вне диапазона обучения, %"] > 20:  # Много ячеек вне диапазона обучения.
    right.warning("Больше пятой части ячеек выходит за диапазон городов обучения: прогноз – экстраполяция.")  # Предупреждение.

with left, st.container(border=True, gap="xsmall"):  # Блок мест-кандидатов.
    st.subheader("Места-кандидаты")  # Заголовок блока.
    st.caption("Потенциал — разность между прогнозом модели и числом существующих объектов в ячейке.")  # Пояснение.
    table = pd.DataFrame({"Адрес": candidates.get("адрес", ""), "Потенциал": candidates["потенциал"],  # Таблица кандидатов.
                          "Ожидается": candidates["ожидаемое"], "Есть сейчас": candidates["объектов темы"].astype(int),
                          "Широта": candidates.geometry.centroid.y, "Долгота": candidates.geometry.centroid.x
                          }).set_axis(pd.RangeIndex(1, len(candidates) + 1, name="№"))
    st.dataframe(table, width="stretch",  # Таблица с форматом чисел и шкалой потенциала.
                 column_config={"Адрес": st.column_config.TextColumn(width="large"),
                                "Широта": st.column_config.NumberColumn(format="%.5f"),
                                "Долгота": st.column_config.NumberColumn(format="%.5f"),
                                "Ожидается": st.column_config.NumberColumn(format="%.2f"),
                                "Потенциал": st.column_config.ProgressColumn(format="+%.2f", min_value=0,
                                                                          max_value=float(max(0.01, candidates["потенциал"].max())))})

with right, st.container(border=True, gap="xsmall"):  # Блок собственной функции и модели.
    if ext.get("name") != "Пример: ячейки с потенциалом больше 0,5" or ext.get("table"):  # Своя функция или её таблица.
        st.subheader(ext.get("name") or "Собственная функция")  # Название функции.
        if ext.get("text"):  # Есть пояснение.
            st.caption(ext["text"])  # Краткий вывод функции.
        if ext.get("table"):  # Есть таблица.
            st.dataframe(pd.DataFrame(ext["table"]), width="stretch", hide_index=True, row_height=28)  # Таблица функции.
    st.subheader("Модель расчёта")  # Заголовок блока.
    st.caption(f"{s['режим'].capitalize()}: {s['модель'].lower()}, обучена: {s['города обучения']}.")  # Режим и модель.
    metrics = pd.DataFrame(s.get("metrics") or {s["модель"]: info.get("metrics", {})})  # Точность моделей.
    st.dataframe(metrics, width="stretch", row_height=28,  # Таблица точности.
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in metrics})
