# Справочник городов и данные OpenStreetMap: граница города, объекты по категориям (готовые файлы, кэш или Overpass API)
# и признаки окружения ячеек сетки H3 – число объектов, длины улиц, расстояния до объектов, число объектов темы.
import hashlib, json, os, threading, urllib.error, urllib.request
from tempfile import TemporaryDirectory
import numpy as np, pandas as pd, geopandas as gpd
import h3, osmnx as ox, shapely
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, shape
from server.overpass_mirrors import select_overpass_mirror


# Справочник 41 города векторного варианта (Пермь не используется: GHS-POP занижает её население).
CITIES = {  # Город: (объект OSM с границей, население по официальным данным).
    "Москва": ("R102269", 13274285),
    "Санкт-Петербург": ("R337422", 5652922),
    "Новосибирск": ("R1751445", 1637266),
    "Екатеринбург": ("R6564910", 1468833),
    "Казань": ("R3437391", 1329825),
    "Красноярск": ("R1430616", 1092851),
    "Нижний Новгород": ("R1752948", 1198245),
    "Челябинск": ("R4442556", 1176770),
    "Уфа": ("R1549169", 1144809),
    "Самара": ("R3368701", 1154223),
    "Ростов-на-Дону": ("R1285772", 1143123),
    "Краснодар": ("R7373058", 1154885),
    "Омск": ("R3442814", 1101367),
    "Воронеж": ("R3536492", 1041722),
    "Волгоград": ("R3374767", 1012219),
    "Саратов": ("R3955288", 838042),
    "Тюмень": ("R2049854", 872077),
    "Тольятти": ("R3338286", 662683),
    "Махачкала": ("R4632505", 662660),
    "Барнаул": ("R1550512", 621598),
    "Ижевск": ("R1670935", 616297),
    "Хабаровск": ("R2049848", 615600),
    "Ульяновск": ("R2049867", 612516),
    "Иркутск": ("R1430614", 605708),
    "Владивосток": ("R1704857", 591628),
    "Ярославль": ("R1701435", 563717),
    "Ставрополь": ("R6712557", 450680),
    "Томск": ("R7261645", 544566),
    "Кемерово": ("R1312868", 542928),
    "Набережные Челны": ("R3437417", 548221),
    "Оренбург": ("W166155507", 536515),
    "Новокузнецк": ("R1437127", 528747),
    "Балашиха": ("R3171497", 544303),
    "Рязань": ("R1782722", 519315),
    "Чебоксары": ("R3374548", 497061),
    "Пенза": ("R2552696", 487978),
    "Калининград": ("R1674442", 488690),
    "Липецк": ("R3134925", 483040),
    "Киров": ("R2383150", 475871),
    "Астрахань": ("R1853866", 465990),
    "Тула": ("R2538203", 456813),
}


OSM_DATA = "https://raw.githubusercontent.com/tester170/geoservice-internship-data/main/osm"  # Готовые данные 41 города.
CACHE = os.environ.get("OSM_CACHE", "osm_data")  # Папка файлов объектов OpenStreetMap.
MAX_AREA_KM2 = float(os.environ.get("MAX_AREA_KM2", 1500))  # Предел площади города не из справочника, км².
DESCRIPTION = ["name", "brand", "opening_hours", "addr:street", "addr:housenumber"]  # Теги для подсказок на карте.
ox.settings.requests_timeout = 600  # Крупный город загружается долго.
MIRROR = {"lock": threading.Lock(), "ready": False}  # Зеркало Overpass выбирается один раз за запуск.

BASE_ENV = {  # Единая система классификации окружения.
    "общепит": {"amenity": ["cafe", "restaurant", "fast_food"]},
    "магазины": {"shop": True},
    "офисы": {"office": True},
    "образование": {"amenity": ["school", "kindergarten", "university", "college"]},
    "медицина": {"amenity": ["clinic", "hospital", "doctors"]},
    "остановки": {"highway": ["bus_stop"], "railway": ["tram_stop", "station", "halt", "subway_entrance"]},
    "парки": {"leisure": ["park"]},
    "жилые здания": {"building": ["apartments", "house", "residential"]},
    "торговые и офисные здания": {"building": ["retail", "commercial", "office"]},
    "парковки": {"amenity": ["parking"]},
    "магистральные улицы": {"highway": ["trunk", "primary", "secondary"]},
    "местные улицы": {"highway": ["tertiary", "residential"]},
}


def overpass():
    """Сервер Overpass API при первой загрузке из OpenStreetMap: OVERPASS_URL или самое быстрое зеркало."""
    with MIRROR["lock"]:  # Одновременные запросы ждут одного выбора.
        if not MIRROR["ready"]:  # Зеркало ещё не выбрано.
            if os.environ.get("OVERPASS_URL"):  # Адрес задан явно.
                ox.settings.overpass_url = os.environ["OVERPASS_URL"]  # Все запросы – на этот сервер.
            else:
                select_overpass_mirror(wait_seconds=60, requests_timeout=600)  # Остальные зеркала – запасные.
            MIRROR["ready"] = True  # Больше не выбираем.


def download(name, path):
    """Подготовленный файл объектов с GitHub; False – файла нет или GitHub недоступен (тогда данные из OpenStreetMap)."""
    try:
        urllib.request.urlretrieve(f"{OSM_DATA}/{name}", path + ".part")  # Скачивание во временный файл.
        os.replace(path + ".part", path)  # Недокачанный файл не попадает в кэш.
        return True  # Файл в кэше.
    except (urllib.error.URLError, OSError):  # Нет файла или нет связи.
        return False  # Данные возьмём из OpenStreetMap.


def env_categories(extra_env):
    """Базовые категории и дополнительные факторы темы – в том же порядке и с теми же именами, что при обучении модели."""
    env = dict(BASE_ENV)  # Копия: справочник не меняется.
    for i, tags in enumerate(extra_env, 1):  # Имя фактора – его теги, например «фактор 1: shop=tyres».
        env[f"фактор {i}: " + ", ".join(f"{k}={'*' if v is True else '|'.join(v)}" for k, v in tags.items())] = tags  # Новая категория.
    return env  # Категория: теги.


def place(city):
    """Номер объекта OSM и граница города: из справочника – по номеру, любой другой город – поиском по названию."""
    if city in CITIES:  # Город справочника – по номеру объекта.
        return CITIES[city][0], ox.geocode_to_gdf(CITIES[city][0], by_osmid=True).geometry.iloc[0]  # Граница по номеру.
    try:
        found = ox.geocode_to_gdf(city)  # Поиск по названию в Nominatim.
    except Exception:  # Ничего не найдено.
        # Сообщение с подсказкой, как уточнить.
        raise ValueError(f"Город «{city}» не найден в OpenStreetMap. Уточните название, например «Томск, Россия».")
    polygon = found.geometry.iloc[0]  # Первый результат поиска.
    if polygon.geom_type not in ("Polygon", "MultiPolygon"):  # Найдена точка или линия, а не граница.
        raise ValueError(f"У объекта «{city}» в OpenStreetMap нет границы. Уточните название города.")  # Без границы ячейки не построить.
    boundary = gpd.GeoSeries([polygon], crs="EPSG:4326")  # Граница в WGS 84.
    area = boundary.to_crs(boundary.estimate_utm_crs()).area.iloc[0] / 1e6  # Площадь в местной проекции UTM, км².
    if area > MAX_AREA_KM2:  # Скорее всего, найден регион.
        # Ошибка вместо долгой загрузки региона.
        raise ValueError(f"«{city}»: площадь {area:.0f} км² больше предела {MAX_AREA_KM2:.0f} км². "
                         "Возможно, найден регион, а не город: уточните название.")
    return found["osm_type"].iloc[0][0].upper() + str(found["osm_id"].iloc[0]), polygon  # Номер вида R1430616.


def cache_path(key, tags):
    """Файл объектов города по набору тегов – так же названы подготовленные файлы объектов."""
    os.makedirs(CACHE, exist_ok=True)  # Папка кэша создаётся при первом обращении.
    digest = hashlib.md5(json.dumps(tags, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]  # Отпечаток тегов.
    return os.path.join(CACHE, f"{key}_{digest}.parquet")  # Например, R1430616_0eca01c998.parquet.


def has_tags(gdf, tag, values):  # Отбор объектов по одному тегу.
    return gdf[tag].notna() if values is True else gdf[tag].isin(values)  # Тег есть (True) или одно из значений.


def kept_columns(gdf, tags):  # Столбцы для файла кэша.
    return list(dict.fromkeys(["geometry"] + [k for k in list(tags) + DESCRIPTION if k in gdf]))  # Столбцы файла кэша без повторов.


def no_objects(columns=()):  # Таблица, когда объектов нет.
    return gpd.GeoDataFrame({k: [] for k in columns}, geometry=[], crs="EPSG:4326",  # Индекс как у таблиц OSMnx: тип и номер объекта.
                            index=pd.MultiIndex.from_arrays([[], []], names=["element", "id"]))


def prefetch(key, polygon, tag_sets, *, prepared=True):
    """Город не из подготовленных: все теги – одним запросом Overpass API, затем раскладка по наборам тегов в кэш."""
    paths = {cache_path(key, tags): tags for tags in tag_sets}  # Одинаковые наборы – один файл.
    missing = {path: tags for path, tags in paths.items() if not os.path.exists(path)  # Нет в кэше и среди подготовленных.
               and not (prepared and download(os.path.basename(path), path))}
    if not missing:  # Все файлы уже есть.
        return  # Запросов к OSM не нужно.
    merged = {}  # Объединение тегов: True поглощает список значений.
    for tags in missing.values():  # Наборы тегов недостающих файлов.
        for k, v in tags.items():  # Значения одного тега складываются.
            merged[k] = True if v is True or merged.get(k) is True else sorted(set(merged.get(k, [])) | set(v))  # Объединение значений.
    overpass()  # Выбор зеркала при первой загрузке.
    # OSMnx заменяет MultiPolygon выпуклой оболочкой. Для своей области загружаем только её части.
    parts = polygon.geoms if not prepared and polygon.geom_type == "MultiPolygon" else [polygon]
    frames = []  # Объекты каждой части.
    for part in parts:  # Один запрос на связную часть.
        try:
            frames.append(ox.features_from_polygon(part, merged))  # Все теги одним запросом.
        except ox._errors.InsufficientResponseError:  # В этой части объектов нет.
            continue  # Следующая часть.
    if frames:  # Хотя бы в одной части есть объекты.
        gdf = pd.concat(frames)  # Объекты всех частей.
        gdf = gdf[~gdf.index.duplicated()]  # Объект на стыке частей – один раз.
    else:
        gdf = no_objects()  # Объектов нет нигде.
    with TemporaryDirectory(dir=CACHE) as tmp:  # Временная папка рядом с кэшем.
        building = os.path.join(tmp, "objects.parquet")  # Файл пишется рядом и затем заменяет кэш целиком.
        for path, tags in missing.items():  # Раскладка по наборам тегов.
            match = pd.Series(False, index=gdf.index)  # Объект подходит хотя бы по одному тегу набора.
            for k, v in tags.items():  # Теги набора.
                if k in gdf:  # Тега нет ни у одного объекта – пропускаем.
                    match |= has_tags(gdf, k, v)  # Добавляем подходящие объекты.
            gdf.loc[match, kept_columns(gdf, tags)].reset_index().to_parquet(building)  # Только нужные столбцы.
            os.replace(building, path)  # Готовый файл заменяет кэш одним действием.


def load_osm(key, polygon, tags):
    """Объекты города по тегам: подготовленный файл с GitHub, сохранённая ранее загрузка или Overpass API."""
    path = cache_path(key, tags)  # Файл кэша этого набора тегов.
    if not os.path.exists(path) and not download(os.path.basename(path), path):  # Нет среди подготовленных – из OSM.
        overpass()  # Выбор зеркала при первой загрузке.
        try:
            gdf = ox.features_from_polygon(polygon, tags)  # Запрос Overpass API.
        except ox._errors.InsufficientResponseError:  # Объектов с такими тегами в городе нет.
            gdf = no_objects(tags)  # Пустая таблица со столбцами тегов.
        gdf[kept_columns(gdf, tags)].reset_index().to_parquet(path)  # Сохраняем для следующих запросов.
    return gpd.read_parquet(path).set_index(["element", "id"])  # Индекс – тип и номер объекта OSM.


def load_target(key, polygon, target):
    """Объекты темы: хотя бы один вариант тегов, внутри варианта – все условия."""
    parts = []  # Объекты каждого варианта.
    for variant in target:  # Варианты объединяются.
        gdf = load_osm(key, polygon, variant)  # Объекты хотя бы с одним тегом варианта.
        for tag, values in variant.items():  # Каждое условие варианта сужает выборку.
            gdf = gdf.iloc[:0] if tag not in gdf else gdf[has_tags(gdf, tag, values)]  # Нет тега – нет объектов.
        parts.append(gdf)  # Объекты варианта.
    target_gdf = pd.concat(parts)  # Все варианты вместе.
    return target_gdf[~target_gdf.index.duplicated()]  # Объект из нескольких вариантов – один раз.


def city_cells(city, card, population):
    """Ячейки H3 города с признаками из карточки модели и фактическим числом объектов темы."""
    key, polygon = place(city)  # Номер объекта OSM и граница.
    return area_cells(key, polygon, card, population)  # Признаки по всей границе города.


def checked_area(geometry):
    """Проверить пользовательскую границу до запросов к OpenStreetMap и Earth Engine."""
    try:
        polygon = shape(geometry)  # GeoJSON в геометрию Shapely.
    except (KeyError, TypeError, ValueError, ShapelyError) as error:  # Неверная запись GeoJSON.
        raise ValueError("Не удалось прочитать границы области.") from error  # Понятная ошибка вместо сообщения Shapely.
    coords = shapely.get_coordinates(polygon)  # Все вершины одним массивом.
    if (polygon.geom_type not in ("Polygon", "MultiPolygon") or polygon.is_empty  # Не полигон, пустой или самопересечение.
            or not np.isfinite(coords).all() or not polygon.is_valid):
        raise ValueError("Укажите корректный полигон или мультиполигон.")  # Сообщение для клиентской части.
    if np.any(np.abs(coords) > [180, 90]):  # Долгота до 180°, широта до 90°.
        raise ValueError("Координаты области должны быть в WGS 84.")  # Похоже на метры вместо градусов.
    return polygon.normalize()  # Одна и та же область – одна и та же запись.


def polygon_cells(geometry, card, population, *, folds=0):
    """Ячейки выбранных областей. Длины и расстояния вычисляются в местных проекциях."""
    polygon = checked_area(geometry)  # Проверка до сетевых запросов.
    zones = {}  # Части области по зонам UTM (номер, северное полушарие).
    for part in polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]:  # Связные части области.
        center = part.representative_point()  # Точка внутри части.
        zone = min(59, int((center.x + 180) // 6)), center.y >= 0  # Зона UTM этой точки.
        zones.setdefault(zone, []).append(part)  # Части одной зоны – вместе.
    jobs, assigned = [], set()  # Ячейка на границе зон считается один раз.
    for parts in zones.values():  # Зоны по порядку появления.
        region = parts[0] if len(parts) == 1 else MultiPolygon(parts)  # Все части зоны.
        selected = set(h3.h3shape_to_cells_experimental(  # Все ячейки, пересекающие контур.
            h3.geo_to_h3shape(region), card["h3_resolution"], contain="overlap"))
        cells = sorted(selected - assigned)  # Без ячеек предыдущих зон.
        if cells:  # В зоне остались ячейки.
            assigned.update(cells)  # Ячейки зоны заняты.
            jobs.append(cells)  # Отдельный расчёт для зоны.
    if not jobs:  # Область не покрыта ни одной ячейкой.
        raise ValueError("Не удалось покрыть область ячейками H3. Проверьте её границу.")  # Ошибка вместо пустого результата.
    if folds and len({h3.cell_to_parent(c, 7) for c in assigned}) < folds:  # Блоки проверки – ячейки разрешения 7.
        raise ValueError(f"Для обучения нужно не меньше {folds} пространственных блоков H3. "  # Мало блоков для проверки по частям.
                         "Увеличьте область или выберите режим «Своя модель».")
    tables, objects = [], []  # Признаки и объекты темы по зонам.
    for cells in jobs:  # Расчёт каждой зоны в своей проекции.
        # Берём целые ячейки на краю, а для признаков окрестности добавляем ближайший ряд.
        context = {neighbor for cell in cells for neighbor in h3.grid_disk(cell, 1)}
        footprint = shape(h3.h3shape_to_geo(h3.cells_to_h3shape(sorted(context)))).normalize()  # Граница с соседним рядом.
        key = "area_" + hashlib.sha256(footprint.wkb).hexdigest()[:24]  # Имя файлов кэша – отпечаток границы.
        table, targets = area_cells(key, footprint, card, None, cells)  # Население – позже, одним запросом.
        tables.append(table)  # Признаки зоны.
        objects.append(targets)  # Объекты темы зоны.
    table = pd.concat(tables)  # Все зоны вместе.
    for name in table.columns:  # Каждый признак таблицы.
        if name != "geometry" and table[name].hasnans:  # Признак есть не во всех зонах.
            table[name] = table[name].fillna(table[name].max() if name.startswith("до:") else 0)  # Расстояние – наибольшее, остальное – 0.
    table["население"] = population(table)  # Один запрос населения на всю область.
    targets = pd.concat(objects)  # Объекты темы всех зон.
    return table, targets[~targets.index.duplicated()]  # Объект на стыке зон – один раз.


def hexagons(cells):
    """Шестиугольники ячеек H3: все вершины одним массивом – быстрее, чем по одному многоугольнику."""
    rings = [h3.cell_to_boundary(c) for c in cells]  # Вершины: (широта, долгота).
    points = np.array([p for ring in rings for p in ring], dtype=float).reshape(-1, 2)[:, ::-1]  # Подряд, как (долгота, широта).
    ring_of = np.repeat(np.arange(len(rings)), [len(ring) for ring in rings])  # Номер ячейки каждой вершины.
    return shapely.polygons(shapely.linearrings(points, indices=ring_of))  # Кольца замыкаются сами.


def cells_of(geometry, res):
    """Ячейка H3 каждого объекта – по точке внутри него (у зданий и площадей – внутри контура)."""
    inside = geometry.representative_point().values  # Точка внутри каждого объекта.
    return [h3.latlng_to_cell(y, x, res) for x, y in zip(shapely.get_x(inside), shapely.get_y(inside))]  # Широта, долгота.


def area_cells(key, polygon, card, population, cells=None):
    """Признаки и цели по границе города либо по выбранным ячейкам пользовательской области."""
    env = env_categories(card["extra_env"])  # Категории окружения модели.
    res = card["h3_resolution"]  # Разрешение сетки H3.
    utm = gpd.GeoSeries([polygon], crs="EPSG:4326").estimate_utm_crs()  # Местная проекция для длин и расстояний.
    # Расстояния считаются до дополнительных факторов и до магистральных улиц.
    near_tags = {f"фактор {i}": t for i, t in enumerate(card["extra_env"], 1)} | {"магистральные улицы": BASE_ENV["магистральные улицы"]}
    custom = cells is not None  # Своя область: ячейки уже выбраны.
    # Недостающие файлы – одним запросом Overpass для всех наборов тегов.
    prefetch(key, polygon, list(card["target"]) + list(env.values()) + list(near_tags.values()), prepared=not custom)
    targets = load_target(key, polygon, card["target"])  # Объекты темы.
    loaded = {cache_path(key, tags): load_osm(key, polygon, tags) for tags in env.values()}  # Каждый файл читается один раз.
    env_objects = []  # Объекты окружения всех категорий.
    for category, tags in env.items():  # Объекты темы не считаются окружением.
        gdf = loaded[cache_path(key, tags)]  # Объекты категории.
        # Только геометрия и имя категории.
        env_objects.append(gdf[~gdf.index.isin(targets.index)].reset_index()[["geometry"]].assign(категория=category))
    env_objects = gpd.GeoDataFrame(pd.concat(env_objects), crs="EPSG:4326")  # Одна таблица с категорией.
    is_street = env_objects["категория"].str.endswith("улицы")  # Улицы учитываются длиной.
    lines = env_objects[is_street]  # Линии улиц.
    points = env_objects[~is_street].copy()  # Остальное – числом объектов.
    points["geometry"] = points.geometry.representative_point()  # Здание или площадь – точкой внутри.

    if not custom:  # Город: ячейки внутри границы и все ячейки, которые она пересекает.
        cells = list(h3.h3shape_to_cells_experimental(h3.geo_to_h3shape(polygon), res, contain="overlap"))  # Номера ячеек H3.
    grid = gpd.GeoDataFrame({"cell": cells}, crs="EPSG:4326", geometry=hexagons(cells)).set_index("cell")  # Сетка.
    counts = points.groupby([cells_of(points.geometry, res), points["категория"]]).size().unstack(fill_value=0)  # Ячейка × категория.
    selected_counts = counts.reindex(grid.index, fill_value=0)  # Только ячейки сетки, пустые – нули.
    # Окрестность – ячейка и шесть соседей; у своей области соседний ряд вне неё тоже учитывается.
    near = pd.DataFrame([(c, n) for c in grid.index for n in h3.grid_disk(c, 1)], columns=["cell", "n"]) \
        .join(counts if custom else selected_counts, on="n").groupby("cell").sum(numeric_only=True).add_suffix(" (окрестность)")
    streets = gpd.overlay(lines.to_crs(utm), grid.reset_index().to_crs(utm), keep_geom_type=True)  # Улицы по ячейкам.
    # Длина улиц каждой категории в ячейке, км.
    length = streets.assign(км=streets.length / 1000).pivot_table(index="cell", columns="категория", values="км", aggfunc="sum")
    result = grid.join(selected_counts).join(near).join(length.add_suffix(", км")).fillna(0)  # Все признаки, пропуски – 0.
    if population is not None:  # Город: население сразу.
        result["население"] = population(result)  # Запрос к Earth Engine.
    centers = result.geometry.to_crs(utm).centroid.to_frame("geometry")  # Центры ячеек в метрах.
    for name, tags in near_tags.items():  # Расстояние от центра ячейки до ближайшего объекта.
        objects = loaded[cache_path(key, tags)].reset_index()[["geometry"]].to_crs(utm)  # Все объекты, и объекты темы тоже.
        if len(objects):  # Объектов нет – признак добавится ниже.
            nearest = gpd.sjoin_nearest(centers, objects, distance_col="м")["м"].groupby(level=0).min()  # Равноудалённые – одно.
            result[f"до: {name}, км"] = (nearest / 1000).round(3)  # Километры.
    target_cells = pd.Series(cells_of(targets.geometry, res), index=targets.index, dtype="object")  # Ячейка объекта темы.
    if custom:  # Объекты темы соседнего ряда не входят в счётчик.
        inside = target_cells.isin(grid.index)  # Объект внутри выбранных ячеек.
        targets, target_cells = targets[inside], target_cells[inside]  # Только внутренние.
    result["объектов темы"] = target_cells.value_counts().reindex(result.index, fill_value=0)  # Фактическое число.
    for feature in card["features"]:  # Признаки модели по порядку.
        # Признака нет в городе: расстояние – наибольшее по городам обучения (как при обучении модели), остальное – 0.
        if feature not in result:
            result[feature] = card.get("feature_max", {}).get(feature, 0.0) if feature.startswith("до:") else 0.0  # Значение по умолчанию.
    return result, targets  # Ячейки с признаками и объекты темы.
