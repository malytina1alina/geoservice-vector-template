# Карты и HTML-отчёт: подложки Esri, кнопки «Подложка» и «Слои», легенда в углу карты, инструменты рисования,
# слои расчёта – граница, тепловая карта потенциала, потенциал и ожидаемое число по ячейкам, объекты темы,
# места-кандидаты; отчёт – параметры расчёта и слои карты по очереди, каждый с описанием.
import datetime
import html
import json

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement
from folium.plugins import Draw, HeatMap
from jinja2 import Template
from client.interface import fmt


ACCENT = "#087f73"  # Цвет нарисованных фигур.
# Потенциал: красный – объектов больше ожидаемого (открывать не нужно), зелёный – меньше ожидаемого (стоит открывать).
GAP_COLORS = ["#a50026", "#e8604a", "#fdbf9a", "#f7f7f7", "#c2e5b3", "#5fb85f", "#00682f"]
GREEN = ["#a6dba0", "#5aae61", "#1b7837", "#00441b"]  # Места-кандидаты: ниже → выше потенциал.
EXT_COLORS = ["#f2f0f7", "#bcbddc", "#807dba", "#54278f", "#2d004b"]  # Слой собственной функции: меньше → больше.
HEAT_OPEN = {0.2: "#d9f0d3", 0.45: "#a6dba0", 0.65: "#5aae61", 0.85: "#1b7837", 1: "#00441b"}  # Где стоит открывать.
HEAT_SKIP = {0.2: "#fddbc7", 0.45: "#f4a582", 0.65: "#d6604d", 0.85: "#b2182b", 1: "#67001f"}  # Где открывать не нужно.
# Подложки Esri: сервис и название в списке.
BASEMAPS = [("World_Imagery", "Спутник Esri"), ("World_Street_Map", "Карта Esri"), ("World_Topo_Map", "Топографическая Esri")]
# Стили карты: размер, кнопки слоёв, легенда, номерные метки, карточки объектов и мест-кандидатов.
MAP_CSS = """<style>
#map_div {height:100vh !important;}                     /* Карта заполняет окно компонента, высоту задаёт страница. */
#map_div:not(.draw-enabled) .leaflet-draw {display:none;}
.leaflet-control-layers {border:none !important; border-radius:10px !important; box-shadow:0 2px 10px rgba(24,43,58,.18) !important;
                         font:13px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;}
.leaflet-control-layers-toggle {width:38px !important; height:38px !important; background-size:20px !important;}
.basemaps .leaflet-control-layers-toggle {background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23182b3a' stroke-width='1.8' stroke-linejoin='round'%3E%3Cpath d='M3 6.5l6-2.5 6 2.5 6-2.5v13.5l-6 2.5-6-2.5-6 2.5z'/%3E%3Cpath d='M9 4v13.5M15 6.5V20'/%3E%3C/svg%3E") !important;}
.leaflet-control-layers-expanded {padding:8px 12px 8px 8px !important;}
.map-legend {position:absolute; z-index:1000; right:10px; bottom:22px; background:rgba(255,255,255,.94); border-radius:10px;
             padding:8px 12px; box-shadow:0 2px 10px rgba(24,43,58,.18); font:12px/1.4 -apple-system,'Segoe UI',Roboto,sans-serif; color:#182b3a;}
.map-legend .bar {height:9px; width:190px; border-radius:5px; margin:5px 0 3px;}
.map-legend .ends {display:flex; justify-content:space-between; color:#5b6b7b; font-size:11px;}
.pin {width:26px; height:26px; border-radius:50%; border:2px solid #fff; box-shadow:0 2px 6px rgba(0,0,0,.45);
      display:grid; place-items:center; font:700 12px/1 -apple-system,'Segoe UI',Roboto,sans-serif; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.6);}
.leaflet-popup-content-wrapper {border-radius:14px !important; box-shadow:0 8px 28px rgba(24,43,58,.25) !important;}
.leaflet-popup-content {margin:14px 16px !important; font:13px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif; color:#182b3a;}
.pc {min-width:240px;}
.pc-head {display:flex; align-items:center; gap:10px;}
.pc-rank {flex:none; width:32px; height:32px; border-radius:50%; display:grid; place-items:center; color:#fff;
          font-weight:700; font-size:14px; text-shadow:0 1px 2px rgba(0,0,0,.5);}
.pc-dot {flex:none; width:12px; height:12px; border-radius:50%; background:#fff; border:2px solid #182b3a;}
.pc-title {font-weight:700; font-size:14.5px; letter-spacing:-.01em;}
.pc-sub {font-size:12px; color:#5b6b7b;}
.pc-addr {margin:10px 0 0; padding-top:9px; border-top:1px solid #e1e9ed; font-size:13px;}
.pc-row {margin-top:4px; font-size:12.5px; color:#3b4b5a;}
.pc-row span {color:#8593a1;}
.pc-stats {display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin:10px 0 8px;}
.pc-stats div {background:#f2f5f7; border-radius:9px; padding:6px 4px; text-align:center;}
.pc-stats b {display:block; font-size:16px; letter-spacing:-.01em;}
.pc-stats span {font-size:11px; color:#5b6b7b;}
.pc-stats .gain {background:#e3f3e5;} .pc-stats .gain b {color:#1b7837;}
.pc-coord {font-size:11px; color:#8593a1;}
</style>"""


class Controls(MacroElement):
    """Две кнопки управления картой: «Подложка» (базовые карты) и «Слои» (слои расчёта)."""
    # Скрипт Leaflet: два переключателя слоёв в правом верхнем углу.
    _template = Template("""{% macro script(this, kwargs) %}
    var bases_{{ this.get_name() }} = L.control.layers({ {% for n, l in this.bases %}{{ n|tojson }}: {{ l.get_name() }},{% endfor %} }, {},
        {collapsed: true, position: 'topright'}).addTo({{ this._parent.get_name() }});
    bases_{{ this.get_name() }}.getContainer().classList.add('basemaps');
    bases_{{ this.get_name() }}.getContainer().title = 'Подложка';
    var layers_{{ this.get_name() }} = L.control.layers({}, { {% for n, l in this.overlays %}{{ n|tojson }}: {{ l.get_name() }},{% endfor %} },
        {collapsed: true, position: 'topright'}).addTo({{ this._parent.get_name() }});
    layers_{{ this.get_name() }}.getContainer().title = 'Слои';
    {% endmacro %}""")

    def __init__(self, bases, overlays):  # Списки для двух кнопок.
        super().__init__()  # Инициализация элемента Folium.
        self._name = "Controls"  # Начало имени переменной в скрипте.
        self.bases, self.overlays = bases, overlays  # Пары (название, слой).


def legend(m, title, colors, low, high):
    """Легенда-шкала в правом нижнем углу карты."""
    gradient = ", ".join(colors)  # Цвета шкалы слева направо.
    # Легенда-шкала поверх карты.
    m.get_root().html.add_child(Element(f'<div class="map-legend"><b>{title}</b><div class="bar" style="background:linear-gradient('
                                        f'90deg, {gradient})"></div><div class="ends"><span>{low}</span><span>{high}</span></div></div>'))


def base_map(bounds=None):
    """Подложки Esri (по умолчанию – спутник) без строки атрибуции; bounds – запад, юг, восток, север."""
    # Карта без своей подложки и строки атрибуции, со шкалой расстояний.
    m = folium.Map(location=[55.7558, 37.6176], zoom_start=11, tiles=None, control_scale=True, attribution_control=False)
    m.bases = []  # Подложки для кнопки «Подложка».
    for service, name in BASEMAPS:  # Подложка за подложкой.
        # Тайлы Esri; виден только спутник.
        layer = folium.TileLayer(f"https://services.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}",
                                 attr="Esri", name=name, show=service == "World_Imagery", control=False).add_to(m)
        m.bases.append((name, layer))  # В список кнопки «Подложка».
    m.overlays = []  # Слои расчёта для кнопки «Слои».
    m.get_root().header.add_child(Element(MAP_CSS))  # Стили карты в заголовке страницы.
    if bounds is not None:  # Охват задан.
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # Углы: юго-запад и северо-восток.
    return m  # Карта пока без слоёв расчёта.


# Фокус по щелчку на карте и метках – без прокрутки страницы: иначе карта выше окна сдвигает страницу и метка уходит из-под курсора.
NO_SCROLL_FOCUS = """
        (function () { var focus = HTMLElement.prototype.focus;
          HTMLElement.prototype.focus = function (o) { return focus.call(this, Object.assign({preventScroll: true}, o || {})); };
          document.addEventListener('mousedown', function (e) {
            if (e.target.closest && e.target.closest('.leaflet-marker-icon')) e.preventDefault(); }, true); })();
    """


class Script(MacroElement):
    """Код JavaScript внутри скрипта карты: st_folium выполняет только его, а не вставки в HTML."""
    _template = Template("{% macro script(this, kwargs) %}{{ this.code }}{% endmacro %}")  # Код вставляется как есть.

    def __init__(self, code):  # Код JavaScript – текстом.
        super().__init__()  # Инициализация элемента Folium.
        self.code = code  # Текст скрипта.


def drawing_tools(m):
    """Подписи инструментов и квадрат с равными сторонами в проекции карты."""
    # Русские подписи, квадрат вместо прямоугольника, свойства фигуры (номер, вид, радиус) для передачи в Python.
    m.add_child(Script(f"""(() => {{
        const t = L.drawLocal.draw;
        t.toolbar.buttons.polygon = 'Нарисовать полигон';
        t.toolbar.buttons.rectangle = 'Нарисовать квадрат';
        t.toolbar.buttons.circle = 'Нарисовать круг';
        t.toolbar.actions.text = 'Отмена';
        t.toolbar.actions.title = 'Отменить рисование';
        t.toolbar.finish.text = 'Завершить';
        t.toolbar.finish.title = 'Завершить полигон';
        t.toolbar.undo.text = 'Удалить последнюю точку';
        t.toolbar.undo.title = 'Удалить последнюю точку';
        t.handlers.polygon.tooltip.start = 'Щёлкните, чтобы начать полигон';
        t.handlers.polygon.tooltip.cont = 'Щёлкните, чтобы продолжить полигон';
        t.handlers.polygon.tooltip.end = 'Щёлкните первую точку, чтобы замкнуть полигон';
        t.handlers.rectangle.tooltip.start = 'Нажмите и потяните от угла квадрата';
        t.handlers.circle.tooltip.start = 'Нажмите и потяните от центра круга';
        t.handlers.circle.radius = 'Радиус';
        t.handlers.simpleshape.tooltip.end = 'Отпустите кнопку, чтобы завершить';

        const drawRectangle = L.Draw.Rectangle.prototype._drawShape;
        L.Draw.Rectangle.include({{
            _drawShape(latlng) {{
                const start = this._map.project(this._startLatLng);
                const delta = this._map.project(latlng).subtract(start);
                const side = Math.max(Math.abs(delta.x), Math.abs(delta.y));
                const corner = start.add(L.point(delta.x < 0 ? -side : side, delta.y < 0 ? -side : side));
                drawRectangle.call(this, this._map.unproject(corner));
            }}
        }});

        {m.get_name()}.on(L.Draw.Event.CREATED, (e) => {{
            const properties = {{area_draw_id: Date.now() + '-' + L.stamp(e.layer), area_shape: e.layerType}};
            if (e.layerType === 'circle') properties.radius = e.layer.getRadius();
            e.layer.feature = {{type: 'Feature', properties}};
        }});
    }})();"""))
    style = {"shapeOptions": {"color": ACCENT}}  # Цвет фигур при рисовании.
    Draw(draw_options={"polygon": style, "rectangle": style, "circle": style,  # Только полигон, квадрат и круг, без правки и удаления.
                       "polyline": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}, show_geometry_on_click=False).add_to(m)


def drawing_mode(enabled):
    """Показать инструменты полигона без пересоздания карты."""
    group = folium.FeatureGroup(control=False)  # Группа, которую st_folium обновляет без перезагрузки карты.
    # Выключение отменяет начатое рисование и скрывает панель.
    group.add_child(Script(f"""(() => {{
        const enabled = {json.dumps(enabled)};
        const map = map_div.getContainer();
        if (!enabled) map.querySelector('.leaflet-draw-actions a[title="Отменить рисование"]')?.click();
        map.classList.toggle('draw-enabled', enabled);
    }})();"""))
    return group  # Группа для feature_group_to_add.


def remember_view(m, bounds):
    """Восстановить положение карты при смене слоёв той же области."""
    region = json.dumps(json.dumps(bounds))  # Область как строка JavaScript.
    # Центр и масштаб хранятся в sessionStorage, пока область та же.
    m.add_child(Script(f"""(() => {{
        const map = {m.get_name()};
        const region = {region};
        const key = 'geomarketing-map-view';
        try {{
            const saved = JSON.parse(sessionStorage.getItem(key));
            if (saved?.region === region) map.setView(saved.center, saved.zoom, {{animate: false}});
            const save = () => {{
                sessionStorage.setItem(key, JSON.stringify({{region, center: map.getCenter(), zoom: map.getZoom()}}));
            }};
            save();
            map.on('moveend', save);
        }} catch {{ /* Карта также работает при отключённом хранилище браузера. */ }}
    }})();"""))


def finish(m):
    """Кнопки «Подложка» и «Слои» – после всех слоёв, затем фокус карты без прокрутки страницы."""
    m.add_child(Controls(m.bases, m.overlays))  # Кнопки видят все добавленные слои.
    m.add_child(Script(NO_SCROLL_FOCUS))  # Скрипт – в коде карты, чтобы st_folium его выполнил.
    return m  # Готовая карта.


def overlay(m, layer, name):  # Слой на карте и в списке кнопки «Слои».
    layer.add_to(m)  # Слой на карте.
    m.overlays.append((name, layer))  # Название в переключателе.


def num(value, digits=2):  # Число с десятичной запятой, без разделения разрядов.
    return f"{value:.{digits}f}".replace(".", ",")  # 1.25 → 1,25


def add_boundary(m, boundary):  # Граница города или области – белый пунктир без заливки.
    folium.GeoJson(boundary, control=False,  # Граница поверх слоёв, без переключателя.
                   style_function=lambda f: {"color": "white", "weight": 2, "dashArray": "6 4", "fill": False}).add_to(m)


def result_layer(collection, columns=()):
    """Слой из ответа сервиса, в том числе при отсутствии объектов или кандидатов."""
    if collection["features"]:  # В ответе есть объекты.
        return gpd.GeoDataFrame.from_features(collection, crs="EPSG:4326")  # Слой GeoJSON → GeoDataFrame.
    return gpd.GeoDataFrame({name: [] for name in columns}, geometry=[], crs="EPSG:4326")  # Пустой слой с нужными столбцами.


def candidate_layer(result):  # Места-кандидаты из ответа сервиса.
    return result_layer(result["candidates"], ("адрес", "ожидаемое", "объектов темы", "потенциал"))  # Пустой слой – с этими столбцами.


def prepare_cells(cells):
    """Только нужные свойства, координаты с точностью около 1 м, подписи с десятичной запятой."""
    cells = cells[["ожидаемое", "объектов темы", "потенциал", "geometry"]].copy()  # Копия: исходная таблица не меняется.
    cells["geometry"] = cells.geometry.set_precision(1e-5)  # Около 1 м: меньше объём HTML.
    cells["Ожидается"] = cells["ожидаемое"].map(num)  # Подписи для всплывающих подсказок.
    cells["Есть сейчас"] = cells["объектов темы"].astype(int).astype(str)  # Целое число без запятой.
    cells["Потенциал"] = cells["потенциал"].map(lambda v: ("+" if v > 0 else "") + num(v))  # Плюс у положительного.
    return cells  # Ячейки для слоёв карты.


def add_heat(m, cells):
    """Тепловая карта потенциала: зелёная – где объектов меньше ожидаемого (стоит открывать), красная – где больше."""
    group = folium.FeatureGroup(control=False)  # Обе тепловые карты – один слой.
    for sign, gradient in [(-1, HEAT_SKIP), (1, HEAT_OPEN)]:  # Сначала красная, затем зелёная.
        part = cells[sign * cells["потенциал"] > 0.02]  # Ячейки нужного знака без околонулевых.
        if len(part):  # Ячейки этого знака есть.
            weights = sign * part["потенциал"]  # Вес точки – модуль потенциала.
            top = float(np.quantile(weights, 0.95))  # Насыщение цвета на 95-м процентиле, а не на одном выбросе.
            # Тепловая карта по центрам ячеек; вес срезан по top – полный цвет начинается с 95-го процентиля.
            HeatMap([[p.y, p.x, min(float(w), top)] for p, w in zip(part.geometry.centroid, weights)], radius=22, blur=18,
                    min_opacity=0.25, gradient=gradient).add_to(group)
    overlay(m, group, "Тепловая карта потенциала")  # Слой и пункт в кнопке «Слои».


def add_gap(m, cells, show=False, opacity=0.6):
    """Потенциал по ячейкам: зелёный – объектов меньше ожидаемого (стоит открывать), красный – больше."""
    limit = max(float(np.quantile(np.abs(cells["потенциал"]), 0.98)), 0.3)  # Край шкалы: 98-й процентиль модуля, не меньше 0,3.
    scale = cm.LinearColormap(GAP_COLORS, vmin=-limit, vmax=limit)  # Шкала симметрична относительно нуля.
    cells = cells[["потенциал", "Ожидается", "Есть сейчас", "Потенциал", "geometry"]]  # Только нужные свойства.
    overlay(m, folium.GeoJson(cells, show=show, control=False,  # Цвет ячейки по потенциалу; за краем шкалы – цвет края.
                              style_function=lambda f: {"fillColor": scale(max(-limit, min(limit, f["properties"]["потенциал"]))),
                                                        "fillOpacity": opacity, "color": "white", "weight": 0.15},
                              tooltip=folium.GeoJsonTooltip(["Ожидается", "Есть сейчас", "Потенциал"], sticky=True)),
            "Потенциал по ячейкам")


def add_expected(m, cells, show=False):  # Ожидаемое число объектов по ячейкам – синяя шкала.
    top = max(float(np.quantile(cells["ожидаемое"], 0.98)), 0.1)  # Верх шкалы: 98-й процентиль, не меньше 0,1.
    scale = cm.LinearColormap(["#f7fbff", "#9ecae1", "#3182bd", "#08306b"], vmin=0, vmax=top)  # От белого к тёмно-синему.
    cells = cells[["ожидаемое", "Ожидается", "Есть сейчас", "geometry"]]  # Только нужные свойства.
    overlay(m, folium.GeoJson(cells, show=show, control=False,  # Синие ячейки; выше верха шкалы – цвет верха.
                              style_function=lambda f: {"fillColor": scale(min(top, f["properties"]["ожидаемое"])), "fillOpacity": 0.6,
                                                        "color": "white", "weight": 0.15},
                              tooltip=folium.GeoJsonTooltip(["Ожидается", "Есть сейчас"], sticky=True)),
            "Ожидаемое число")


def add_targets(m, targets, label):  # Существующие объекты темы – белые точки с карточкой.
    group = folium.FeatureGroup(control=False)  # Все объекты – один слой.
    for _, row in targets.iterrows():  # Карточка и точка на каждый объект.
        point = row.geometry  # Точка объекта.
        name = row.get("name")  # Название из OpenStreetMap.
        name = name if isinstance(name, str) else label  # Без названия – подпись темы.
        # Заголовок карточки: название и тема.
        card = (f'<div class="pc"><div class="pc-head"><span class="pc-dot"></span><div><div class="pc-title">{name}</div>'
                f'<div class="pc-sub">{label}: существующий объект</div></div></div>')
        if isinstance(row.get("addr:street"), str):  # Адрес есть в OpenStreetMap.
            card += f'<div class="pc-addr">{row["addr:street"]} {row.get("addr:housenumber") or ""}</div>'  # Улица и дом.
        if isinstance(row.get("opening_hours"), str):  # Часы работы указаны.
            card += f'<div class="pc-row"><span>Часы работы:</span> {row["opening_hours"]}</div>'  # Строка с часами работы.
        card += f'<div class="pc-coord" style="margin-top:8px">{point.y:.5f}, {point.x:.5f}</div></div>'  # Широта, долгота.
        # Белая точка с карточкой по щелчку.
        folium.CircleMarker([point.y, point.x], radius=4, color="#182b3a", weight=1, fill=True, fill_color="white",
                            fill_opacity=1, tooltip=name, popup=folium.Popup(card, max_width=300)).add_to(group)
    overlay(m, group, f"{label}: существующие")  # Слой объектов темы.


def add_candidates(m, candidates, label):
    """Места-кандидаты – номерные метки, цвет по потенциалу: чем темнее зелёный, тем выше потенциал."""
    if candidates.empty:  # Расчёт не нашёл мест-кандидатов.
        return  # Нет кандидатов – нет слоя.
    group = folium.FeatureGroup(control=False)  # Все метки – один слой.
    low, high = float(candidates["потенциал"].min()), float(candidates["потенциал"].max())  # Пределы шкалы цвета.
    scale = cm.LinearColormap(GREEN, vmin=low, vmax=max(high, low + 1e-6))  # Шкала не вырождается при равных значениях.
    for i, (_, row) in enumerate(candidates.iterrows(), 1):  # Номера мест с 1.
        c = row.geometry.centroid  # Точка метки.
        color = scale(row["потенциал"])  # Цвет номера.
        gain = num(row["потенциал"])  # Потенциал с запятой.
        place = row.get("адрес")  # Адрес из обратного геокодирования.
        place = place if isinstance(place, str) and place else "Адрес не найден"  # Пустой адрес – заглушка.
        # Карточка: адрес, три числа, координаты.
        popup = (f'<div class="pc"><div class="pc-head"><span class="pc-rank" style="background:{color}">{i}</span><div>'
                 f'<div class="pc-title">Место-кандидат №{i}</div><div class="pc-sub">{label}</div></div></div>'
                 f'<div class="pc-addr">{place}</div>'
                 f'<div class="pc-stats"><div><b>{num(row["ожидаемое"])}</b><span>ожидается</span></div>'
                 f'<div><b>{int(row["объектов темы"])}</b><span>есть сейчас</span></div>'
                 f'<div class="gain"><b>+{gain}</b><span>потенциал</span></div></div>'
                 f'<div class="pc-coord">{c.y:.5f}, {c.x:.5f}</div></div>')
        folium.Marker([c.y, c.x], tooltip=f"№{i}: потенциал +{gain}", popup=folium.Popup(popup, max_width=320),  # Круглая метка с номером.
                      icon=folium.DivIcon(html=f'<div class="pin" style="background:{color}">{i}</div>',
                                          icon_size=(26, 26), icon_anchor=(13, 13))).add_to(group)
    overlay(m, group, "Места-кандидаты")  # Слой мест-кандидатов.


def value_label(value):  # Значение для подсказки: целое – без дробной части, дробное – с запятой, текст – как есть.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):  # Не число.
        return "" if value is None else str(value)  # Текст или пустая строка.
    if not np.isfinite(value):  # Пропуск в данных.
        return "–"  # Прочерк вместо nan.
    return fmt(float(value), 0) if float(value).is_integer() else num(float(value))  # 1500 → «1 500», 0.25 → «0,25».


def add_extension(m, layer, name, column=None):
    """Слой собственной функции: жёлтый контур и заливка по числовому столбцу column – от светлой (меньше) к тёмной
    (больше); без такого столбца – только контур. Подсказка показывает значения столбцов слоя."""
    if layer is None or not len(layer):  # Функция не вернула слой.
        return
    fields = [c for c in layer.columns if c != "geometry"][:6]  # До шести столбцов в подсказке.
    shown = layer[["geometry"]].copy()  # Слой для карты: подписи вместо исходных значений.
    for c in fields:  # Каждый столбец подсказки.
        shown[c] = layer[c].map(value_label)  # Число с запятой.
    tooltip = folium.GeoJsonTooltip(fields, sticky=True) if fields else None  # Подсказка при наведении.
    colored = column in layer.columns and pd.api.types.is_numeric_dtype(layer[column]) and layer[column].notna().any()
    if not colored:  # Столбец не задан или не числовой – только контур.
        overlay(m, folium.GeoJson(shown, control=False, tooltip=tooltip,  # Жёлтый контур без заливки.
                                  style_function=lambda f: {"color": "#ffd166", "weight": 2.5, "fillOpacity": 0}), name)
        return
    values = layer[column].astype(float)  # Значения для цвета.
    low, high = float(values.min()), float(values.max())  # Пределы шкалы.
    scale = cm.LinearColormap(EXT_COLORS, vmin=low, vmax=max(high, low + 1e-9))  # Шкала не вырождается при равных значениях.
    shown["_цвет"] = [scale(v) if np.isfinite(v) else "#9aa5b1" for v in values]  # Цвет каждой геометрии; пропуск – серый.
    overlay(m, folium.GeoJson(shown, control=False, tooltip=tooltip,  # Заливка по значению, жёлтый контур.
                              style_function=lambda f: {"fillColor": f["properties"]["_цвет"], "fillOpacity": 0.75,
                                                        "color": "#ffd166", "weight": 1.5}), name)


def full_map(result, boundary, label):
    """Карта со всеми слоями расчёта; по умолчанию видны тепловая карта, объекты и места-кандидаты."""
    cells = prepare_cells(gpd.GeoDataFrame.from_features(result["cells"], crs="EPSG:4326"))  # Ячейки H3 с подписями.
    m = base_map(cells.total_bounds)  # Карта на охват ячеек.
    add_expected(m, cells)  # Скрытые слои ячеек – первыми.
    add_gap(m, cells)  # Потенциал по ячейкам, скрыт.
    add_heat(m, cells)  # Тепловая карта, видна.
    if boundary is not None:  # Граница известна.
        add_boundary(m, boundary)  # Белый пунктир поверх ячеек.
    ext = result["extension"]  # Ответ функции расширения.
    add_extension(m, result_layer(ext["layer"]) if ext.get("layer") else None, ext["name"], ext.get("column"))  # Слой расширения, если он есть.
    add_targets(m, result_layer(result["targets"]), label)  # Существующие объекты темы.
    add_candidates(m, candidate_layer(result), label)  # Места-кандидаты – поверх объектов.
    legend(m, "Потенциал размещения", GAP_COLORS, "открывать не нужно", "стоит открывать")  # Легенда шкалы потенциала.
    return finish(m)  # Кнопки слоёв – последними.


# Стили отчёта: страница, заголовки, карточки, таблицы, карты и плитки показателей.
STYLE = """body{font:15px/1.6 -apple-system,'Segoe UI',Roboto,sans-serif;color:#182b3a;background:#f6f8fa;margin:0}
main{max-width:1000px;margin:0 auto;padding:36px 24px 48px}h1{font-size:28px;margin:0;letter-spacing:-.04em}
.sub{color:#5b6b7b;font-size:14px;margin:6px 0 24px}h2{font-size:19px;margin:36px 0 6px;letter-spacing:-.02em}
.card{background:#fff;border:1px solid #e1e9ed;border-radius:12px;padding:16px 20px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e1e9ed;padding:6px 8px;text-align:left}
.map{border-radius:12px;overflow:hidden;border:1px solid #e1e9ed}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:#fff;border:1px solid #e1e9ed;border-radius:12px;padding:12px 16px}
.metric b{display:block;font-size:22px;letter-spacing:-.02em}.metric span{color:#5b6b7b;font-size:12px}.note{color:#8593a1;font-size:12px}
@media (max-width:700px){.metrics{grid-template-columns:repeat(2,1fr)}}"""


def tags_text(tags, join=", "):  # Теги OpenStreetMap строкой: «ключ=значение|значение», «ключ=*» – любое значение.
    # Условия одной группы – через join (у фактора – «или»), группы – через точку с запятой.
    groups = [join.join(f"{k}={'|'.join(v) if isinstance(v, list) else '*'}" for k, v in group.items()) for group in tags]
    return "; ".join(groups) or "–"  # Прочерк, если тегов нет.


def section(title, text, m=None):
    """Раздел отчёта: заголовок, описание по шаблону и карта слоя."""
    block = f"<h2>{title}</h2><p>{text}</p>"  # Заголовок и описание раздела.
    if m is not None:  # Раздел с картой.
        block += f'<div class="map">{m._repr_html_()}</div>'  # Карта folium во встроенном окне.
    return block  # HTML-фрагмент раздела.


def build_report(result, boundary, params):
    """Отчёт HTML: показатели, параметры, карты слоёв с описаниями и таблица мест-кандидатов."""
    s, ext, label = result["summary"], result["extension"], params["label"]  # Итоги, расширение, подпись темы.
    cells = prepare_cells(gpd.GeoDataFrame.from_features(result["cells"], crs="EPSG:4326"))  # Ячейки H3 с подписями.
    targets = result_layer(result["targets"])  # Существующие объекты.
    candidates = candidate_layer(result)  # Места-кандидаты.
    bounds = cells.total_bounds  # Общий охват всех карт.

    m1 = base_map(bounds)  # Карта 1: граница и существующие объекты.
    add_boundary(m1, boundary)  # Граница города или области.
    add_targets(m1, targets, label)  # Существующие объекты темы.
    finish(m1)  # Кнопки слоёв.
    m2 = base_map(bounds)  # Карта 2: ожидаемое число по ячейкам.
    add_expected(m2, cells, show=True)  # Ожидаемое число, слой виден.
    add_boundary(m2, boundary)  # Граница поверх ячеек.
    finish(m2)  # Кнопки слоёв.
    legend(m2, "Ожидаемое число в ячейке", ["#f7fbff", "#9ecae1", "#3182bd", "#08306b"], "0", "больше")  # Синяя шкала ожидаемого числа.
    m3 = base_map(bounds)  # Карта 3: потенциал, тепловая карта, места-кандидаты.
    add_gap(m3, cells, show=True)  # Потенциал по ячейкам, слой виден.
    add_heat(m3, cells)  # Тепловая карта потенциала.
    add_boundary(m3, boundary)  # Граница поверх ячеек.
    add_candidates(m3, candidates, label)  # Номерные метки мест-кандидатов.
    finish(m3)  # Кнопки слоёв.
    legend(m3, "Потенциал размещения", GAP_COLORS, "открывать не нужно", "стоит открывать")  # Шкала потенциала.
    table = pd.DataFrame({"Адрес": candidates.get("адрес", ""),  # Таблица мест-кандидатов.
                          "Широта, долгота": [f"{p.y:.5f}, {p.x:.5f}" for p in candidates.geometry.centroid],
                          "Ожидается": candidates["ожидаемое"].map(lambda v: fmt(v, 2)),
                          "Есть сейчас": candidates["объектов темы"].astype(int),
                          "Потенциал": candidates["потенциал"].map(lambda v: "+" + fmt(v, 2))})
    table.index = pd.RangeIndex(1, len(candidates) + 1, name="№")  # Номера 1–10 после сборки: иначе строки сдвигаются.

    is_city = params.get("method", "Город") == "Город"  # Город из списка или своя область.
    region = "городе" if is_city else "области"  # Окончание для подзаголовка.
    subject = "Город" if is_city else "Область"  # Начало заголовка раздела 1.
    selection = "Ячейки H3 разрешения 9 – все, которые лежат внутри границы или пересекают её"  # Как ячейки отбираются.
    title = html.escape(params["title"])  # Заголовок отчёта и вкладки.
    # Подпись и значение каждой плитки.
    metrics = [(f"{label.lower()}: есть сейчас", fmt(s["объектов темы"], 0)), ("ожидается по модели", fmt(s["ожидаемое число"])),
               ("ячеек сетки H3", fmt(s["ячеек"], 0)),
               ("ячеек вне диапазона обучения", fmt(s["ячеек вне диапазона обучения, %"]) + " %")]
    tiles = "".join(f'<div class="metric"><b>{v}</b><span>{k}</span></div>' for k, v in metrics)  # Плитки показателей.
    # Шапка, плитки, параметры, три раздела с картами, таблица.
    parts = [f"<h1>{title}</h1><div class='sub'>{html.escape(label)} в {region} «{html.escape(s['город'])}» · "
             f"режим – {s['режим']}; {datetime.date.today():%d.%m.%Y}</div>",
             f'<div class="metrics">{tiles}</div>',
             f'<div class="card">{html.escape(label)}: <b>{html.escape(tags_text(params["target"]))}</b><br>Дополнительные факторы: '
             f'{html.escape(tags_text(params["extra_env"], " или "))}<br>Модель: {html.escape(s["модель"])}; '
             f'обучена: {html.escape(s["города обучения"])}</div>',
             section(f"1. {subject} и {html.escape(label.lower())}", f"Граница и {fmt(s['объектов темы'], 0)} существующих объектов "
                     f"«{html.escape(label.lower())}» из OpenStreetMap. {selection}: "
                     f"{fmt(s['ячеек'], 0)} ячеек площадью около 0,1 км².", m1),
             section("2. Ожидаемое число объектов", f"Модель «{html.escape(s['модель'])}» рассчитала по окружению каждой ячейки "
                     f"(объекты 12 категорий, улицы, население GHS-POP), сколько объектов можно ожидать: всего "
                     f"{fmt(s['ожидаемое число'])} при фактических {fmt(s['объектов темы'], 0)}.", m2),
             section("3. Потенциал и места-кандидаты", "Потенциал – разность ожидаемого и фактического числа объектов в ячейке. "
                     "Зелёные ячейки и зелёные пятна тепловой карты – объектов меньше, чем позволяет окружение (стоит открывать), "
                     "красные – больше (открывать не нужно); номерные метки – десять мест-кандидатов с наибольшим потенциалом, "
                     "чем темнее зелёный, тем выше потенциал"
                     + (", среди ячеек в диапазоне признаков городов обучения." if s["режим"] == "своя модель" else "."), m3),
             f'<div class="card">{table.to_html(border=0)}</div>']
    if ext.get("name"):  # Раздел функции расширения.
        m4 = None  # Без слоя – раздел без карты.
        if ext.get("layer"):  # Функция вернула слой.
            m4 = base_map(bounds)  # Карта 4: слой расширения.
            add_boundary(m4, boundary)  # Граница области.
            add_extension(m4, result_layer(ext["layer"]), ext["name"], ext.get("column"))  # Слой функции расширения.
            finish(m4)  # Кнопки слоёв.
        parts.append(section(f"4. {html.escape(ext['name'])}", html.escape(ext.get("text") or ""), m4))  # Раздел: название, вывод, карта.
        if ext.get("table"):  # Функция вернула таблицу.
            parts.append(f'<div class="card">{pd.DataFrame(ext["table"]).to_html(border=0, index=False)}</div>')  # Таблица расширения.
    if s["режим"] == "своя модель" and s["ячеек вне диапазона обучения, %"] > 20:  # Предупреждение об экстраполяции.
        # Текст предупреждения.
        parts.append('<div class="card">⚠️ Больше пятой части ячеек выходит за диапазон признаков городов обучения: '
                     "прогноз для этой области – экстраполяция.</div>")
    if s.get("metrics"):  # Метрики есть только при обучении на лету.
        parts.append("<h2>Проверка моделей по пространственным блокам</h2><div class='card'>"  # Таблица метрик: строки – модели.
                     + pd.DataFrame(s["metrics"]).T.to_html(border=0) + "</div>")
    source = "Объекты и границы" if is_city else "Объекты"  # Для области граница своя.
    parts.append(f"<p class='note'>{source} – © сообщество OpenStreetMap (ODbL), население – GHS-POP "  # Подпись об источниках данных.
                 "(CC BY 4.0), подложка – Esri.</p>")
    head = f"<head><meta charset='utf-8'><title>{title}</title><style>{STYLE}</style></head>"  # Кодировка, заголовок, стили.
    return f"<!doctype html><html lang='ru'>{head}<body><main>{''.join(parts)}</main></body></html>"  # Полная страница отчёта.
