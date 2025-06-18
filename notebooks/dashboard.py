import pickle
import threading
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from optimization import load_speeds, optimize_staff
from tensorflow.keras.models import load_model

ROOT_DIR = Path.cwd().resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" 
MODELS_TASK1_DIR = MODELS_DIR / "task1" # Модели для задачи 1
MODELS_TASK2_DIR = MODELS_DIR / "task2" # Модели для задачи 2

DATA_FILE_TASK1 = DATA_DIR / "data1_logs_daily.xlsx" # Данные для задачи 1
DATA_FILE_TASK2 = DATA_DIR / "data2_all.csv"

# Импорт моделей
# Задача 1: Прием заявок
MODEL_LSTM = MODELS_TASK1_DIR / "lstm_daily_model.keras"
MODEL_SARIMA = MODELS_TASK1_DIR / "arima_daily_model.pkl"
MODEL_XGB = MODELS_TASK1_DIR / "xgb_daily_model.pkl"

with open(MODEL_SARIMA, "rb") as file:
    sarima = pickle.load(file)
with open(MODEL_XGB, "rb") as file:
    xgb = pickle.load(file)

MODELS_TASK1 = {
    'LSTM'  : load_model(MODEL_LSTM, compile=False),
    'XGB'   : xgb,
    'SARIMA': sarima
}

# Задача 2: Прием оригиналов
MODEL_TASK2_ARIMAX_IN = MODELS_TASK2_DIR / "arimax_in.pkl"
MODEL_TASK2_ARIMAX_OUT = MODELS_TASK2_DIR / "arimax_out.pkl"
MODEL_TASK2_RF_IN = MODELS_TASK2_DIR / "rf_in.pkl"
MODEL_TASK2_RF_OUT = MODELS_TASK2_DIR / "rf_out.pkl"
MODEL_TASK2_RNN_IN = MODELS_TASK2_DIR / "rnn_in.keras"
MODEL_TASK2_RNN_OUT = MODELS_TASK2_DIR / "rnn_out.keras"

with open(MODEL_TASK2_ARIMAX_IN, "rb") as file:
    arimax_in = pickle.load(file)
with open(MODEL_TASK2_ARIMAX_OUT, "rb") as file:
    arimax_out = pickle.load(file)
with open(MODEL_TASK2_RF_IN, "rb") as file:
    rf_in = pickle.load(file)
with open(MODEL_TASK2_RF_OUT, "rb") as file:
    rf_out = pickle.load(file)

MODELS_TASK2 = {
    "RNN_IN": load_model(MODEL_TASK2_RNN_IN, compile=False),
    "RNN_OUT": load_model(MODEL_TASK2_RNN_OUT, compile=False),
    "RF_IN": rf_in,
    "RF_OUT": rf_out,
    "ARIMAX_IN": arimax_in,
    "ARIMAX_OUT": arimax_out
}

LAGS = list(range(1,8))
WINDOWS = list(range(1,8))
MAX_HORIZON = 36
MAX_HORIZON_TASK2 = 10
COLOR_MAIN = "#1f77b4"

X_AXIS_START, X_AXIS_END = (6,19), (7,26) # 19.06 – 26.07
DATA_START, DATA_END  = (6,20), (7,25) # 20.06 – 25.07

TASK2_START = (7, 27)
TASK2_END = {
    2014: (8, 9),
    2015: (8, 6),
    2016: (8, 8),
    2017: (8, 8),
    2018: (8, 8),
    2019: (8, 8),
    2020: (8, 8),
    2021: (8, 11),
    2022: (8, 3),
    2023: (8, 3),
    2024: (8, 3),
    2025: (8, 5)
}

WEEK_MS = 7*24*3600*1000

# Данные для задачи 1
df_actual = pd.read_excel(DATA_FILE_TASK1)
df_actual['date'] = pd.to_datetime(df_actual['date'])
df_actual.sort_values('date', inplace=True)

raw = pd.read_csv(DATA_DIR / "first_mention.csv", index_col=0, low_memory=False)
raw['zayav_d'] = pd.to_datetime(raw['zayav_d'], errors='coerce')

RAW_LOGS_TASK1 = (
    raw
    .assign(date=lambda df: df['zayav_d'].dt.floor('D'))
    [['date', 'y',
      'is_parallel', 'pk_user_kod', 'is_interdekanat']]
    .dropna(subset=['date'])       # выбрасываем строки, где zayav_d = NaT
    .copy()
)

# Данные для задачи 2
df_task2 = pd.read_csv(DATA_FILE_TASK2, parse_dates=['date'])
df_task2["all"] = df_task2["in"] + df_task2["out"]

logs_task2 = pd.read_csv(DATA_DIR / 'raw_logs_task2.csv', parse_dates=['date'])
logs_task2['date'] = logs_task2['date'].dt.floor("D")
RAW_LOGS_TASK2 = logs_task2.copy()

SPEEDS = load_speeds(DATA_DIR / "first_mention.csv")

def load_speeds_task2(logs: pd.DataFrame) -> np.ndarray:
    curr_date = datetime.today().date()
    curr_year = datetime.today().year
    target_year = (curr_year - 1) if curr_date <= date(curr_year, *TASK2_END.get(curr_year, (8, 5))) else curr_year
    df = logs[logs['date'].dt.year == target_year]

    df['date_day'] = df['date'].dt.floor("D")
    daily = df.groupby(['pk_user_kod', 'date_day']).size().reset_index(name="ops")
    summary = daily.groupby("pk_user_kod")["ops"].agg(["mean", "count"])
    summary = summary[summary['count'] >= 8]
    return summary['mean'].values

SPEEDS_TASK2 = load_speeds_task2(RAW_LOGS_TASK2)

years_in_data = sorted(df_actual['date'].dt.year.unique())
ALL_YEARS     = years_in_data + ([years_in_data[-1] + 1])
DEFAULT_YEAR  = years_in_data[-1]


# Добавление календарных признаков
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['year']     = out['date'].dt.year
    out['month']    = out['date'].dt.month
    out['day']      = out['date'].dt.day
    out['weekday']  = out['date'].dt.weekday
    out['day_of_year'] = out['date'].dt.dayofyear
    out['is_weekend']  = (out['weekday'] >= 5).astype(int)

    out = out.sort_values('date').reset_index(drop=True)
    out['campaign_days_elapsed']  = out.groupby('year').cumcount().add(1)
    out['campaign_days_remained'] = out.groupby('year')['campaign_days_elapsed'].transform('max') - out['campaign_days_elapsed']
    out['campaign_week']          = out['campaign_days_elapsed'] // 7 + 1
    out['sin_weekday'] = np.sin(2*np.pi*out['weekday']/7)
    out['cos_weekday'] = np.cos(2*np.pi*out['weekday']/7)
    return out

# Прогнозирование
lag_cols    = [f'y_lag_{lag}' for lag in LAGS]
static_cols = ['year','month','day','weekday','day_of_year','is_weekend',
               'campaign_days_elapsed','campaign_days_remained','campaign_week',
               'sin_weekday','cos_weekday'] + [f'ma_{w}' for w in WINDOWS]

def predict(model_kind: str, model, X_test: pd.DataFrame, init_win, task1: bool = True) -> np.ndarray:
    """Рекурсивный прогноз на len(X_test) шагов"""
    if task1:
        if model_kind == "SARIMA":
            pred = model.predict(n_periods=len(X_test))
            pred = np.array(pred)
            pred = np.where(pred < 0, 0, pred)
            return pred
            
        else:
            out, cw = [], list(init_win)
            for _, row in X_test.reset_index(drop=True).iterrows():
                feats = {
                    'year': row['year'], 'month': row['month'], 'day': row['day'],
                    'weekday': row['weekday'], 'day_of_year': row['day_of_year'],
                    'is_weekend': row['is_weekend'],
                    'campaign_days_elapsed': row['campaign_days_elapsed'],
                    'campaign_days_remained': row['campaign_days_remained'],
                    'sin_weekday': row['sin_weekday'], 'cos_weekday': row['cos_weekday']
                }
                for lag in LAGS:
                    feats[f'y_lag_{lag}'] = cw[-lag]

                if model_kind == "XGB":
                    feats['ma_3'] = np.mean(cw[-3:])
                    feats['ma_7'] = np.mean(cw[-7:])
                    X = pd.DataFrame([feats])

                    pred = model.named_steps['regressor'].predict(
                            model.named_steps['feature_selection'].transform(X))[0]

                else:  # LSTM
                    feats['campaign_week'] = feats['campaign_days_elapsed'] // 7 + 1
                    for w in WINDOWS:
                        feats[f'ma_{w}'] = np.mean(cw[-w:])
                    X = pd.DataFrame([feats])

                    seq   = X[lag_cols   ].values.reshape(1, len(lag_cols), 1)
                    stat  = X[static_cols].values
                    pred  = model.predict([seq, stat], verbose=0)[0,0]

                cw.pop(0)
                cw.append(pred)
                out.append(pred)
            
            pred = np.array(out)
    else:
        if model_kind == "ARIMAX":
            exog_cols = ['day_of_week', 'is_weekend', 'day_of_year', 'day', 'month', 'year', 'days_since_start', 'days_until_end']
            exog = X_test[exog_cols]
            pred = model.forecast(steps=len(X_test), exog=exog)
            pred = np.where(pred < 0, 0, pred)
            return pred
        out, cw = [], list(init_win)
        exog_cols = [c for c in X_test.columns if "lag" not in c and "ma" not in c and c != 'date']
        for _, row in X_test.reset_index(drop=True).iterrows():
            new_features = {c: row[c] for c in exog_cols}
            for lag in LAGS:
                new_features[f'lag_{lag}'] = float(cw[-lag])
            for window in WINDOWS:
                new_features[f'ma_{window}'] = float(np.mean(cw[-window:]))
            X = pd.DataFrame([new_features])
            if model_kind == "RF":
                pred = model.named_steps['regressor'].predict(model.named_steps['feature_selection'].transform(X))[0]
            else:
                lag_cols_task2 = [f"lag_{lag}" for lag in LAGS]
                seq = X[lag_cols_task2].values.reshape(1, len(lag_cols_task2), 1)
                stat_cols_task2 = [c for c in X.columns if c not in lag_cols_task2 + ['date']]
                stat = X[stat_cols_task2].values
                pred = model.predict([seq, stat], verbose=0)[0, 0]
            cw.pop(0)
            cw.append(pred)
            out.append(pred)

        pred = np.array(out)

    return pred

# UI
external = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/darkly/bootstrap.min.css"
]
app = dash.Dash(__name__, external_stylesheets=external, title='Приёмная кампания', suppress_callback_exceptions=True)

COLOR_BG = "#121212"
COLOR_TXT = "#eee"
COLOR_ACC = "#1f77b4"

TAB_STYLE = {"backgroundColor": COLOR_BG, "color": COLOR_TXT, "padding": "6px 12px", "border": "1px solid #333"}
TAB_SELECTED = TAB_STYLE | {"border": f"1px solid {COLOR_ACC}", "fontWeight": "bold"}

TEST_CONTROLS = [
    html.Br(),
    html.Label("Текущая дата"),
    dcc.DatePickerSingle(
        id="mock-date-picker",
        min_date_allowed=date(min(ALL_YEARS), 1, 1),
        max_date_allowed=date(max(ALL_YEARS), 12, 31),
        date=date.today(),
        className="small-date",
        display_format="DD.MM.YYYY",
        style={"marginBottom": "1rem"},
    ),
    html.Label("Показать фактические значения"),
    dcc.Checklist(
        id="show-future-fact",
        options=[{"label": "Да", "value": "show"}],
        value=[],
        inputStyle={"marginRight": 6},
        labelStyle={"display": "block"},
        style={"marginBottom": "1rem"},
    ),
    html.Hr(style={"margin": "6px 0"}),
]

def make_options(col):
    mapping = {0: 'Нет', 1: 'Да'}
    values = sorted(RAW_LOGS_TASK1[col].dropna().unique())
    return [{'label': mapping[v], 'value': int(v)} for v in values]

sidebar_submissions_stage1 = html.Div(
    style={'width': '250px', 'marginRight': '1.5rem'},

    children=[
        html.Div(TEST_CONTROLS, id="test-box-s1", style={"display": "none"}),

        html.Br(),
        html.Label('Год'),
        dcc.Dropdown(
            id='year-dd',
            options=[{'label': y, 'value': y} for y in ALL_YEARS],
            value=DEFAULT_YEAR,
            clearable=False,
            style={'marginBottom': '1rem', 'background': '#1e1e1e', 'color': '#fff'}
        ),

        # Аналитика
        html.Div(id='analytics-box', children=[
            html.Label('Параллельное зачисление'),
            dcc.Checklist(id='chk-zayav',  options=make_options('is_parallel'),
                          value=[], inputStyle={'marginRight': 6},
                          labelStyle={'display': 'block'},
                          style={'marginBottom': '1rem'}),

            html.Label('Иностранный гражданин'),
            dcc.Checklist(id='chk-foreign', options=make_options('is_interdekanat'),
                          value=[], inputStyle={'marginRight': 6},
                          labelStyle={'display': 'block'},
                          style={'marginBottom': '1rem'}),

            html.Label('Персонал'),
            dcc.Checklist(id='pk-cl', options=[], value=[],
                          inputStyle={'marginRight': 6},
                          labelStyle={'display': 'inline-block', 'width': '50%'},
                          style={'marginBottom': '1rem', 'color': '#fff'}),
        ]),

        # Прогноз
        html.Div( id='forecast-box', children=[
            html.Label('Модель'),
            dcc.Dropdown(id='model-dd',
                         options=[{'label': m, 'value': m}
                                  for m in ['LSTM', 'XGB', 'SARIMA']],
                         value='LSTM', clearable=False,
                         style={'marginBottom': '1rem',
                                'background': '#1e1e1e', 'color': '#fff'}),

            html.Label('Горизонт прогноза, дней'),
            dcc.Slider(id='horizon', min=1, max=MAX_HORIZON, value=MAX_HORIZON,
                       step=1,
                       marks={i: str(i) for i in range(0, MAX_HORIZON+1, 5)},
                       updatemode='drag')
        ])
    ]
)

sidebar_submissions_stage2 = html.Div(
    style={'width': '250px', 'marginRight': '1.5rem'},
    children=[
        html.Div(TEST_CONTROLS, id="test-box-s1", style={"display": "none"}),


        html.Br(),
        html.Label('Год'),
        dcc.Dropdown(
            id='year2-dd',
            options=[{'label': y, 'value': y} for y in ALL_YEARS],
            value=DEFAULT_YEAR,
            clearable=False,
            style={'marginBottom': '1rem', 'background': '#1e1e1e', 'color': '#fff'}
        ),

        # Аналитика
        html.Div(id='analytics2-box', children=[
            html.Label('Параллельное зачисление'),
            dcc.Checklist(id='chk-zayav2',  options=make_options('is_parallel'),
                          value=[], inputStyle={'marginRight': 6},
                          labelStyle={'display': 'block'},
                          style={'marginBottom': '1rem'}),

            html.Label('Иностранный гражданин'),
            dcc.Checklist(id='chk-foreign2', options=make_options('is_interdekanat'),
                          value=[], inputStyle={'marginRight': 6},
                          labelStyle={'display': 'block'},
                          style={'marginBottom': '1rem'}),

            html.Label('Персонал'),
            dcc.Checklist(id='pk2-cl', options=[], value=[],
                          inputStyle={'marginRight': 6},
                          labelStyle={'display': 'inline-block', 'width': '50%'},
                          style={'marginBottom': '1rem', 'color': '#fff'}),
        ]),

        # Прогноз
        html.Div(id='forecast2-box', children=[
            html.Label('Модель'),
            dcc.Dropdown(id='model2-dd',
                         options=[{'label': m, 'value': m}
                                  for m in ['RNN', 'RF', 'ARIMAX']],
                         value='RNN', clearable=False,
                         style={'marginBottom': '1rem',
                                'background': '#1e1e1e', 'color': '#fff'}),

            html.Label('Горизонт прогноза, дней'),
            dcc.Slider(id='horizon2', min=1, max=MAX_HORIZON_TASK2, value=MAX_HORIZON_TASK2,
                       step=1,
                       marks={i: str(i) for i in range(0, MAX_HORIZON_TASK2+1, 1)},
                       updatemode='drag')
        ])
    ]
)

sidebar_staff_stage1 = html.Div(
    style={'width':'250px','marginRight':'1.5rem'},
    children=[
        html.Div(TEST_CONTROLS, id="test-box-s1", style={"display": "none"}),


        html.Br(),
        html.Label('Год'),
        dcc.Dropdown(
            id='year-dd',
            options=[{'label': y, 'value': y} for y in ALL_YEARS],
            value=DEFAULT_YEAR,
            clearable=False,
            style={'marginBottom': '1rem', 'background': '#1e1e1e', 'color': '#fff'}
        ),

        dcc.Checklist(id='chk-zayav', value=[], style={'display': 'none'}),
        dcc.Checklist(id='chk-foreign', value=[], style={'display': 'none'}),
        dcc.Checklist(id='pk-cl', value=[], style={'display': 'none'}),

        html.Label('Показать среднюю нагрузку'),
        dcc.Checklist(
            id='show-load',
            options=[{'label': 'Да', 'value': 'show'}],
            value=[],
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block'},
            style={'marginBottom': '1rem'}
        ),

        html.Div(id='forecast-staff-box', children=[
            html.Label('Модель'),
            dcc.Dropdown(
                id='model-dd',
                options=[{'label': m, 'value': m} for m in ['LSTM', 'XGB', 'SARIMA']],
                value='LSTM',
                clearable=False,
                style={'marginBottom': '1rem',
                    'background': '#1e1e1e', 'color': '#fff'}
            ),
            html.Label('Горизонт прогноза, дней'),
            dcc.Slider(
                id='horizon-staff',
                min=1,
                max=MAX_HORIZON,
                value=MAX_HORIZON,
                step=1,
                marks={i: str(i) for i in range(0, MAX_HORIZON+1, 5)},
                updatemode='mouseup'
            )
        ], style={'display': 'none'}),
    ]
)

sidebar_staff_stage2 = html.Div(
    style={'width':'250px','marginRight':'1.5rem'},
    children=[
        html.Div(TEST_CONTROLS, id="test-box-s1", style={"display": "none"}),

        html.Br(),
        html.Label('Год'),
        dcc.Dropdown(
            id='year2-dd',
            options=[{'label': y, 'value': y} for y in ALL_YEARS],
            value=DEFAULT_YEAR,
            clearable=False,
            style={'marginBottom': '1rem', 'background': '#1e1e1e', 'color': '#fff'}
        ),

        dcc.Checklist(id='chk-zayav2', value=[], style={'display': 'none'}),
        dcc.Checklist(id='chk-foreign2', value=[], style={'display': 'none'}),
        dcc.Checklist(id='pk-cl2', value=[], style={'display': 'none'}),
        html.Div(id='analytics-box', style={'display': 'none'}),
        html.Div(id='forecast-box',  style={'display': 'none'}),

        html.Label('Показать среднюю нагрузку'),
        dcc.Checklist(
            id='show-load2',
            options=[{'label': 'Да', 'value': 'show'}],
            value=[],
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block'},
            style={'marginBottom': '1rem'}
        ),

        html.Div(id='forecast-staff2-box', children=[
            html.Label('Модель'),
            dcc.Dropdown(
                id='model2-dd',
                options=[{'label': m, 'value': m} for m in ['RNN', 'RF', 'ARIMAX']],
                value='RNN',
                clearable=False,
                style={'marginBottom': '1rem',
                    'background': '#1e1e1e', 'color': '#fff'}
            ),
            html.Label('Горизонт прогноза, дней'),
            dcc.Slider(
                id='horizon2-staff',
                min=1,
                max=MAX_HORIZON_TASK2,
                value=MAX_HORIZON_TASK2,
                step=1,
                marks={i: str(i) for i in range(0, MAX_HORIZON_TASK2+1, 1)},
                updatemode='mouseup'
            )
        ], style={'display': 'none'}),
    ]
)

app.layout = html.Div(
    style={
        'display': 'flex', 'height': '100vh',
        'background': COLOR_BG, 'color': COLOR_TXT,
        'padding': '1rem', 'flexDirection': "column"
    },
    children=[
        dcc.Tabs(id="main-tabs",
                 value="submissions_stage1",
                 style={"background": COLOR_BG},
                 children=[
                     dcc.Tab(label="Прием заявок (1 этап)", value="submissions_stage1", style=TAB_STYLE, selected_style=TAB_SELECTED),
                     dcc.Tab(label="Распределение персонала (1 этап)", value="staff_stage1", style=TAB_STYLE, selected_style=TAB_SELECTED),
                     dcc.Tab(label="Прием оригиналов (2 этап)", value='submissions_stage2', style=TAB_STYLE, selected_style=TAB_SELECTED),
                     dcc.Tab(label="Распределение персонала (2 этап)", value="staff_stage2", style=TAB_STYLE, selected_style=TAB_SELECTED)
                 ]),

        dcc.Store(id='mock-today', storage_type='memory'),

        # Заглушки
        dcc.Dropdown(id='year-dd', options=[], style={'display': 'none'}),
        dcc.Dropdown(id='model-dd', options=[], style={'display': 'none'}),
        dcc.Checklist(id='chk-zayav', options=[], style={'display': 'none'}),
        dcc.Checklist(id='chk-foreign', options=[], style={'display': 'none'}),
        dcc.Checklist(id='pk-cl', options=[], style={'display': 'none'}),
        html.Div(id='analytics-box', style={'display': 'none'}),
        html.Div(id='forecast-box', style={'display': 'none'}),
        html.Div(id="test-box-s1",      style={"display": "none"}),
        html.Div(id="test-box-s2",      style={"display": "none"}),
        html.Div(id="test-box-staff1",  style={"display": "none"}),
        html.Div(id="test-box-staff2",  style={"display": "none"}),
        dcc.Dropdown (id="year2-dd", options=[], style={"display": "none"}),
        
        dcc.Store(id="cache-submissions-stage1"),
        dcc.Store(id="cache-submissions-stage2"),
        dcc.Store(id="test-mode", data=False),
        html.Button("Test", id="test-btn", n_clicks=0,
                    style={
                        "position": "fixed",
                        "bottom": "20px",
                        "right": "20px",
                        "width": "56px",
                        "height": "56px",
                        "borderRadius": "50%",
                        "background": COLOR_ACC,
                        "color": "#fff",
                        "border": "none",
                        "fontWeight": "bold",
                        "fontSize": "16px",
                        "zIndex": 1000
                    }),
        html.Div(id="page-content", style={"flexGrow": 1, "display": "flex"})
    ]
)

# callbacks
def get_today(mock):
    return datetime.strptime(mock, "%Y-%m-%d").date() if mock else date.today()

def campaign_status(year: int, today: date, stage1: bool = True) -> str:
    if stage1:
        start = date(year, *DATA_START)
        end = date(year, *DATA_END)
    else:
        start = date(year, *TASK2_START)
        end = date(year, *TASK2_END.get(year, (8, 5)))
    if today > end:
        return "past"
    elif today < start:
        return "future"
    else:
        return "running"


@app.callback(
    Output("page-content", "children"),
    Input("main-tabs", "value")
)
def render_page(tab):
    if tab == "submissions_stage1" or not tab:
        return [sidebar_submissions_stage1,  dcc.Graph(id='graph1', style={"flexGrow": 1})]
    if tab == "staff_stage1":
        return [sidebar_staff_stage1, dcc.Graph(id='graph1-staff', style={"flexGrow": 1})]
    if tab == "submissions_stage2":
        return [sidebar_submissions_stage2, dcc.Graph(id='graph2', style={"flexGrow": 1})]
    if tab == "staff_stage2":
        return [sidebar_staff_stage2, dcc.Graph(id='graph2-staff', style={"flexGrow": 1})]

@app.callback(
    Output("test-mode",  "data"),
    Output("mock-today", "data"),
    Input("test-btn",          "n_clicks"),
    Input("mock-date-picker",  "date"),
    prevent_initial_call=True
)
def manage_test_and_date(n_clicks, picked_date):
    trg = dash.callback_context.triggered_id

    if trg == "test-btn":
        on = bool(n_clicks % 2)
        return on, None if not on else dash.no_update

    if trg == "mock-date-picker":
        return dash.no_update, picked_date

    raise PreventUpdate

@app.callback(
    Output("test-btn", "style"),
    Input("test-mode", "data"),
    prevent_initial_call=True
)
def recollor_button(on):
    style = {
        "position": "fixed",
        "bottom": "20px",
        "right": "20px",
        "width": "56px",
        "height": "56px",
        "borderRadius": "50%",
        "color": "#fff",
        "border": "none",
        "fontWeight": "bold",
        "fontSize": "16px",
        "zIndex": 1000
    }
    style["background"] = "#ff9800" if on else COLOR_ACC
    return style

@app.callback(
    Output("test-box-s1", "style"),
    Output("test-box-s2", "style"),
    Output("test-box-staff1", "style"),
    Output("test-box-staff2", "style"),
    Input("test-mode", "data")
)
def show_test_boxes(on):
    style = {} if on else {"display": "none"}
    return style, style, style, style

@app.callback(
    Output('pk-cl', 'options'),
    Input('year-dd', 'value')
)
def update_pk_checklist(year):
    if not year:
        raise PreventUpdate
    
    year = int(year)
    codes = RAW_LOGS_TASK1.loc[RAW_LOGS_TASK1['y'] == year, 'pk_user_kod'].dropna().unique()
    opts = [{'label': str(pk)[:-2], 'value': int(pk)} for pk in sorted(codes)]
    return opts

@app.callback(
    Output("pk2-cl", "options"),
    Input("year2-dd", "value")
)
def update_pk_checklist2(year):
    if not year:
        raise PreventUpdate
    
    year = int(year)
    codes = RAW_LOGS_TASK2.loc[RAW_LOGS_TASK2["date"].dt.year == year, "pk_user_kod"].dropna().unique()
    opts = [{"label": str(pk)[:-2], "value": int(pk)} for pk in sorted(codes)]
    return opts

@app.callback(
    Output('analytics-box', 'style'),
    Output('forecast-box', 'style'),
    Output('chk-zayav', 'value'),
    Output('chk-foreign', 'value'),
    Output('pk-cl', 'value'),
    Input('year-dd', 'value'),
    Input('main-tabs', 'value'),
    Input('mock-today', 'data')
)
def switch_sidebar_blocks(selected_year, active_tab, mock_today):
    if active_tab != "submissions_stage1" or not selected_year:
        raise PreventUpdate
    
    y = int(selected_year)
    today = get_today(mock_today)
    campaign_end = datetime(y, *DATA_END).date()
    
    past = y < today.year or (y == today.year and today > campaign_end)


    hide = {'display': 'none'}
    show = {}

    if past:
        return show, hide, [], [], []
    else:
        return hide, show, [], [], []
    
@app.callback(
    Output('analytics2-box', 'style'),
    Output('forecast2-box', 'style'),
    Output('chk-zayav2', 'value'),
    Output('chk-foreign2', 'value'),
    Output('pk2-cl', 'value'),
    Input('year2-dd', 'value'),
    Input('main-tabs', 'value'),
    Input('mock-today', 'data')
)
def switch_sidebar2_blocks(selected_year, active_tab, mock_today):
    if active_tab != "submissions_stage2" or not selected_year:
        raise PreventUpdate
    
    y = int(selected_year)
    today = get_today(mock_today)
    campaign_end = datetime(y, *TASK2_END.get(y, (8, 5))).date()
    
    past = y < today.year or (y == today.year and today > campaign_end)


    hide = {'display': 'none'}
    show = {}

    if past:
        return show, hide, [], [], []
    else:
        return hide, show, [], [], []
    
@app.callback(
    Output('forecast-staff-box', 'style'),
    Input('year-dd', 'value'),
    Input('mock-today', 'data'),
    Input('main-tabs', 'value'),
    prevent_initial_call=True
)
def toggle_staff_controls(year, mock_today, active_tab):
    if active_tab != 'staff_stage1' or not year:
        raise PreventUpdate
    today = get_today(mock_today)
    end = date(year, *DATA_END)

    if today > end:
        return {'display': 'none'}
    return {}

@app.callback(
    Output('forecast-staff2-box', 'style'),
    Input('year2-dd', 'value'),
    Input('mock-today', 'data'),
    Input('main-tabs', 'value'),
    prevent_initial_call=True
)
def toggle_staff2_controls(year, mock_today, active_tab):
    if active_tab != 'staff_stage2' or not year:
        raise PreventUpdate
    today = get_today(mock_today)
    end = date(year, *TASK2_END.get(year, (8, 5)))

    if today > end:
        return {'display': 'none'}
    return {}
    
@app.callback(
    Output('horizon', 'max'),
    Output('horizon', 'marks'),
    Output('horizon', 'value'),
    Input('year-dd', 'value'),
    Input('mock-today', 'data'),
    Input('horizon', 'value'),
    prevent_initial_call=True
)
def limit_horizon_max(selected_year, mock_today, curr_val):
    y = selected_year or DEFAULT_YEAR
    today = get_today(mock_today)

    start = date(y, *DATA_START)
    end = date(y, *DATA_END)

    if today >= end:
        m = 1
    elif today < start:
        m = (end - start).days
    else:
        m = (end - today).days

    marks = {i: str(i) for i in range(0, m+1, 5)}
    value = min(curr_val, m)

    return m, marks, value

@app.callback(
    Output('horizon2', 'max'),
    Output('horizon2', 'marks'),
    Output('horizon2', 'value'),
    Input('year2-dd', 'value'),
    Input('mock-today', 'data'),
    Input('horizon2', 'value'),
    prevent_initial_call=True
)
def limit_horizon2_max(selected_year, mock_today, curr_val):
    y = selected_year or DEFAULT_YEAR
    today = get_today(mock_today)

    start = date(y, *TASK2_START)
    end = date(y, *TASK2_END.get(y, (8, 5)))

    if today >= end:
        m = 1
    elif today < start:
        m = (end - start).days
    else:
        m = (end - today).days

    marks = {i: str(i) for i in range(0, m+1, 1)}
    value = curr_val if curr_val <= m else m

    return m, marks, value

@app.callback(
    Output('cache-submissions-stage1', 'data'),
    Input('year-dd', 'value'),
    Input('chk-zayav', 'value'),
    Input('chk-foreign', 'value'),
    Input('pk-cl', 'value'),
    Input('mock-today', 'data'),
    Input('show-future-fact', 'value'),
    Input('test-mode', 'data'),
    prevent_initial_call=True
)
def build_cache_stage1(year, flt_parallel, flt_foreign, flt_pk, mock_today, future_fact, test_mode):
    if not year:
        year = DEFAULT_YEAR
    data_start = pd.Timestamp(year, *DATA_START)
    data_end   = pd.Timestamp(year, *DATA_END)

    today = get_today(mock_today)
    fact_end = min(today, data_end.date())
    full_range = pd.date_range(data_start, fact_end)

    fact_full = RAW_LOGS_TASK1[(RAW_LOGS_TASK1['date'] >= data_start) &
                (RAW_LOGS_TASK1['date'] <= pd.Timestamp(data_end)) & 
                (RAW_LOGS_TASK1['y'] == year)]

    fact = fact_full[fact_full['date'] <= pd.Timestamp(today)]

    past_year = year < today.year
    if past_year:
        if flt_pk:
            fact = fact[fact['pk_user_kod'].isin(flt_pk)]
        if flt_parallel:
            fact = fact[fact['is_parallel'].isin(flt_parallel)]
        if flt_foreign:
            fact = fact[fact['is_interdekanat'].isin(flt_foreign)]

    campaign_running = (year == today.year) and (today <= data_end.date())
    future_year = year > today.year
    
    fact_daily = fact.groupby('date').size().reindex(full_range, fill_value=0)

    if fact_daily.any():
        last_fact_date = fact_daily.index[fact_daily.gt(0)].max()
    else:
        last_fact_date = data_start - timedelta(days=1)
    
    cache = {
        'year': year,
        'fact_dates': fact_daily.index.strftime('%Y-%m-%d').tolist(),
        'fact_y': fact_daily.values.tolist(),
        'last_fact': last_fact_date.strftime('%Y-%m-%d'),
        'future_fact_dates': fact_full[fact_full['date'] > pd.Timestamp(today)].groupby('date').size().index.strftime("%Y-%m-%d").tolist(),
        'future_fact_y': fact_full[fact_full['date'] > pd.Timestamp(today)].groupby('date').size().tolist()
    }

    for model_key, model in MODELS_TASK1.items():
        need_forecast = campaign_running or future_year

        if need_forecast:
            window = df_actual['y'].tail(max(LAGS)).tolist()
            future_dates = pd.date_range(start=last_fact_date + timedelta(days=1),
                                        periods=MAX_HORIZON, freq='D')
            X_test = add_calendar(pd.DataFrame({'date': future_dates}))
            mean = predict(model_key, model, X_test, window)
            dates = future_dates.strftime('%Y-%m-%d').tolist()

            max_pred = max(mean) if mean.size else 0
            min_pred = min(mean) if mean.size else 0
            
            cache[model_key] = dict(
                dates=dates,
                mean=mean,
                y_max_pred=max_pred,
                y_min_pred=min_pred
            )
        else:
            cache[model_key] = dict(dates=[], mean=[], y_max_pred=0, y_min_pred=0)

    return cache

@app.callback(
    Output('cache-submissions-stage2', 'data'),
    Input('main-tabs', 'value'),
    Input('year2-dd', 'value'),
    Input('chk-zayav2', 'value'),
    Input('chk-foreign2', 'value'),
    Input('pk2-cl', 'value'),
    Input('mock-today', 'data'),
    Input('show-future-fact', 'value'),
    prevent_initial_call=True
)
def build_cache_stage2(tab, year, flt_parallel, flt_foreign, flt_pk, mock_today, future_fact):
    # if tab != 'submissions_stage2' and tab != 'staff_stage2':
    #     raise PreventUpdate
    year = year or DEFAULT_YEAR
    start = pd.Timestamp(year, *TASK2_START)
    end = pd.Timestamp(year, *TASK2_END.get(year, (8, 5))) # Если год не внесен в TASK2_END, 05.08 – значение по умолчанию

    today = get_today(mock_today)
    fact_end = pd.Timestamp(min(today, end.date()))
    idx_fact = pd.date_range(start, fact_end)
    idx_full = pd.date_range(start, end)

    logs_full = RAW_LOGS_TASK2[
        (RAW_LOGS_TASK2["date"] >= start) &
        (RAW_LOGS_TASK2["date"] <= end) &
        (RAW_LOGS_TASK2["date"].dt.year == year)
    ].copy()

    logs = logs_full[logs_full['date'] <= pd.Timestamp(today)]

    past_year = year < today.year
    if past_year:
        if flt_parallel:
            logs = logs[logs["is_parallel"].isin(flt_parallel)]
        if flt_foreign:
            logs = logs[logs["is_interdekanat"].isin(flt_foreign)]
        if flt_pk:
            logs = logs[logs["pk_user_kod"].isin(flt_pk)]

    logs["date_day"] = logs["date"].dt.floor("D")
    logs_full["date_day"] = logs_full["date"].dt.floor("D")

    daily = logs.groupby(["date_day", "transition"]).size().unstack(fill_value=0).rename_axis(None, axis=1)
    daily_full = logs_full.groupby(["date_day", "transition"]).size().unstack(fill_value=0).rename_axis(None, axis=1)
    
    for col in ("in", "out"):
        if col not in daily.columns:
            daily[col] = 0
        if col not in daily_full.columns:
            daily_full[col] = 0

    daily = daily[["in", "out"]].sort_index()
    daily_full = daily_full[["in", "out"]].sort_index()
    daily["all"] = daily["in"] + daily["out"]
    daily_full["all"] = daily_full["in"] + daily_full["out"]

    fact = daily.reindex(idx_fact, fill_value=0)
    fact_full = daily_full.reindex(idx_full, fill_value=0)

    future_part = fact_full.loc[fact_full.index > pd.Timestamp(today)]
    fut_dates = future_part.index.strftime('%Y-%m-%d').tolist()
    fut_in = future_part['in'].tolist() if not future_part.empty else []
    fut_out = future_part['out'].tolist() if not future_part.empty else []
    fut_all = future_part['all'].tolist() if not future_part.empty else []

    if pd.isna(fact.index.max()):
        last_fact = start - timedelta(days=1)
    else:
        last_fact = fact.index.max()

    cache = dict(
        year=year,
        fact_dates=fact.index.strftime('%Y-%m-%d').tolist(),
        fact_in=fact['in'].tolist(),
        fact_out=fact['out'].tolist(),
        fact_all=fact['all'].tolist(),
        last_fact=last_fact.strftime('%Y-%m-%d'),
        future_fact_dates=fut_dates,
        future_fact_in=fut_in,
        future_fact_out=fut_out,
        future_fact_all=fut_all,
    )

    need_forecast = today <= end.date()
    if need_forecast:
        future_idx = pd.date_range(last_fact + timedelta(days=1), periods=MAX_HORIZON_TASK2, freq='D')
        future_dates = future_idx.strftime('%Y-%m-%d').tolist()

        X_test = pd.DataFrame({'date': future_idx})
        X_test['day_of_week'] = X_test['date'].dt.dayofweek
        X_test['is_weekend'] = X_test['day_of_week'].isin([5, 6]).astype(int)
        X_test['day_of_year'] = X_test['date'].dt.dayofyear
        X_test['day'] = X_test['date'].dt.day
        X_test['month'] = X_test['date'].dt.month
        X_test['year'] = X_test['date'].dt.year
        X_test['days_since_start'] = (X_test['date'] - start).dt.days + 1
        X_test['days_until_end'] = (end - X_test['date']).dt.days + 1
        X_test['sin_weekday'] = np.sin(2 * np.pi * X_test['day_of_week'] / 7)
        X_test['cos_weekday'] = np.cos(2 * np.pi * X_test['day_of_week'] / 7)

        for model_type in ["RNN", "RF", "ARIMAX"]:

            init_win_in = df_task2[df_task2['date'] <= fact_end]['in'].tail(max(LAGS)).tolist()
            in_pred = predict(model_type, MODELS_TASK2[f'{model_type}_IN'], X_test, init_win_in, task1=False)

            init_win_out = df_task2[df_task2['date'] <= fact_end]['out'].tail(max(LAGS)).tolist()
            out_pred = predict(model_type, MODELS_TASK2[f'{model_type}_OUT'], X_test, init_win_out, task1=False)

            all_pred = in_pred + out_pred
            max_pred = max(all_pred) if all_pred.size else 0
            min_pred = min(min(in_pred), min(out_pred)) if in_pred.size or out_pred.size else 0

            cache[model_type] = dict(
                dates=future_dates,
                mean_in=in_pred.tolist(),
                mean_out=out_pred.tolist(),
                mean_all=all_pred.tolist(),
                y_max_pred=max_pred,
                y_min_pred=min_pred
            )
    else:
        empty_pred = dict(dates=[], mean_in=[], mean_out=[], mean_all=[], y_max_pred=0, y_min_pred=0)
        for model_type in ["RNN", "RF", "ARIMAX"]:
            cache[model_type] = empty_pred.copy()

    return cache

@app.callback(
    Output("graph1", "figure"),
    Input("cache-submissions-stage1", "data"),
    Input("horizon", "value"),
    Input("model-dd", "value"),
    Input("show-future-fact", "value"),
    Input("test-mode", "data")
)
def update_plot_stage1(cache, horizon, model_key, show_chk, test_mode):
    if not cache:
        return go.Figure()

    bloc = cache[model_key]
    horizon = int(horizon)

    dates   = pd.to_datetime(bloc["dates"][:horizon])
    mean    = np.array(bloc["mean"][:horizon])

    fact_dates = pd.to_datetime(cache["fact_dates"])
    fact_y     = cache["fact_y"]
    last_fact  = pd.to_datetime(cache["last_fact"])
    y_max_pred = bloc['y_max_pred']
    y_min_pred = bloc['y_min_pred']

    fig = go.Figure()

    if len(fact_dates):
        fig.add_trace(go.Scatter(x=fact_dates, y=fact_y,
                                 mode='lines+markers', name='Факт', showlegend=False,
                                 line=dict(color=COLOR_MAIN, width=2)))

    if len(mean) and fact_dates.size:
            fig.add_trace(go.Scatter(x=[last_fact, dates[0]],
                                    y=[fact_y[-1] if fact_y else mean[0], mean[0]],
                                    mode='lines', showlegend=False, hoverinfo='skip',
                                    line=dict(color=COLOR_MAIN, dash='dash')))
    
    if mean.size:
        fig.add_trace(go.Scatter(x=dates, y=mean,
                                 mode='lines+markers', name='Прогноз', showlegend=False,
                                 line=dict(color=COLOR_MAIN, dash='dash'), hovertemplate="%{y:.0f}"))
        
        if test_mode and "show" in (show_chk or []) and cache.get("future_fact_dates"):
            fut_x = pd.to_datetime(cache["future_fact_dates"])[:horizon]
            fut_y = cache["future_fact_y"][:horizon]

            if len(fact_dates) and len(fut_x):
                fig.add_scatter(
                    x=[last_fact, fut_x[0]],
                    y=[fact_y[-1] if fact_y else fut_y[0], fut_y[0]],
                    mode='lines',
                    hoverinfo="skip",
                    line=dict(color=COLOR_MAIN),
                    opacity=0.6,
                    showlegend=False
                )
            
            fig.add_scatter(
                x=fut_x,
                y=fut_y,
                mode="lines+markers",
                name="Фактические данные",
                line=dict(color=COLOR_MAIN, width=1),
                opacity=0.6,
                showlegend=False,
                hovertemplate="%{y:.0f}"
            )

    year = cache["year"]
    axis_start = datetime(year, *X_AXIS_START)
    axis_end   = datetime(year, *X_AXIS_END)
    first_mon  = axis_start + timedelta(days=(7-axis_start.weekday()) % 7)

    extra_fact = cache["future_fact_y"] if (test_mode and "show" in (show_chk or [])) else []
    all_vals = list(fact_y) + list(mean) + list(extra_fact)

    y_max = max(max(all_vals) if all_vals else 0, y_max_pred) + 20
    y_min = max(0, min(min(all_vals) if all_vals else 0, y_min_pred) - 20)

    fig.update_xaxes(range=[axis_start, axis_end], dtick=WEEK_MS, tick0=first_mon,
                     tickformat='%d.%m', showgrid=True, gridcolor='#444')
    fig.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor='#444')
    fig.update_layout(template='plotly_dark', paper_bgcolor='#121212',
                      plot_bgcolor='#121212',
                      margin=dict(l=20, r=20, t=40, b=20), hovermode='x unified')
    return fig

@app.callback(
    Output("graph2", "figure"),
    Input("cache-submissions-stage2", "data"),
    Input("horizon2", "value"),
    Input("model2-dd", "value"),
    Input("show-future-fact", "value"),
    Input("test-mode", "data")
)
def update_plot_stage2(cache, horizon, model_key, show_chk, test_mode):
    if not cache:
        return go.Figure()
    
    bloc = cache[model_key]
    horizon = int(horizon)

    idx_pred = pd.to_datetime(bloc['dates'][:horizon])
    in_pred = np.array(bloc['mean_in'][:horizon])
    out_pred = np.array(bloc['mean_out'][:horizon])
    all_pred = np.array(bloc['mean_all'][:horizon])

    fact_idx = pd.to_datetime(cache['fact_dates'])
    fact_in = cache['fact_in']
    fact_out = cache['fact_out']
    fact_all = cache['fact_all']

    fig = go.Figure()
    if len(fact_idx):
        fig.add_trace(go.Scatter(x=fact_idx, y=fact_in, mode='lines+markers', name='Подано', line=dict(color='#2ecc71'), hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=fact_idx, y=fact_out, mode='lines+markers', name='Забрано', line=dict(color='#d62728'), hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=fact_idx, y=fact_all, mode='lines+markers', name='Всего', line=dict(color='#1f77b4'), hovertemplate="%{y:.0f}"))

    if idx_pred.size:
        last_fact_date = fact_idx.max()
        if len(fact_idx) and len(idx_pred):
            fig.add_scatter(x=[last_fact_date, idx_pred[0]], y=[fact_in[-1], in_pred[0]], mode="lines", line=dict(color='#2ecc71', dash='dash'), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=[last_fact_date, idx_pred[0]], y=[fact_out[-1], out_pred[0]], mode="lines", line=dict(color='#d62728', dash='dash'), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=[last_fact_date, idx_pred[0]], y=[fact_all[-1], all_pred[0]], mode="lines", line=dict(color='#1f77b4', dash='dash'), hoverinfo="skip", showlegend=False)

        fig.add_trace(go.Scatter(x=idx_pred, y=in_pred, mode='lines+markers', name='Подано', line=dict(color='#2ecc71', dash='dash'), hovertemplate="%{y:.0f}", showlegend=False))
        fig.add_trace(go.Scatter(x=idx_pred, y=out_pred, mode='lines+markers', name='Забрано', line=dict(color='#d62728', dash='dash'), hovertemplate="%{y:.0f}", showlegend=False))
        fig.add_trace(go.Scatter(x=idx_pred, y=all_pred, mode='lines+markers', name='Всего', line=dict(color='#1f77b4', dash='dash'), hovertemplate="%{y:.0f}", showlegend=False))

        if test_mode and "show" in (show_chk or []) and cache.get("future_fact_dates"):
            fut_x   = pd.to_datetime(cache["future_fact_dates"])[:horizon]
            fut_in  = cache["future_fact_in"] [:horizon]
            fut_out = cache["future_fact_out"][:horizon]
            fut_all = cache["future_fact_all"][:horizon]

            if len(fact_idx) and len(fut_x):
                fig.add_scatter(x=[fact_idx.max(), fut_x[0]], y=[fact_in[-1], fut_in[0]],
                                mode="lines", line=dict(color="#2ecc71"), opacity=.6,
                                hoverinfo="skip", showlegend=False)
                fig.add_scatter(x=[fact_idx.max(), fut_x[0]], y=[fact_out[-1], fut_out[0]],
                                mode="lines", line=dict(color="#d62728"), opacity=.6,
                                hoverinfo="skip", showlegend=False)
                fig.add_scatter(x=[fact_idx.max(), fut_x[0]], y=[fact_all[-1], fut_all[0]],
                                mode="lines", line=dict(color="#1f77b4"), opacity=.6,
                                hoverinfo="skip", showlegend=False)

            fig.add_scatter(x=fut_x, y=fut_in,  mode="lines+markers",
                            line=dict(color="#2ecc71"), opacity=.6,
                            name="Подано (факт)",   showlegend=False, hovertemplate="%{y:.0f}")
            fig.add_scatter(x=fut_x, y=fut_out, mode="lines+markers",
                            line=dict(color="#d62728"), opacity=.6,
                            name="Забрано (факт)",  showlegend=False, hovertemplate="%{y:.0f}")
            fig.add_scatter(x=fut_x, y=fut_all, mode="lines+markers",
                            line=dict(color="#1f77b4"), opacity=.6,
                            name="Всего (факт)",    showlegend=False, hovertemplate="%{y:.0f}")
            
    year = cache["year"]
    axis_start = datetime(year, *TASK2_START)
    axis_end = datetime(year, *TASK2_END.get(year, (8, 5)))

    fig.update_xaxes(
        range=[axis_start, axis_end],
        tickformat="%d.%m",
        showgrid=True,
        gridcolor="#444"
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    return fig

@app.callback(
    Output("graph1-staff", "figure"),
    Input("cache-submissions-stage1", "data"),
    Input("horizon-staff", "value"),
    Input("show-load", "value"),
    Input("main-tabs", "value"),
    Input("model-dd", "value"),
    Input("year-dd", "value"),
    Input("mock-today", "data")
)
def update_staff_graph(cache, horizon, show_load, active_tab, model_key, year, mock_today):
    if active_tab != "staff_stage1" or not cache:
        return go.Figure()
    
    today = get_today(mock_today)
    status = campaign_status(year, today)

    campaign_dates = pd.date_range(datetime(year, *DATA_START), datetime(year, *DATA_END))

    staff_fact = (RAW_LOGS_TASK1[RAW_LOGS_TASK1["y"] == year]
                  .groupby("date")["pk_user_kod"]
                  .nunique()
                  .reindex(campaign_dates, fill_value=0)
                  .astype(int))

    fact_series = pd.Series(0, index=campaign_dates, dtype=float)
    if cache["fact_dates"]:
        fact_df = pd.Series(cache["fact_y"],
                            index=pd.to_datetime(cache["fact_dates"]))
        fact_series.update(fact_df)

    lambdas_series = fact_series.copy()
    if status in ("future", "running"):
        bloc       = cache[model_key]
        pred_idx   = pd.to_datetime(bloc["dates"])
        lambdas_series.update(pd.Series(bloc["mean"], index=pred_idx))

        staff_pred = optimize_staff(bloc["mean"], SPEEDS, target1=0.80, limit1=1, target2=0.95, limit2=2, random_state=42)
        staff_pred = pd.Series(staff_pred, index=pred_idx)
        staff_fact.update(staff_pred)

    staff_plan = staff_fact
    lambdas_full = lambdas_series

    horizon     = int(horizon)
    staff_plot  = staff_plan.iloc[:horizon]
    dates_plot  = staff_plot.index

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates_plot,
        y=staff_plot,
        name="Число сотрудников",
        marker_color=COLOR_ACC,
        showlegend=False
    ))

    if "show" in (show_load or []):
        load = np.divide(lambdas_full[:horizon], staff_plot.values, out=np.zeros_like(staff_plot, dtype=float), where=staff_plot.values>0)
        y2_min, y2_max = load.min(), load.max()

        fig.add_scatter(
            x=dates_plot,
            y=load,
            mode="lines+markers",
            name="Средняя нагрузка, заявок/чел",
            yaxis="y2",
            line=dict(color="#2ecc71"),
            showlegend=False,
            hovertemplate="%{y:.2f}"
        )

        fig.update_layout(
            yaxis2=dict(
                title="Заявок на 1 сотрудника",
                range=[y2_min - 5, y2_max + 5],
                autorange=False,
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    axis_start = datetime(year, *X_AXIS_START)
    axis_end = datetime(year, *X_AXIS_END)
    first_mon = axis_start + timedelta(days=(7 - axis_start.weekday()) % 7)

    y_max = staff_plan.max() + 1

    fig.update_xaxes(
        range=[axis_start, axis_end],
        tick0=first_mon,
        tickformat="%d.%m",
        showgrid=True,
        gridcolor="#444"
    )
    fig.update_yaxes(
        range=[0, y_max],
        title_text="Число сотрудников",
        showgrid=True,
        gridcolor="#444"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )

    return fig

@app.callback(
    Output("graph2-staff", "figure"),
    Input("cache-submissions-stage2", "data"),
    Input("horizon2-staff", "value"),
    Input("show-load2", "value"),
    Input("main-tabs", "value"),
    Input("model2-dd", "value"),
    Input("year2-dd", "value"),
    Input("mock-today", "data")
)
def update_staff_graph2(cache, horizon, show_load, active_tab, model_key, year, mock_today):
    if active_tab != "staff_stage2" or not cache:
        return go.Figure()
    
    today = get_today(mock_today)
    status = campaign_status(year, today, stage1=False)

    start = pd.Timestamp(year, *TASK2_START)
    end = pd.Timestamp(year, *TASK2_END.get(year, (8, 5)))

    campaign_dates = pd.date_range(start, end)

    staff_fact = (
        RAW_LOGS_TASK2[RAW_LOGS_TASK2['date'].dt.year == year]
        .groupby('date')['pk_user_kod']
        .nunique()
        .reindex(campaign_dates[campaign_dates <= pd.Timestamp(today)], fill_value=0)
        .astype(int)
    )

    fact_series = pd.Series(0, index=campaign_dates, dtype=float)
    if cache["fact_dates"]:
        fact_all = pd.Series(cache['fact_all'], index=pd.to_datetime(cache['fact_dates']))
        fact_series.update(fact_all)

    staff_pred = pd.Series(dtype=int, index=pd.DatetimeIndex([]))
    lambdas_series = fact_series.copy()
    if status in ("future", "running"):
        bloc = cache[model_key]
        pred_idx = pd.to_datetime(bloc["dates"])
        pred_all = np.array(bloc["mean_all"])
        lambdas_series.update(pd.Series(pred_all, index=pred_idx))

        staff_pred = optimize_staff(pred_all, SPEEDS_TASK2, target1=0.95, limit1=1, target2=0.95, limit2=1, random_state=42)
        staff_pred = pd.Series(staff_pred, index=pred_idx)
        #staff_fact.update(staff_pred)

    actual_staff  = staff_fact.copy()
    forecast_mask = staff_pred.index.difference(actual_staff.index)
    forecast      = staff_pred.loc[forecast_mask]



    staff_plan = staff_fact
    lambdas_full = lambdas_series

    horizon = int(horizon)
    staff_plot = staff_plan.iloc[:horizon]
    dates_plot = staff_plot.index

    fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x=dates_plot,
    #     y=staff_plot,
    #     name="Число сотрудников",
    #     marker_color=COLOR_ACC,
    #     showlegend=False
    # ))

    fig.add_bar(x=actual_staff.index,  y=actual_staff,
                marker_color=COLOR_ACC, name='Факт',  showlegend=False)

    fig.add_bar(x=forecast.index, y=forecast,
                marker_color="rgba(31,119,180,0.5)",
                marker_pattern_shape='\\',
                marker_pattern_fgcolor=COLOR_ACC,
                marker_line_color=COLOR_ACC,
                marker_line_width=1,    
                name='План', showlegend=False)

    if "show" in (show_load or []):
        load = np.divide(lambdas_full[:horizon], staff_plot.values, out=np.zeros_like(staff_plot, dtype=float), where=staff_plot.values>0)
        y2_min, y2_max = load.min(), load.max()

        fig.add_scatter(
            x=dates_plot,
            y=load,
            mode="lines+markers",
            name="Средняя нагрузка, заявок/чел",
            yaxis="y2",
            line=dict(color="#2ecc71"),
            showlegend=False,
            hovertemplate="%{y:.2f}"
        )

        fig.update_layout(
            yaxis2=dict(
                title="Операций на 1 сотрудника",
                range=[y2_min - 5, y2_max + 5],
                autorange=False,
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    y_max = max(staff_plan.max(), forecast.max() if not forecast.empty else 0) + 1

    fig.update_xaxes(
        tickformat="%d.%m",
        showgrid=True,
        gridcolor="#444"
    )

    fig.update_xaxes(
        tickformat="%d.%m",
        showgrid=True,
        gridcolor="#444"
    )
    fig.update_yaxes(
        range=[0, y_max],
        title_text="Число сотрудников",
        showgrid=True,
        gridcolor="#444"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )

    return fig


if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8050')).start()
    app.run(port=8050)