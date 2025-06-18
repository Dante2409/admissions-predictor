from datetime import date, datetime
from typing import Tuple

import numpy as np
import pandas as pd


def load_speeds(csv_path: str) -> np.ndarray:
    """Загружает массив средних дневных скоростей персонала за предыдущую приемную кампанию

    Args:
        csv_path (str): Путь к файлу "first_mention.csv"

    Returns:
        np.ndarray: Массив скоростей
    """
    df = pd.read_csv(csv_path, parse_dates=["reg_ts"], low_memory=False)
    
    # Определение года ближайшей кампании
    curr_date = datetime.today().date()
    curr_year = datetime.today().year
    target_year = (curr_year - 1) if curr_date <= date(curr_year, 7, 25) else curr_year
    df = df[df['y'] == target_year]

    # Расчет средней дневной продуктивности сотрудника
    df["reg_ts_day"] = pd.to_datetime(df["reg_ts"], errors='coerce').dt.floor("D")
    daily_prod = df.groupby(["pk_user_kod", "reg_ts_day"]).size().reset_index(name="apps_count")
    daily_summary = daily_prod.groupby("pk_user_kod")["apps_count"].agg(["mean", "sum", "count"])
    daily_summary = daily_summary[daily_summary['sum'] >= 100] # фильтрация по количеству обслуженных заявок
    daily_summary.rename(columns={"mean": "avg_per_day"}, inplace=True)
    speeds = daily_summary["avg_per_day"].values
    return speeds

def simulate_campaign(
    lambdas: np.ndarray,
    staff: np.ndarray,
    speeds: np.ndarray,
    target1: float = 0.80,
    backlog_limit_days1: int = 1,
    target2: float = 0.95,
    backlog_limit_days2: int = 2,
    reps: int = 100,
    random_state: int | None = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Симуляция приемной кампании

    Args:
        lambdas (np.ndarray): Прогнозные значения заявок на день t
        staff (np.ndarray): Планируемое количество сотрудников на день t
        speeds (np.ndarray): Эмпирическое распределение средних дневных скоростей сотрудников
        target1 (float, optional): Первый целевой процент обслуживания заявок. Defaults to 0.80.
        backlog_limit_days1 (int, optional): Максимум срока заявки в днях для первого целевого уровня обслуживания. Defaults to 1.
        target2 (float, optional): Второй целевой процент обслуживания заявок. Defaults to 0.95.
        backlog_limit_days2 (int, optional): Максимум срока заявки в днях для второго целевого уровня обслуживания. Defaults to 2.
        reps (int, optional): Количество повторений симуляции. Defaults to 100.
        random_state (int | None, optional): Seed для воспроизводимости. Defaults to None.

    Returns:
        success1 (np.ndarray) – доля заявок, обработанных за первый целевой максимум срока заявки
        success2 (np.ndarray) – доля заявок, обработанных за второй целевой максимум срока заявки
        gap1 (float) – средний зазор между реальным уровнем сервиса и первым целевым
        gap2 (float) – средниз зазор между реальным уровнем сервиса и вторым целевым

    """
    rng = np.random.default_rng(random_state)

    T = len(lambdas)
    # Глобальные счётчики (для всех реализаций кампании)
    total_arr = np.zeros(T) # Сумма всех поступивших заявок
    served1 = np.zeros(T) # Сумма заявок, обработанных вовремя для первого таргета
    served2 = np.zeros(T) # Сумма заявок, обработанных вовремя для второго таргета

    # Каждая итерация – независимая реализация кампании
    for _ in range(reps):
        backlog = [] # FIFO: [age, qty] – [кол-во дней заявке, кол-во необслуженных]

        # Локальные счётчики (для одной кампании)
        daily_arr = np.zeros(T) # Пришло в день
        daily_served1 = np.zeros(T) # Обработано в день в рамках первого лимита
        daily_served2 = np.zeros(T) # Обработано в день в рамках второго лимита

        # Для каждого дня приемной кампании
        for t in range(T):
            # Приход заявок
            a = rng.poisson(lambdas[t])
            daily_arr[t] += a
            backlog.append([0, a])

            capacity = rng.choice(speeds, size=staff[t], replace=True).sum() # Сколько заявок можно обработать в этот день

            # Обслуживание очереди
            while backlog and capacity > 0:
                age, qty = backlog[0]
                take = min(qty, capacity)
                capacity -= take

                # Если успели обработать заявку вовремя, то записываем в день, когда она пришла
                if age <= backlog_limit_days1:
                    daily_served1[t - age] += take
                if age <= backlog_limit_days2:
                    daily_served2[t - age] += take

                if qty == take: # если обработали полностью
                    backlog.pop(0)
                else: # если обработали частично
                    backlog[0][1] -= take

            for rec in backlog: # старение заявок
                rec[0] += 1

        total_arr += daily_arr
        served1 += daily_served1
        served2 += daily_served2

    success1 = np.divide(served1, total_arr, out=np.ones_like(served1), where=total_arr > 0) # если в день t не было заявок, то значение по умолчанию = 100%
    success2 = np.divide(served2, total_arr, out=np.ones_like(served2), where=total_arr > 0)

    gap1 = np.mean(success1 - target1)
    gap2 = np.mean(success2 - target2)
    return success1, success2, gap1, gap2

def optimize_staff_bottomup(
    lambdas: np.ndarray,
    speeds: np.ndarray,
    init_staff: np.ndarray,
    target1: float = 0.80,
    limit1: int = 1,
    target2: float = 0.95,
    limit2: int = 2,
    reps: int = 100,
    random_state: int | None = None
) -> np.ndarray:
    """Подбор плана распределения персонала, начиная с минимального плана, увеличением штата на 1 человека

    Args:
        lambdas (np.ndarray): Прогнозные значения заявок на день t
        speeds (np.ndarray): Эмпирическое распределение средних дневных скоростей сотрудников
        init_staff (np.ndarray): Начальный план распределения сотрудников
        target1 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.80.
        limit1 (int, optional): Целевой максимума срока заявки в днях. Defaults to 1.
        target2 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.95.
        limit2 (int, optional): Целевой максимума срока заявки в днях. Defaults to 2.
        reps (int, optional): Количество повторений симуляции. Defaults to 100
        random_state (int | None, optional): Seed для воспроизводимости. Defaults to None.

    Returns:
        staff (np.ndarray): План распределения персонала на прогнозируемую приемную кампанию
    """
    rng = np.random.default_rng(random_state)
    staff = init_staff.copy()

    while True:
        seed = rng.integers(1e9)
        success1, success2, gap1, gap2 = simulate_campaign(lambdas, staff, speeds, reps=reps, target1=target1, backlog_limit_days1=limit1, backlog_limit_days2=limit2, target2=target2, random_state=seed)

        if gap1 >= 0 and gap2 >= 0:
            break
        
        # Увеличение количества сотрудников в день с минимальным уровнем обслуживания
        margin1 = success1 - target1
        margin2 = success2 - target2

        worst_day = np.argmin(np.minimum(margin1, margin2))
        staff[worst_day] += 1
    return staff

def optimize_staff_topdown(
    lambdas: np.ndarray,
    speeds: np.ndarray,
    init_staff: np.ndarray,
    target1: float = 0.80,
    limit1: int = 1,
    target2: float = 0.95,
    limit2: int = 2,
    random_state: int | None = None
) -> np.ndarray:
    """Подбор плана распределения персонала, начиная с максимального плана, уменьшением штата на 1 человека

    Args:
        lambdas (np.ndarray): Прогнозные значения заявок на день t
        speeds (np.ndarray): Эмпирическое распределение средних дневных скоростей сотрудников
        init_staff (np.ndarray): Начальный план распределения сотрудников
        target1 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.80.
        limit1 (int, optional): Целевой максимума срока заявки в днях. Defaults to 1.
        target2 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.95.
        limit2 (int, optional): Целевой максимума срока заявки в днях. Defaults to 2.
        random_state (int | None, optional): Seed для воспроизводимости. Defaults to None.

    Returns:
        staff (np.ndarray): План распределения персонала на прогнозируемую приемную кампанию
    """
    rng = np.random.default_rng(random_state)
    staff = init_staff.copy()

    while True:
        seed = rng.integers(1e9)
        success1, success2, _, _ = simulate_campaign(lambdas, staff, speeds, backlog_limit_days1=limit1, target1=target1, backlog_limit_days2=limit2, target2=target2, random_state=seed)

        margins = np.minimum(success1 - target1, success2 - target2)

        order = np.argsort(-margins) # Идем от самого избыточного дня
        changed = False # Флаг, было ли изменение

        for d in order:
            while staff[d] > 1 and margins[d] > 0:
                trial = staff.copy()
                trial[d] -= 1
                success_trial1, success_trial2, gap_trial1, gap_trial2 = simulate_campaign(lambdas, trial, speeds, backlog_limit_days1=limit1, target1=target1, backlog_limit_days2=limit2, target2=target2, random_state=seed)

                if gap_trial1 >= 0 and gap_trial2 >= 0: # Изменение удачно
                    staff = trial
                    margins = np.minimum(success_trial1 - target1, success_trial2 - target2)
                    changed = True
                else: # Уровень сервиса стал меньше целевого
                    break

        if not changed: # Если не было изменений ни за один день
            break

    return staff

# Rebalance
def optimize_staff_rebalance(
    lambdas: np.ndarray,
    speeds: np.ndarray,
    init_staff: np.ndarray,
    target1: float = 0.80,
    limit1: int = 1,
    target2: float = 0.95,
    limit2: int = 2,
    max_iters: int = 500,
    random_state: int | None = None
) -> np.ndarray:
    """Изменение распределения сотрудников внутри приемной кампании с целью уменьшения общего количества сотрудников за всю приемную кампанию
    Полученное распределение считается лучше предыдущего, если выполняется одно из условий:
    1. Общее количество сотрудников меньше, чем было
    2. Общее количество сотрудников осталось прежним, но план стал равномернее (стандартное отклонение стало меньше)

    Args:
        lambdas (np.ndarray): Прогнозные значения заявок на день t
        speeds (np.ndarray): Эмпирическое распределение средних дневных скоростей сотрудников
        init_staff (np.ndarray): Начальный план распределения сотрудников, удовлетворяющий уровню сервиса
        target1 (float, optional):  Целевой процент обслуживания заявок. Defaults to 0.80.
        limit1 (int, optional): Целевой максимума срока заявки в днях. Defaults to 1.
        target2 (float, optional):  Целевой процент обслуживания заявок. Defaults to 0.95.
        limit2 (int, optional): Целевой максимума срока заявки в днях. Defaults to 2.
        max_iters (int, optional): Максимальное количество попыток ребалансировки. Defaults to 500.
        random_state (int | None, optional): Seed для воспроизводимости. Defaults to None.

    Returns:
        staff (np.ndarray): План распределения персонала на прогнозируемую приемную кампанию
    """
    rng = np.random.default_rng(random_state)
    staff = init_staff.copy()
    
    # Базовые метрики
    base_total = staff.sum()
    base_std = staff.std()

    for _ in range(max_iters):
        seed = rng.integers(1e9) # генератор случайных seed для воспроизводимости
        success1, success2, _, _ = simulate_campaign(lambdas, staff, speeds, backlog_limit_days1=limit1, target1=target1, backlog_limit_days2=limit2, target2=target2, random_state=seed)
        margins = np.minimum(success1 - target1, success2 - target2)

        donors_idx = np.where((staff > 1) & (margins > 0))[0]
        if donors_idx.size == 0:
            break

        donors_idx = donors_idx[np.argsort(-margins[donors_idx])] # индексы в порядке убывания избыточности дней
        improved = False

        for donor_idx in donors_idx:
            recipient_idx = np.argmin(margins)

            if donor_idx == recipient_idx:
                continue

            trial = staff.copy()
            trial[donor_idx] -= 1
            trial[recipient_idx] += 1

            _, _, gap1, gap2 = simulate_campaign(lambdas, trial, speeds, backlog_limit_days1=limit1, target1=target1, backlog_limit_days2=limit2, target2=target2, random_state=seed)

            if gap1 < 0 or gap2 < 0:
                continue

            trial_opt = optimize_staff_topdown(lambdas, speeds, trial, limit1=limit1, target1=target1, limit2=limit2, target2=target2, random_state=seed)
            new_total = trial_opt.sum()
            new_std = trial_opt.std()

            better = (new_total < base_total) or (new_total == base_total and new_std < base_std)

            if better:
                staff = trial_opt
                base_total = new_total
                base_std = new_std
                improved = True
                break

        if not improved:
            break

    # Сглаживание количества персонала между днями 
    changed = True
    while changed:
        changed = False

        donors_idx = np.argsort(-staff)
        recipients_idx = np.argsort(staff)

        for donor_idx in donors_idx:
            for rec_idx in recipients_idx:
                # Добавим допустимое различие в количестве сотрудников между днями
                # для предотвращения вырождения плана в равномерное распределение
                if staff[donor_idx] - staff[rec_idx] <= 1:
                    break

                trial = staff.copy()
                trial[donor_idx] -= 1
                trial[rec_idx] += 1
                
                seed = rng.integers(1e7)
                _, _, gap1, gap2 = simulate_campaign(lambdas, trial, speeds, target1=target1, backlog_limit_days1=limit1, target2=target2, backlog_limit_days2=limit2, random_state=seed)
                if gap1 >= 0 and gap2 >= 0:
                    staff = trial
                    changed = True
                    break
            if changed:
                break

    return staff

def optimize_staff(
    lambdas: np.ndarray,
    speeds: np.ndarray,
    target1: float = 0.80,
    limit1: int = 1,
    target2: float = 0.95,
    limit2: int = 2,
    reps: int = 100,
    max_iters: int = 500,
    init_staff_per_day: int = 8,
    random_state: int | None = None
) -> np.ndarray:
    """Пайплайн для оптимизации распределения сотрудников приемной комиссии: bottom-up + rebalance

    Args:
        lambdas (np.ndarray): Прогнозные значения заявок на день t
        speeds (np.ndarray): Эмпирическое распределение средних дневных скоростей сотрудников
        target1 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.80.
        limit1 (int, optional): Целевой максимума срока заявки в днях. Defaults to 1.
        target2 (float, optional): Целевой процент обслуживания заявок. Defaults to 0.95.
        limit2 (int, optional): Целевой максимума срока заявки в днях. Defaults to 2.
        reps (int, optional): Количество повторений симуляции приемной кампании. Defaults to 100.
        max_iters (int, optional): Максимальное количество попыток ребалансировки. Defaults to 500.
        init_staff_per_day (int, optional): Начальное количество персонала в день. Defaults to 8.
        random_state (int | None, optional): Seed для воспроизводимости. Defaults to None.

    Returns:
        staff (np.ndarray): План распределения персонала на прогнозируемую приемную кампанию
    """
    init_staff = np.full(len(lambdas), init_staff_per_day, dtype=int)
    opt_staff = optimize_staff_bottomup(lambdas, speeds, init_staff, target1=target1, limit1=limit1, target2=target2, limit2=limit2, reps=reps, random_state=random_state)
    reb_staff = optimize_staff_rebalance(lambdas, speeds, opt_staff, target1=target1, limit1=limit1, target2=target2, limit2=limit2, max_iters=max_iters, random_state=random_state)
    return reb_staff