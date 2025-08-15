
# -*- coding: utf-8 -*-
"""
1부 실습: 시계열 데이터 이해와 분석(EDA) - Bike Sharing (Hourly)
- matplotlib만 사용 (강의 규칙 준수: seaborn 사용 X, 색상 지정 X, 서브플롯 X)
- 시계열 특성 이해에 필요한 다양한 실습 포함
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess

# ===== 0. 데이터 불러오기 및 기본 전처리 =====
def load_data():
    cand = ['train.csv', './data/train.csv', '/mnt/data/train.csv']
    path = None
    for c in cand:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError("train.csv 경로를 찾을 수 없습니다. 현재 디렉토리에 놓거나 ./data, /mnt/data 중 한 곳에 두세요.")
    df = pd.read_csv(path)
    # 타입 처리
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 정렬 및 인덱스
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[['datetime','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']]
    return df

df = load_data()
print("데이터 크기:", df.shape)
print("기간:", df['datetime'].min(), "→", df['datetime'].max())
print(df.head(3))

# ===== 1. 시간 간격/결측/중복 체크 =====
def check_time_gaps(data):
    # 중복
    dup = data['datetime'].duplicated().sum()
    print("중복 datetime 개수:", dup)

    # 시간 간격
    diffs = data['datetime'].diff().dropna()
    print("가장 흔한 간격(top 3):")
    print(diffs.value_counts().head(3))

    # 누락 시간(정규 hourly 가정)
    full = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='H')
    missing = np.setdiff1d(full.values, data['datetime'].values)
    print("누락된 시간 개수:", len(missing))

check_time_gaps(df)

# 인덱스 설정(시계열 분석 편의)
df = df.set_index('datetime')

# ===== 2. 집계 단위 변환: 시간→일/주 평균 =====
def resample_and_plot(series, rule, title):
    agg = series.resample(rule).mean()
    plt.figure(figsize=(12,4))
    plt.plot(agg.index, agg.values)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    return agg

hourly = df['count']
daily = resample_and_plot(hourly, 'D', '일 단위 평균 대여량 추세')
weekly = resample_and_plot(hourly, 'W', '주 단위 평균 대여량 추세')

# ===== 3. 달력(Weekday x Week) 히트맵으로 패턴 보기 =====
def calendar_heatmap_daily(daily_series):
    # 월 단위 캘린더 히트맵: (week_of_month x weekday)
    s = daily_series.copy()
    dti = s.index
    year_month = dti.to_period('M')
    # 월별로 그리기
    for ym, ss in s.groupby(year_month):
        cal = pd.DataFrame(index=range(6), columns=range(7), dtype=float)
        # 첫 날 위치 계산
        first = ss.index.min()
        week = 0
        for d, val in ss.items():
            if d.weekday() == 0 and d.day != 1 and d.day <= 7:
                week = 1
            # pandas에는 weekofmonth 기본 제공 X, 간단히 주차 계산
            week_of_month = ((d.day + first.weekday() - 1) // 7)
            cal.loc[week_of_month, d.weekday()] = val

        plt.figure(figsize=(7,4))
        plt.imshow(cal.values, aspect='auto', interpolation='nearest')
        plt.title(f'Calendar Heatmap: {ym}')
        plt.yticks(range(6), [f'w{i+1}' for i in range(6)])
        plt.xticks(range(7), ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        plt.colorbar(label='Avg Count')
        plt.tight_layout()
        plt.show()

calendar_heatmap_daily(daily)

# ===== 4. 범주별 박스플롯(연/월/요일/시간) – matplotlib =====
def boxplot_by_category(df_reset, col, title, labels=None):
    plt.figure(figsize=(10,4))
    groups = [g['count'].values for _, g in df_reset.groupby(col)]
    plt.boxplot(groups, showfliers=True)
    if labels is None:
        labels = sorted(df_reset[col].unique())
    plt.xticks(range(1, len(groups)+1), labels, rotation=0)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

df_reset = df.reset_index()
df_reset['year'] = df_reset['datetime'].dt.year
df_reset['month'] = df_reset['datetime'].dt.month
df_reset['dayofweek'] = df_reset['datetime'].dt.dayofweek
df_reset['hour'] = df_reset['datetime'].dt.hour

boxplot_by_category(df_reset, 'year', '연도별 분포')
boxplot_by_category(df_reset, 'month', '월별 분포', labels=list(range(1,13)))
boxplot_by_category(df_reset, 'dayofweek', '요일별 분포 (0=Mon)', labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
boxplot_by_category(df_reset, 'hour', '시간대별 분포', labels=list(range(24)))

# ===== 5. 주중/주말 & 휴일 효과 =====
def mean_ci(vals):
    m = np.mean(vals)
    s = np.std(vals, ddof=1)
    n = len(vals)
    se = s/np.sqrt(n)
    return m, 1.96*se  # 95% CI

profiles = []
for name, cond in {
    'Workingday': df_reset['workingday']==1,
    'Weekend': df_reset['dayofweek']>=5,
    'Holiday': df_reset['holiday']==1
}.items():
    m, ci = mean_ci(df_reset.loc[cond, 'count'])
    profiles.append((name, m, ci))

plt.figure(figsize=(6,4))
x = np.arange(len(profiles))
means = [p[1] for p in profiles]
cis = [p[2] for p in profiles]
plt.bar(x, means, yerr=cis)
plt.xticks(x, [p[0] for p in profiles])
plt.title('근무일/주말/공휴일 평균과 95% CI')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ===== 6. 날씨 변수와 수요의 비선형 관계(LOWESS 스무딩) =====
def scatter_lowess(x, y, xlabel, title):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    sm = lowess(ys, xs, frac=0.1, return_sorted=True)
    plt.figure(figsize=(7,4))
    plt.scatter(xs, ys, s=5, alpha=0.2)
    plt.plot(sm[:,0], sm[:,1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

scatter_lowess(df_reset['temp'].values, df_reset['count'].values, 'Temp', '온도 vs 대여량 (LOWESS)')
scatter_lowess(df_reset['humidity'].values, df_reset['count'].values, 'Humidity', '습도 vs 대여량 (LOWESS)')
scatter_lowess(df_reset['windspeed'].values, df_reset['count'].values, 'Windspeed', '풍속 vs 대여량 (LOWESS)')

# ===== 7. 이동평균/이동분산으로 비정상성(평균/분산 변동) 보기 =====
def rolling_stats(series, window, title_prefix):
    rm = series.rolling(window).mean()
    rv = series.rolling(window).var()
    plt.figure(figsize=(12,3.5))
    plt.plot(series.index, series.values)
    plt.title(f'{title_prefix}: 원시 시계열')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,3.5))
    plt.plot(rm.index, rm.values)
    plt.title(f'{title_prefix}: 이동평균(window={window})')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,3.5))
    plt.plot(rv.index, rv.values)
    plt.title(f'{title_prefix}: 이동분산(window={window})')
    plt.tight_layout()
    plt.show()

rolling_stats(daily, 7, '일 단위')

# ===== 8. 분포 왜도/첨도 및 변환(log1p) 비교 =====
def hist_compare(raw, title_prefix):
    plt.figure(figsize=(6,4))
    plt.hist(raw.dropna(), bins=30, alpha=0.8)
    plt.title(f'{title_prefix}: 원시 분포')
    plt.tight_layout()
    plt.show()

    transformed = np.log1p(raw)
    plt.figure(figsize=(6,4))
    plt.hist(transformed.dropna(), bins=30, alpha=0.8)
    plt.title(f'{title_prefix}: log1p 변환 후 분포')
    plt.tight_layout()
    plt.show()

hist_compare(daily, '일 평균 대여량')

# ===== 9. 정상성 검정(ADF) – 원시/차분 비교 =====
def adf_test(series, name):
    series = series.dropna()
    res = adfuller(series, autolag='AIC')
    print(f'[ADF] {name}')
    print('  Test Statistic:', res[0])
    print('  p-value       :', res[1])
    print('  Lags Used     :', res[2])
    print('  N Observations:', res[3])
    print('  Critical Values:', res[4])
    print('-'*40)

adf_test(daily, 'Daily mean count')
adf_test(daily.diff(), 'Daily diff(1)')
adf_test(daily.diff(7), 'Daily diff(7) – 주간 차분')

# ===== 10. 자기상관/부분자기상관 =====
def acf_pacf(series, lags, title_prefix):
    series = series.dropna()
    plt.figure(figsize=(8,4))
    plot_acf(series, lags=lags)
    plt.title(f'{title_prefix} ACF (lags={lags})')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plot_pacf(series, lags=lags, method='ywm')
    plt.title(f'{title_prefix} PACF (lags={lags})')
    plt.tight_layout()
    plt.show()

acf_pacf(daily, 60, 'Daily')
acf_pacf(daily.diff().dropna(), 60, 'Daily diff(1)')

# ===== 11. STL 분해 (추세/계절/잔차) – 각 컴포넌트 개별 플롯 =====
def stl_components(series, period, title_prefix):
    series = series.asfreq('D')
    stl = STL(series, period=period, robust=True).fit()
    comp = {
        'Trend': stl.trend,
        'Seasonal': stl.seasonal,
        'Residual': stl.resid
    }
    for k, v in comp.items():
        plt.figure(figsize=(12,3.5))
        plt.plot(v.index, v.values)
        plt.title(f'{title_prefix} {k} (period={period})')
        plt.tight_layout()
        plt.show()
    return comp

stl_components(daily, period=7, title_prefix='STL (Daily)')

# ===== 12. 주기 탐색(FFT) =====
def top_periods_fft(series, max_k=5, freq='D'):
    s = series.dropna().values
    s = s - s.mean()
    n = len(s)
    # 다음 2의 제곱수 길이로 패딩(속도/가독성)
    nfft = int(2**np.ceil(np.log2(n)))
    sp = np.fft.rfft(s, n=nfft)
    power = (sp*np.conj(sp)).real
    freqs = np.fft.rfftfreq(nfft, d=1.0)  # 샘플 간격 1
    # DC 성분 제외
    freqs, power = freqs[1:], power[1:]
    # 주기(길이)로 변환
    periods = 1/freqs
    idx = np.argsort(power)[::-1][:max_k]
    top = list(zip(periods[idx], power[idx]))
    print("FFT Top periods (길이, 파워):")
    for p, pw in top:
        if freq == 'H':
            print(f'  {p:.1f} 시간 (power={pw:.2e})')
        else:
            print(f'  {p:.1f} 일 (power={pw:.2e})')

    # 스펙트럼 플롯
    plt.figure(figsize=(10,4))
    plt.plot(periods, power)
    plt.xlim(0, min(60, periods.max()))
    plt.title('Periodogram (주기-파워)')
    plt.xlabel('주기(일 기준)')
    plt.ylabel('Power')
    plt.tight_layout()
    plt.show()

top_periods_fft(daily, max_k=5, freq='D')

# ===== 13. 시점/시즌별 프로파일(시간대 x 시즌) =====
def hourly_profile_by_season(df_reset):
    fig_data = {}
    for s in sorted(df_reset['season'].unique()):
        sub = df_reset[df_reset['season']==s]
        prof = sub.groupby('hour')['count'].mean()
        fig_data[s] = prof

    plt.figure(figsize=(10,4))
    for s, prof in fig_data.items():
        plt.plot(prof.index, prof.values, label=f'season={s}')
    plt.title('시간대 프로파일(시즌별)')
    plt.xlabel('Hour')
    plt.ylabel('Avg Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

hourly_profile_by_season(df_reset)

# ===== 14. 래그 상관(1~48시간) =====
def lag_correlation(series, max_lag=48):
    corrs = []
    y = series.values
    for k in range(1, max_lag+1):
        x = series.shift(k).values
        valid = ~np.isnan(x)
        corrs.append(np.corrcoef(x[valid], y[valid])[0,1])
    lags = np.arange(1, max_lag+1)
    plt.figure(figsize=(10,3.5))
    plt.stem(lags, corrs, use_line_collection=True)
    plt.title('Lag Correlation (1~48시간)')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.show()

lag_correlation(hourly, 48)

# ===== 15. 이상치(Outlier) 탐색 – IQR 기준 상위 구간 표시 =====
def outlier_mark(series, k=1.5, title='IQR 기반 이상치'):
    q1, q3 = np.percentile(series.dropna(), [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    mask = (series < lo) | (series > hi)
    plt.figure(figsize=(12,3.5))
    plt.plot(series.index, series.values, alpha=0.7)
    plt.scatter(series.index[mask], series.values[mask], s=10)
    plt.title(f'{title} (lo={lo:.1f}, hi={hi:.1f})')
    plt.tight_layout()
    plt.show()
    print("이상치 개수:", mask.sum())

outlier_mark(daily, k=2.0, title='일 평균 대여량 이상치')

# ===== 16. 누락값이 있다고 가정한 보간 시연(Fill) =====
def imputation_demo(series):
    s = series.copy()
    # 임의로 일부 값 누락(시연용)
    idx = s.sample(frac=0.01, random_state=42).index
    s.loc[idx] = np.nan

    plt.figure(figsize=(10,3.5))
    plt.plot(s.index, s.values)
    plt.title('임의 결측 삽입')
    plt.tight_layout()
    plt.show()

    ffill = s.ffill()
    plt.figure(figsize=(10,3.5))
    plt.plot(ffill.index, ffill.values)
    plt.title('Forward Fill 결과')
    plt.tight_layout()
    plt.show()

    lin = s.interpolate(method='linear')
    plt.figure(figsize=(10,3.5))
    plt.plot(lin.index, lin.values)
    plt.title('Linear Interpolation 결과')
    plt.tight_layout()
    plt.show()

imputation_demo(daily)

# ===== 17. 간단한 베이스라인(계절 나이브)로 홀드아웃 점검 =====
def seasonal_naive_baseline(hourly_series, test_days=7):
    s = hourly_series.asfreq('H')
    split = s.index.max() - pd.Timedelta(days=test_days)
    train = s.loc[:split]
    test = s.loc[split+pd.Timedelta(hours=1):]

    # 24시간 계절 나이브: yhat_t = y_{t-24}
    yhat = test.shift(24).copy()
    yhat[:] = train.iloc[-24:].values  # 마지막 하루 패턴으로 초기화
    yhat = s.shift(24).loc[test.index]  # 단순 계절 나이브

    # 성능
    def rmse(a, b):
        return np.sqrt(np.mean((a-b)**2))
    def mape(a, b):
        eps = 1e-8
        return np.mean(np.abs((a-b)/(a+eps)))*100

    r = rmse(test.values, yhat.values)
    m = mape(test.values, yhat.values)
    print(f'[Seasonal Naive] 홀드아웃 {test_days}일 RMSE={r:.2f}, MAPE={m:.2f}%')

    plt.figure(figsize=(12,4))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(yhat.index, yhat.values, label='Seasonal Naive (24h)')
    plt.title('홀드아웃 예측: 계절 나이브(24h)')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

seasonal_naive_baseline(hourly, test_days=7)

print("✅ 실습 스크립트 실행 완료")
