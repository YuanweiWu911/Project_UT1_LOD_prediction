# UT1/LOD 预测软件用户手册

## 概述
本Python软件包提供了使用LS+AR方法预测世界时1（UT1）和日长（LOD）变化的工具。它基于IERS规范实现了算法，包括以下功能：

- 月球/太阳潮汐效应计算
- UT1/LOD时间序列分析
- 功率谱分析
- 最小二乘建模
- 基于ARMA的预测
- 地球定向参数（EOP）数据处理

## 软件包结构
该软件包由三个主要模块组成：

1. **lunisolar.py** - 月球/太阳潮汐计算
2. **miscfunc.py** - 实用函数和预测
3. **__init__.py** - 软件包初始化和导出

## 主要功能

### 月球/太阳潮汐效应
- `lunisolarf5()`: 计算月球/太阳章动参数
- `calc_dut1()`: 计算DUT1、DLOD、DOMEGA值
- `tidal_table()`: 读取IERS2010潮汐表

### 时间序列分析
- `powersp()`: 功率谱密度计算
- `diff()`: 数值微分
- `acumu()`: 数值积分
- `extrap()`: 时间序列外推

### 预测
- `ARforecast_ut1_1/2/3()`: UT1预测方法
- `ARforecast_pmxy_1/2/3()`: 极移预测
- `ar_forecast()`: 通用ARMA预测

### 数据处理
- `read_eopc04()`: 读取EOP C04格式文件
- `read_usno()`: 读取USNO格式文件
- `get_leap_second()`: 获取闰秒信息

## 使用示例

### 基本设置
```python
from iers import lunisolarf5, calc_dut1, ARforecast_ut1_1
```

### 月球/太阳计算
```python
# 计算月球/太阳章动参数
mjd = 59000.0
f1, f2, f3, f4, f5 = lunisolarf5(mjd)

# 计算潮汐效应
dut1, dlod, domega = calc_dut1(mjd)
```

### 时间序列分析
```python
# 计算功率谱
freq, psd = powersp(ut1_series)

# 数值微分
diff_series = diff(ut1_series)
```

### 预测
```python
# 预测下一天的UT1
new_mjd, forecast_ut1, p = ARforecast_ut1_1(ut1_series, mjd_series, 'aic', 100)
```

### 数据处理
```python
# 读取EOP C04数据
eop_data = read_eopc04('eopc04.1962-now')

# 获取闰秒信息
leap_sec = get_leap_second(mjd)
```

## 输入/输出格式

### 输入数据
- MJD（修正儒略日）作为主要时间格式
- UT1-UTC序列（秒）
- LOD序列（秒）
- 极坐标（角秒）

### 输出数据
- 预测结果与输入单位相同
- 功率谱为频率/振幅格式
- 潮汐效应（秒）

## 依赖项
该软件包需要：
- NumPy
- Pandas
- SciPy
- statsmodels
- astropy

## 参考文献
- IERS规范（2010）
- USNO地球定向产品
- IERS EOP C04系列
