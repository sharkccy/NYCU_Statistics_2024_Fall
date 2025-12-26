#pip install geopandas, imageio first
import statistics
import statistics as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import geopandas as gpd
import imageio


period_all = pd.read_csv('period_all.csv')
period1 = pd.read_csv('period1.csv')
period2 = pd.read_csv('period2.csv')
period3 = pd.read_csv('period3.csv')
period4 = pd.read_csv('period4.csv')

mean_takedowns_all = period_all["取締件數每十萬人"].mean()
mean_takedowns_period1 = period1["取締件數每十萬人"].mean()
mean_takedowns_period2 = period2["取締件數每十萬人"].mean()
mean_takedowns_period3 = period3["取締件數每十萬人"].mean()
mean_takedowns_period4 = period4["取締件數每十萬人"].mean()

mean_sent_to_prison_all = period_all["移送法辦每十萬人"].mean()
mean_sent_to_prison_period1 = period1["移送法辦每十萬人"].mean()
mean_sent_to_prison_period2 = period2["移送法辦每十萬人"].mean()
mean_sent_to_prison_period3 = period3["移送法辦每十萬人"].mean()
mean_sent_to_prison_period4 = period4["移送法辦每十萬人"].mean()

mean_accident_all = period_all["事故件數每十萬人"].mean()
mean_accident_period1 = period1["事故件數每十萬人"].mean()
mean_accident_period2 = period2["事故件數每十萬人"].mean()
mean_accident_period3 = period3["事故件數每十萬人"].mean()
mean_accident_period4 = period4["事故件數每十萬人"].mean()

mean_death_all = period_all["死亡人數每十萬人"].mean()
mean_death_period1 = period1["死亡人數每十萬人"].mean()
mean_death_period2 = period2["死亡人數每十萬人"].mean()
mean_death_period3 = period3["死亡人數每十萬人"].mean()
mean_death_period4 = period4["死亡人數每十萬人"].mean()

mean_injury_all = period_all["受傷人數每十萬人"].mean()
mean_injury_period1 = period1["受傷人數每十萬人"].mean()
mean_injury_period2 = period2["受傷人數每十萬人"].mean()
mean_injury_period3 = period3["受傷人數每十萬人"].mean()
mean_injury_period4 = period4["受傷人數每十萬人"].mean()

var_takedowns_all = period_all["取締件數每十萬人"].var()
var_takedowns_period1 = period1["取締件數每十萬人"].var()
var_takedowns_period2 = period2["取締件數每十萬人"].var()
var_takedowns_period3 = period3["取締件數每十萬人"].var()
var_takedowns_period4 = period4["取締件數每十萬人"].var()

var_sent_to_prison_all = period_all["移送法辦每十萬人"].var()
var_sent_to_prison_period1 = period1["移送法辦每十萬人"].var()
var_sent_to_prison_period2 = period2["移送法辦每十萬人"].var()
var_sent_to_prison_period3 = period3["移送法辦每十萬人"].var()
var_sent_to_prison_period4 = period4["移送法辦每十萬人"].var()

vqr_accident_all = period_all["事故件數每十萬人"].var()
var_accident_period1 = period1["事故件數每十萬人"].var()
var_accident_period2 = period2["事故件數每十萬人"].var()
var_accident_period3 = period3["事故件數每十萬人"].var()
var_accident_period4 = period4["事故件數每十萬人"].var()

var_death_all = period_all["死亡人數每十萬人"].var()
var_death_period1 = period1["死亡人數每十萬人"].var()
var_death_period2 = period2["死亡人數每十萬人"].var()
var_death_period3 = period3["死亡人數每十萬人"].var()
var_death_period4 = period4["死亡人數每十萬人"].var()

var_injury_all = period_all["受傷人數每十萬人"].var()
var_injury_period1 = period1["受傷人數每十萬人"].var()
var_injury_period2 = period2["受傷人數每十萬人"].var()
var_injury_period3 = period3["受傷人數每十萬人"].var()
var_injury_period4 = period4["受傷人數每十萬人"].var()

#---------------------Takedowns anova---------------------
print("takedowns anova")
alpha = 0.05
treatments = 4
len_period1 = len(period1)
len_period2 = len(period2)
len_period3 = len(period3)
len_period4 = len(period4)

SST = len_period1 * (mean_takedowns_period1 - mean_takedowns_all) ** 2 + len_period2 * (mean_takedowns_period2 - mean_takedowns_all) ** 2 + len_period3 * (mean_takedowns_period3 - mean_takedowns_all) ** 2 + len_period4 * (mean_takedowns_period4 - mean_takedowns_all) ** 2
SSE = (len_period1 - 1) * var_takedowns_period1 + (len_period2 - 1) * var_takedowns_period2 + (len_period3 - 1) * var_takedowns_period3 + (len_period4 - 1) * var_takedowns_period4
MST = SST / (treatments - 1)
MSE = SSE / (len_period1 + len_period2 + len_period3 + len_period4 - treatments)
F = MST / MSE
F_critical = stats.f.ppf(1 - alpha, treatments - 1, len_period1 + len_period2 + len_period3 + len_period4 - treatments)
p_value = 1 - stats.f.cdf(F, treatments - 1, len_period1 + len_period2 + len_period3 + len_period4 - treatments)
print("F: ", F)
print("F critical: ", F_critical)
print("p-value: ", p_value)

#---------------------Takedowns t-test---------------------
print("takedowns t-test")
alpha = 0.05
alpha = alpha / 6
dof_MSE = len_period1 + len_period2 + len_period3 + len_period4 - treatments
pool_estimator = MSE ** 0.5

dof_pair1 = len_period1 + len_period2 - 2
dof_pair2 = len_period1 + len_period3 - 2
dof_pair3 = len_period1 + len_period4 - 2
dof_pair4 = len_period2 + len_period3 - 2
dof_pair5 = len_period2 + len_period4 - 2
dof_pair6 = len_period3 + len_period4 - 2

p_value_of_pair1 = 2 * (1 - stats.t.cdf((mean_takedowns_period1 - mean_takedowns_period2) / (pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5), dof_pair1))
p_value_of_pair2 = 2 * (1 - stats.t.cdf((mean_takedowns_period1 - mean_takedowns_period3) / (pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5), dof_pair2))
p_value_of_pair3 = 2 * (1 - stats.t.cdf((mean_takedowns_period1 - mean_takedowns_period4) / (pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5), dof_pair3))
p_value_of_pair4 = 2 * (1 - stats.t.cdf((mean_takedowns_period2 - mean_takedowns_period3) / (pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5), dof_pair4))
p_value_of_pair5 = 2 * (1 - stats.t.cdf((mean_takedowns_period2 - mean_takedowns_period4) / (pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5), dof_pair5))
p_value_of_pair6 = 2 * (1 - stats.t.cdf((mean_takedowns_period3 - mean_takedowns_period4) / (pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5), dof_pair6))

upper_bound_pair1 = (mean_takedowns_period1 - mean_takedowns_period2) + stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5
lower_bound_pair1 = (mean_takedowns_period1 - mean_takedowns_period2) - stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5
upper_bound_pair2 = (mean_takedowns_period1 - mean_takedowns_period3) + stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5
lower_bound_pair2 = (mean_takedowns_period1 - mean_takedowns_period3) - stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5
upper_bound_pair3 = (mean_takedowns_period1 - mean_takedowns_period4) + stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5
lower_bound_pair3 = (mean_takedowns_period1 - mean_takedowns_period4) - stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5
upper_bound_pair4 = (mean_takedowns_period2 - mean_takedowns_period3) + stats.t.ppf(1 - alpha/2, dof_pair4) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
lower_bound_pair4 = (mean_takedowns_period2 - mean_takedowns_period3) - stats.t.ppf(1 - alpha/2, dof_pair4) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
upper_bound_pair5 = (mean_takedowns_period2 - mean_takedowns_period4) + stats.t.ppf(1 - alpha/2, dof_pair5) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
lower_bound_pair5 = (mean_takedowns_period2 - mean_takedowns_period4) - stats.t.ppf(1 - alpha/2, dof_pair5) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
upper_bound_pair6 = (mean_takedowns_period3 - mean_takedowns_period4) + stats.t.ppf(1 - alpha/2, dof_pair6) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5
lower_bound_pair6 = (mean_takedowns_period3 - mean_takedowns_period4) - stats.t.ppf(1 - alpha/2, dof_pair6) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5

print("1-2 p-value: ", p_value_of_pair1)
print("1-2 Upper bound pair1: ", upper_bound_pair1)
print("1-2 Lower bound pair1: ", lower_bound_pair1)

print("1-3 p-value: ", p_value_of_pair2)
print("1-3 Upper bound pair2: ", upper_bound_pair2)
print("1-3 Lower bound pair2: ", lower_bound_pair2)

print("1-4 p-value: ", p_value_of_pair3)
print("1-4 Upper bound pair3: ", upper_bound_pair3)
print("1-4 Lower bound pair3: ", lower_bound_pair3)

print("2-3 p-value: ", p_value_of_pair4)
print("2-3 Upper bound pair4: ", upper_bound_pair4)
print("2-3 Lower bound pair4: ", lower_bound_pair4)

print("2-4 p-value: ", p_value_of_pair5)
print("2-4 Upper bound pair5: ", upper_bound_pair5)
print("2-4 Lower bound pair5: ", lower_bound_pair5)

print("3-4 p-value: ", p_value_of_pair6)
print("3-4 Upper bound pair6: ", upper_bound_pair6)
print("3-4 Lower bound pair6: ", lower_bound_pair6)

#---------------------Sent to prison anova---------------------
print("sent to prison anova")
alpha = 0.05
treatments = 4
len_period1 = len(period1)
len_period2 = len(period2)
len_period3 = len(period3)
len_period4 = len(period4)

SST = len_period1 * (mean_sent_to_prison_period1 - mean_sent_to_prison_all) ** 2 + len_period2 * (mean_sent_to_prison_period2 - mean_sent_to_prison_all) ** 2 + len_period3 * (mean_sent_to_prison_period3 - mean_sent_to_prison_all) ** 2 + len_period4 * (mean_sent_to_prison_period4 - mean_sent_to_prison_all) ** 2
SSE = (len_period1 - 1) * var_sent_to_prison_period1 + (len_period2 - 1) * var_sent_to_prison_period2 + (len_period3 - 1) * var_sent_to_prison_period3 + (len_period4 - 1) * var_sent_to_prison_period4
MST = SST / (treatments - 1)
MSE = SSE / (len_period1 + len_period2 + len_period3 + len_period4 - treatments)
F = MST / MSE
F_critical = stats.f.ppf(1 - alpha, treatments - 1, len_period1 + len_period2 + len_period3 + len_period4 - treatments)
p_value = 1 - stats.f.cdf(F, treatments - 1, len_period1 + len_period2 + len_period3 + len_period4 - treatments)
print("F: ", F)
print("F critical: ", F_critical)
print("p-value: ", p_value)

#---------------------Sent to prison t-test---------------------
print("sent to prison t-test")
alpha = 0.05
alpha = alpha / 6
dof_MSE = len_period1 + len_period2 + len_period3 + len_period4 - treatments
pool_estimator = MSE ** 0.5

dof_pair1 = len_period1 + len_period2 - 2
dof_pair2 = len_period1 + len_period3 - 2
dof_pair3 = len_period1 + len_period4 - 2
dof_pair4 = len_period2 + len_period3 - 2
dof_pair5 = len_period2 + len_period4 - 2
dof_pair6 = len_period3 + len_period4 - 2

p_value_of_pair1 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period1 - mean_sent_to_prison_period2) / (pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5), dof_pair1))
p_value_of_pair2 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period1 - mean_sent_to_prison_period3) / (pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5), dof_pair2))
p_value_of_pair3 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period1 - mean_sent_to_prison_period4) / (pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5), dof_pair3))
p_value_of_pair4 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period2 - mean_sent_to_prison_period3) / (pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5), dof_pair4))
p_value_of_pair5 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period2 - mean_sent_to_prison_period4) / (pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5), dof_pair5))
p_value_of_pair6 = 2 * (1 - stats.t.cdf((mean_sent_to_prison_period3 - mean_sent_to_prison_period4) / (pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5), dof_pair6))
                        

upper_bound_pair1 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period2) + stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5
lower_bound_pair1 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period2) - stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period1 + 1 / len_period2) ** 0.5
upper_bound_pair2 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period3) + stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5
lower_bound_pair2 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period3) - stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period1 + 1 / len_period3) ** 0.5
upper_bound_pair3 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period4) + stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5
lower_bound_pair3 = (mean_sent_to_prison_period1 - mean_sent_to_prison_period4) - stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period1 + 1 / len_period4) ** 0.5
upper_bound_pair4 = (mean_sent_to_prison_period2 - mean_sent_to_prison_period3) + stats.t.ppf(1 - alpha/2, dof_pair4) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
lower_bound_pair4 = (mean_sent_to_prison_period2 - mean_sent_to_prison_period3) - stats.t.ppf(1 - alpha/2, dof_pair4) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
upper_bound_pair5 = (mean_sent_to_prison_period2 - mean_sent_to_prison_period4) + stats.t.ppf(1 - alpha/2, dof_pair5) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
lower_bound_pair5 = (mean_sent_to_prison_period2 - mean_sent_to_prison_period4) - stats.t.ppf(1 - alpha/2, dof_pair5) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
upper_bound_pair6 = (mean_sent_to_prison_period3 - mean_sent_to_prison_period4) + stats.t.ppf(1 - alpha/2, dof_pair6) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5
lower_bound_pair6 = (mean_sent_to_prison_period3 - mean_sent_to_prison_period4) - stats.t.ppf(1 - alpha/2, dof_pair6) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5

print("1-2 p-value: ", p_value_of_pair1)
print("1-2 Upper bound pair1: ", upper_bound_pair1)
print("1-2 Lower bound pair1: ", lower_bound_pair1)

print("1-3 p-value: ", p_value_of_pair2)
print("1-3 Upper bound pair2: ", upper_bound_pair2)
print("1-3 Lower bound pair2: ", lower_bound_pair2)

print("1-4 p-value: ", p_value_of_pair3)
print("1-4 Upper bound pair3: ", upper_bound_pair3)
print("1-4 Lower bound pair3: ", lower_bound_pair3)

print("2-3 p-value: ", p_value_of_pair4)
print("2-3 Upper bound pair4: ", upper_bound_pair4)
print("2-3 Lower bound pair4: ", lower_bound_pair4)

print("2-4 p-value: ", p_value_of_pair5)
print("2-4 Upper bound pair5: ", upper_bound_pair5)
print("2-4 Lower bound pair5: ", lower_bound_pair5)

print("3-4 p-value: ", p_value_of_pair6)
print("3-4 Upper bound pair6: ", upper_bound_pair6)
print("3-4 Lower bound pair6: ", lower_bound_pair6)

#---------------------Accident anova---------------------
print("accident anova")
alpha = 0.05
treatments = 3
len_period2 = len(period2)
len_period3 = len(period3)
len_period4 = len(period4)

SST = len_period2 * (mean_accident_period2 - mean_accident_all) ** 2 + len_period3 * (mean_accident_period3 - mean_accident_all) ** 2 + len_period4 * (mean_accident_period4 - mean_accident_all) ** 2
SSE = (len_period2 - 1) * var_accident_period2 + (len_period3 - 1) * var_accident_period3 + (len_period4 - 1) * var_accident_period4
MST = SST / (treatments - 1)
MSE = SSE / (len_period2 + len_period3 + len_period4 - treatments)
F = MST / MSE
F_critical = stats.f.ppf(1 - alpha, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
p_value = 1 - stats.f.cdf(F, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
print("F: ", F)
print("F critical: ", F_critical)
print("p-value: ", p_value)

#---------------------Accident t-test---------------------
print("accident t-test")
alpha = 0.05
alpha = alpha / 3
dof_MSE = len_period2 + len_period3 + len_period4 - treatments
pool_estimator = MSE ** 0.5

dof_pair1 = len_period2 + len_period3 - 2
dof_pair2 = len_period2 + len_period4 - 2
dof_pair3 = len_period3 + len_period4 - 2

p_value_of_pair1 = 2 * (1 - stats.t.cdf((mean_accident_period2 - mean_accident_period3) / (pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5), dof_pair1))
p_value_of_pair2 = 2 * (1 - stats.t.cdf((mean_accident_period2 - mean_accident_period4) / (pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5), dof_pair2))
p_value_of_pair3 = 2 * (1 - stats.t.cdf((mean_accident_period3 - mean_accident_period4) / (pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5), dof_pair3))


upper_bound_pair1 = (mean_accident_period2 - mean_accident_period3) + stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
lower_bound_pair1 = (mean_accident_period2 - mean_accident_period3) - stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
upper_bound_pair2 = (mean_accident_period2 - mean_accident_period4) + stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
lower_bound_pair2 = (mean_accident_period2 - mean_accident_period4) - stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
upper_bound_pair3 = (mean_accident_period3 - mean_accident_period4) + stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5
lower_bound_pair3 = (mean_accident_period3 - mean_accident_period4) - stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5

print("2-3 p-value: ", p_value_of_pair1)
print("2-3 Upper bound pair1: ", upper_bound_pair1)
print("2-3 Lower bound pair1: ", lower_bound_pair1)

print("2-4 p-value: ", p_value_of_pair2)
print("2-4 Upper bound pair2: ", upper_bound_pair2)
print("2-4 Lower bound pair2: ", lower_bound_pair2)

print("3-4 p-value: ", p_value_of_pair3)
print("3-4 Upper bound pair3: ", upper_bound_pair3)
print("3-4 Lower bound pair3: ", lower_bound_pair3)

#---------------------Death anova---------------------
print("death anova")
alpha = 0.05
treatments = 3
len_period2 = len(period2)
len_period3 = len(period3)
len_period4 = len(period4)

SST = len_period2 * (mean_death_period2 - mean_death_all) ** 2 + len_period3 * (mean_death_period3 - mean_death_all) ** 2 + len_period4 * (mean_death_period4 - mean_death_all) ** 2
SSE = (len_period2 - 1) * var_death_period2 + (len_period3 - 1) * var_death_period3 + (len_period4 - 1) * var_death_period4
MST = SST / (treatments - 1)
MSE = SSE / (len_period2 + len_period3 + len_period4 - treatments)
F = MST / MSE
F_critical = stats.f.ppf(1 - alpha, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
p_value = 1 - stats.f.cdf(F, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
print("F: ", F)
print("F critical: ", F_critical)
print("p-value: ", p_value)

#---------------------Death t-test---------------------
print("death t-test")
alpha = 0.05
alpha = alpha / 3
dof_MSE = len_period2 + len_period3 + len_period4 - treatments
pool_estimator = MSE ** 0.5

dof_pair1 = len_period2 + len_period3 - 2
dof_pair2 = len_period2 + len_period4 - 2
dof_pair3 = len_period3 + len_period4 - 2

p_value_of_pair1 = 2 * (1 - stats.t.cdf((mean_death_period2 - mean_death_period3) / (pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5), dof_pair1))
p_value_of_pair2 = 2 * (1 - stats.t.cdf((mean_death_period2 - mean_death_period4) / (pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5), dof_pair2))
p_value_of_pair3 = 2 * (1 - stats.t.cdf((mean_death_period3 - mean_death_period4) / (pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5), dof_pair3))


upper_bound_pair1 = (mean_death_period2 - mean_death_period3) + stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
lower_bound_pair1 = (mean_death_period2 - mean_death_period3) - stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
upper_bound_pair2 = (mean_death_period2 - mean_death_period4) + stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
lower_bound_pair2 = (mean_death_period2 - mean_death_period4) - stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
upper_bound_pair3 = (mean_death_period3 - mean_death_period4) + stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5
lower_bound_pair3 = (mean_death_period3 - mean_death_period4) - stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5


print("2-3 p-value: ", p_value_of_pair1)
print("2-3 Upper bound pair1: ", upper_bound_pair1)
print("2-3 Lower bound pair1: ", lower_bound_pair1)

print("2-4 p-value: ", p_value_of_pair2)
print("2-4 Upper bound pair2: ", upper_bound_pair2)
print("2-4 Lower bound pair2: ", lower_bound_pair2)

print("3-4 p-value: ", p_value_of_pair3)
print("3-4 Upper bound pair3: ", upper_bound_pair3)
print("3-4 Lower bound pair3: ", lower_bound_pair3)

#---------------------Injury anova---------------------
print("injury anova")
alpha = 0.05
treatments = 3
len_period2 = len(period2)
len_period3 = len(period3)
len_period4 = len(period4)

SST = len_period2 * (mean_injury_period2 - mean_injury_all) ** 2 + len_period3 * (mean_injury_period3 - mean_injury_all) ** 2 + len_period4 * (mean_injury_period4 - mean_injury_all) ** 2
SSE = (len_period2 - 1) * var_injury_period2 + (len_period3 - 1) * var_injury_period3 + (len_period4 - 1) * var_injury_period4
MST = SST / (treatments - 1)
MSE = SSE / (len_period2 + len_period3 + len_period4 - treatments)
F = MST / MSE
F_critical = stats.f.ppf(1 - alpha, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
p_value = 1 - stats.f.cdf(F, treatments - 1, len_period2 + len_period3 + len_period4 - treatments)
print("F: ", F)
print("F critical: ", F_critical)
print("p-value: ", p_value)

#---------------------Injury t-test---------------------
print("injury t-test")
alpha = 0.05
alpha = alpha / 3
dof_MSE = len_period2 + len_period3 + len_period4 - treatments
pool_estimator = MSE ** 0.5

dof_pair1 = len_period2 + len_period3 - 2
dof_pair2 = len_period2 + len_period4 - 2
dof_pair3 = len_period3 + len_period4 - 2

p_value_of_pair1 = 2 * (1 - stats.t.cdf((mean_injury_period2 - mean_injury_period3) / (pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5), dof_pair1))
p_value_of_pair2 = 2 * (1 - stats.t.cdf((mean_injury_period2 - mean_injury_period4) / (pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5), dof_pair2))
p_value_of_pair3 = 2 * (1 - stats.t.cdf((mean_injury_period3 - mean_injury_period4) / (pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5), dof_pair3))

upper_bound_pair1 = (mean_injury_period2 - mean_injury_period3) + stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
lower_bound_pair1 = (mean_injury_period2 - mean_injury_period3) - stats.t.ppf(1 - alpha/2, dof_pair1) * pool_estimator * (1 / len_period2 + 1 / len_period3) ** 0.5
upper_bound_pair2 = (mean_injury_period2 - mean_injury_period4) + stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
lower_bound_pair2 = (mean_injury_period2 - mean_injury_period4) - stats.t.ppf(1 - alpha/2, dof_pair2) * pool_estimator * (1 / len_period2 + 1 / len_period4) ** 0.5
upper_bound_pair3 = (mean_injury_period3 - mean_injury_period4) + stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5
lower_bound_pair3 = (mean_injury_period3 - mean_injury_period4) - stats.t.ppf(1 - alpha/2, dof_pair3) * pool_estimator * (1 / len_period3 + 1 / len_period4) ** 0.5

print("2-3 p-value: ", p_value_of_pair1)
print("2-3 Upper bound pair1: ", upper_bound_pair1)
print("2-3 Lower bound pair1: ", lower_bound_pair1)

print("2-4 p-value: ", p_value_of_pair2)
print("2-4 Upper bound pair2: ", upper_bound_pair2)
print("2-4 Lower bound pair2: ", lower_bound_pair2)

print("3-4 p-value: ", p_value_of_pair3)
print("3-4 Upper bound pair3: ", upper_bound_pair3)
print("3-4 Lower bound pair3: ", lower_bound_pair3)

#---------------------Takedowns---------------------
plt.clf()
plt.boxplot([period1["取締件數每十萬人"], period2["取締件數每十萬人"], period3["取締件數每十萬人"], period4["取締件數每十萬人"]], tick_labels=["Period1", "Period2", "Period3", "Period4"])
plt.title("Mean Takedowns")
plt.xlabel("Period")
plt.ylabel("Mean")
plt.savefig("Mean Takedowns.png")
# plt.show()

#---------------------Sent to prison---------------------
plt.clf()
plt.boxplot([period1["移送法辦每十萬人"], period2["移送法辦每十萬人"], period3["移送法辦每十萬人"], period4["移送法辦每十萬人"]], tick_labels=["Period1", "Period2", "Period3", "Period4"])
plt.title("Mean Sent to Prison")
plt.xlabel("Period")
plt.ylabel("Mean")
plt.savefig("Mean Sent to Prison.png")
# plt.show()

#---------------------Accident---------------------
plt.clf()
plt.boxplot([period2["事故件數每十萬人"].dropna(), period3["事故件數每十萬人"], period4["事故件數每十萬人"]], tick_labels=["Period2", "Period3", "Period4"])
plt.title("Mean Accident")
plt.xlabel("Period")
plt.ylabel("Mean")
plt.savefig("Mean Accident.png")
# plt.show()

#---------------------Death---------------------
plt.clf()
plt.boxplot([period2["死亡人數每十萬人"].dropna(), period3["死亡人數每十萬人"], period4["死亡人數每十萬人"]], tick_labels=["Period2", "Period3", "Period4"])
plt.title("Mean Death")
plt.xlabel("Period")
plt.ylabel("Mean")
plt.savefig("Mean Death.png")
# plt.show()

#---------------------Injury---------------------
plt.clf()
plt.boxplot([period2["受傷人數每十萬人"].dropna(), period3["受傷人數每十萬人"], period4["受傷人數每十萬人"]], tick_labels=["Period2", "Period3", "Period4"])
plt.title("Mean Injury")
plt.xlabel("Period")
plt.ylabel("Mean")
plt.savefig("Mean Injury.png")
# plt.show()

#---------------------Relationship between takedowns and injury and death---------------------
x_axis = period_all["取締件數每十萬人"]
y_axis = period_all["受傷人數每十萬人"] + period_all["死亡人數每十萬人"]
data = pd.DataFrame({"Takedowns": x_axis, "Injury + Death": y_axis}).dropna()
# print(data)
x_axis = data["Takedowns"]
y_axis = data["Injury + Death"]
print(len(x_axis))
print(len(y_axis))

r, p_value = stats.pearsonr(x_axis, y_axis)
print("r: ", r)
print("p-value: ", p_value)
plt.clf()

plt.xlim(0, max(x_axis))
plt.ylim(0, max(y_axis))
plt.scatter(x_axis, y_axis)
plt.title("Relationship between Takedowns and Injury and Death")
plt.xlabel("Takedowns")
plt.ylabel("Injury + Death")
plt.savefig("Relationship between Takedowns and Injury and Death.png")
# plt.show()
plt.close()

#---------------------Hotmap---------------------

file_path = "事故次數每年.csv"
data = pd.read_csv(file_path)
# print(data.head(10))

map_path = "./country"
taiwan_map = gpd.read_file(map_path)
taiwan_map = taiwan_map.rename(columns={"COUNTYNAME": "縣市"})
# print(taiwan_map.head(10))

merged_data = pd.melt(data.reset_index(), id_vars=['年份'], var_name='縣市', value_name='事故數')
taiwan_map_merged = taiwan_map.merge(merged_data, on="縣市", how="left")
single_year = taiwan_map_merged[taiwan_map_merged['年份'] == 103]

axis = single_year.plot(column='事故數', cmap='Reds', legend=True, edgecolor='black')
cbar = axis.get_figure().get_axes()[1]
cbar.set_ylabel('hundred thousands people per year')
axis.set_xlim(119.5, 122)
axis.set_ylim(21.5, 25.5)
plt.title("103 drunk driving accidents hotmap")
# plt.show()
plt.close()

#---------------------Hotmap Gif---------------------

frames = []
years = taiwan_map_merged['年份'].unique()

for year in years:
    vmin = 0
    vmax = 2500
    single_year = taiwan_map_merged[taiwan_map_merged['年份'] == year]
    axis = single_year.plot(column='事故數', cmap='Reds', legend=True, vmin = vmin, vmax = vmax, edgecolor='black')
    cbar = axis.get_figure().get_axes()[1]
    cbar.set_ylabel('hundred thousands people per year')
    plt.title(f"{year} drunk driving accidents hotmap")
    plt.xlim(119.5, 122)
    plt.ylim(21.5, 25.5)
    plt.savefig(f"{year} drunk driving accidents hotmap.png")
    frames.append(imageio.v2.imread(f"{year} drunk driving accidents hotmap.png"))
    plt.close()

imageio.mimsave("drunk driving accidents hotmap.gif", frames, fps= 1.25)
