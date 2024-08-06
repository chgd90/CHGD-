import streamlit as st
import numpy as np
from scipy.optimize import linprog
from deap import base, creator, tools, algorithms
import random
from pyswarm import pso
import datetime
import holidays

# 获取2024年中国的所有假日
china_holidays = holidays.China(years=2024)

# 固定的人力和培训天数
fixed_H1 = 38
fixed_H2 = 8
fixed_H3 = 10
fixed_H4 = 3

training_days = {'H1': 2, 'H2': 5, 'H3': 0, 'H4': 2}

# 生产线和包装线的产量
assy_production = 6000
pack_production = 8000

# 人力成本
cost_per_day_H1 = 100
cost_per_day_H2 = 140  # 1.4倍
cost_per_day_H3 = 100
cost_per_day_H4 = 140  # 1.4倍

# 每周工作天数
work_days_per_week = 7
# 定义线性规划求解函数
def linear_programming(demand, assy_lines, pack_lines):
    # 固定的生产线和包装线人数
    H1 = fixed_H1 * assy_lines
    H2 = fixed_H2 * assy_lines
    H3 = fixed_H3 * pack_lines
    H4 = fixed_H4 * pack_lines

    daily_cost = H1 * cost_per_day_H1 + H2 * cost_per_day_H2 + H3 * cost_per_day_H3 + H4 * cost_per_day_H4
    total_cost = daily_cost * work_days_per_week

    return H1, H2, H3, H4, total_cost
# 定义遗传算法求解函数
def genetic_algorithm(demand, assy_lines, pack_lines):
    def evalCost(individual):
        assy_lines, pack_lines = individual
        H1 = fixed_H1 * assy_lines
        H2 = fixed_H2 * assy_lines
        H3 = fixed_H3 * pack_lines
        H4 = fixed_H4 * pack_lines

        daily_cost = H1 * cost_per_day_H1 + H2 * cost_per_day_H2 + H3 * cost_per_day_H3 + H4 * cost_per_day_H4
        total_cost = daily_cost * work_days_per_week
        return total_cost,

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 10, 20)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalCost)

    population = toolbox.population(n=50)
    NGEN = 40
    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    assy_lines, pack_lines = best_ind[0], best_ind[1]
    H1 = fixed_H1 * assy_lines
    H2 = fixed_H2 * assy_lines
    H3 = fixed_H3 * pack_lines
    H4 = fixed_H4 * pack_lines
    daily_cost = H1 * cost_per_day_H1 + H2 * cost_per_day_H2 + H3 * cost_per_day_H3 + H4 * cost_per_day_H4
    total_cost = daily_cost * work_days_per_week
    return H1, H2, H3, H4, total_cost
# 定义粒子群算法求解函数
def particle_swarm(demand, assy_lines, pack_lines):
    def cost_function(x):
        assy_lines, pack_lines = x
        H1 = fixed_H1 * assy_lines
        H2 = fixed_H2 * assy_lines
        H3 = fixed_H3 * pack_lines
        H4 = fixed_H4 * pack_lines

        daily_cost = H1 * cost_per_day_H1 + H2 * cost_per_day_H2 + H3 * cost_per_day_H3 + H4 * cost_per_day_H4
        total_cost = daily_cost * work_days_per_week
        return total_cost

    lb = [10, 6]
    ub = [20, 12]

    xopt, fopt = pso(cost_function, lb, ub)
    assy_lines, pack_lines = int(xopt[0]), int(xopt[1])
    H1 = fixed_H1 * assy_lines
    H2 = fixed_H2 * assy_lines
    H3 = fixed_H3 * pack_lines
    H4 = fixed_H4 * pack_lines
    daily_cost = H1 * cost_per_day_H1 + H2 * cost_per_day_H2 + H3 * cost_per_day_H3 + H4 * cost_per_day_H4
    total_cost = daily_cost * work_days_per_week
    return H1, H2, H3, H4, total_cost
# 计算并输出结果
def calculate_and_display(demands, algorithm):
    assy_lines = 10
    pack_lines = 6

    total_training_cost = 0

    for week, demand in enumerate(demands):
        if algorithm == "Linear Programming":
            H1, H2, H3, H4, total_cost = linear_programming(demand, assy_lines, pack_lines)
        elif algorithm == "Genetic Algorithm":
            H1, H2, H3, H4, total_cost = genetic_algorithm(demand, assy_lines, pack_lines)
        elif algorithm == "Particle Swarm Optimization":
            H1, H2, H3, H4, total_cost = particle_swarm(demand, assy_lines, pack_lines)

        st.subheader(f"Week {week+1}")
        st.write(f"每日排班人数: H1: {fixed_H1} 人, H2: {fixed_H2} 人, H3: {fixed_H3} 人, H4: {fixed_H4} 人")

        # 按一周的生产量计算
        total_weekly_assy_production = assy_lines * assy_production * work_days_per_week
        total_weekly_pack_production = pack_lines * pack_production * work_days_per_week

        # 判断是否需要增加产能
        if total_weekly_assy_production < demand or total_weekly_pack_production < demand:
            required_assy_lines = int(np.ceil(demand / (assy_production * work_days_per_week))) - assy_lines
            required_pack_lines = int(np.ceil(demand / (pack_production * work_days_per_week))) - pack_lines
            st.write(f"建议增加: {required_assy_lines} 条ASSY线, {required_pack_lines} 条PACK线")

            # 计算新增生产线的培训成本
            training_cost = (
                required_assy_lines * (fixed_H1 * training_days['H1'] * cost_per_day_H1 + fixed_H2 * training_days['H2'] * cost_per_day_H2)
                + required_pack_lines * (fixed_H3 * training_days['H3'] * cost_per_day_H3 + fixed_H4 * training_days['H4'] * cost_per_day_H4)
            )
            st.write(f"建议提前训练: H1: {fixed_H1 * required_assy_lines} 人, H2: {fixed_H2 * required_assy_lines} 人, H3: {fixed_H3 * required_pack_lines} 人, H4: {fixed_H4 * required_pack_lines} 人")

            # 更新生产线数量
            assy_lines += required_assy_lines
            pack_lines += required_pack_lines

            # 计算新增的总培训成本
            total_training_cost += training_cost

        # 计算当前周的总成本
        current_total_cost = (
                (assy_lines * (fixed_H1 * cost_per_day_H1 + fixed_H2 * cost_per_day_H2) +
                 pack_lines * (fixed_H3 * cost_per_day_H3 + fixed_H4 * cost_per_day_H4)) *
                work_days_per_week
        )

        st.write(f"预估成本: {current_total_cost:.2f} 人天金额")

# Streamlit 应用界面
st.title("生产需求与排班优化")

# 计算2024年从1月1日开始的周数
start_date = datetime.date(2024, 1, 1)
week_number = [(start_date + datetime.timedelta(weeks=i)).strftime("W%U") for i in range(20)]

# 输入需求产量
st.header("输入每周需求产量")
demands = []
for week in week_number:
    demand = st.number_input(f"{week} 需求产量", min_value=0, step=1000)
    demands.append(demand)

# 选择算法
algorithm = st.selectbox("选择算法", ["Linear Programming", "Genetic Algorithm", "Particle Swarm Optimization"])

# 计算结果
st.header("优化结果")

if st.button("计算"):
    calculate_and_display(demands, algorithm)
