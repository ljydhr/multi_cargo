import csv
import random
import math


# 定义基础数据结构
class Cargo:
    def __init__(self, id, weight, start_city, end_city):
        self.id = id
        self.weight = weight
        self.start_city = start_city
        self.end_city = end_city


class Vehicle:
    def __init__(self, id, capacity, mode):
        self.id = id
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.mode = mode  # 'air' 或 'rail'
        self.path = []


class Individual:
    def __init__(self, path):
        self.path = path
        self.has_exchanged = False
        self.objectives = self.evaluate()
        self.rank = None
        self.distance = 0

    def evaluate(self):
        total_distance, total_cost, total_time, total_co2_emission, _, _, _, _, _, _, _, _ = calculate_total_distance_cost_and_time(
            self.path)
        return [total_cost, total_time, total_co2_emission]

    def update_has_exchanged(self, exchanged):
        self.has_exchanged = exchanged


class Sparrow:
    def __init__(self, city_list):
        self.path = city_list
        self.has_exchanged = False
        self.objectives = self.evaluate()
        self.rank = None
        self.distance = 0

    def evaluate(self):
        total_distance, total_cost, total_time, total_co2_emission, _, _, _, _, _, _, _, _ = calculate_total_distance_cost_and_time(
            self.path)
        return [total_cost, total_time, total_co2_emission]

    def update_has_exchanged(self, exchanged):
        self.has_exchanged = exchanged


# 从CSV文件加载数据
def load_cities(file_path):
    cities = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            city_name = row[0]
            coordinates = (float(row[1]), float(row[2]))
            cities[city_name] = coordinates
    return cities


def load_cargoes(file_path):
    cargoes = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            cargo = Cargo(int(row[0]), float(row[1]), row[2], row[3])
            cargoes.append(cargo)
    return cargoes


def load_vehicles(file_path):
    vehicles = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            vehicle = Vehicle(int(row[0]), float(row[1]), row[2])
            vehicles.append(vehicle)
    return vehicles


# 打印路径进行调试
cities_file_path = 'C:/Users/12082/Downloads/cities.csv'
cargoes_file_path = 'C:/Users/12082/Downloads/cargoes.csv'
vehicles_file_path = 'C:/Users/12082/Downloads/vehicles.csv'

print(f"Loading cities from: {cities_file_path}")
print(f"Loading cargoes from: {cargoes_file_path}")
print(f"Loading vehicles from: {vehicles_file_path}")

# 加载数据
cities = load_cities(cities_file_path)
cargoes = load_cargoes(cargoes_file_path)
vehicles = load_vehicles(vehicles_file_path)

# 示例运输距离数据（此部分需要根据具体需求修改或从CSV文件中加载）
transport_distances = {
    ("city1", "city2"): {"air": 200, "rail": 300},
    ("city1", "city3"): {"air": 400, "rail": 500},
    # 添加更多城市间的距离...
}


# 计算路径总距离、成本、时间和CO2排放量的函数（示例函数，需要根据实际情况实现）
def calculate_total_distance_cost_and_time(path):
    total_distance = 0
    total_cost = 0
    total_time = 0
    total_co2_emission = 0
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        transport = transport_distances.get((city1, city2)) or transport_distances.get((city2, city1))
        if transport:
            mode = 'air' if 'air' in transport else 'rail'
            distance = transport[mode]
            total_distance += distance
            total_cost += distance * 0.1  # 假设每公里成本为0.1单位
            total_time += distance / (800 if mode == "air" else 100)  # 假设飞机速度800km/h，火车100km/h
            total_co2_emission += distance * 0.05  # 假设每公里CO2排放为0.05单位
    return total_distance, total_cost, total_time, total_co2_emission, 0, 0, 0, 0, 0, 0, 0, 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_transport_info(city1, city2, mode):
    if (city1, city2) in transport_distances:
        distances = transport_distances[(city1, city2)]
    elif (city2, city1) in transport_distances:
        distances = transport_distances[(city2, city1)]
    else:
        return None, None

    if mode in distances:
        return mode, distances[mode]
    else:
        return None, None


# 评估个体路径的函数
def evaluate_individual(path, time_weight=1.0, cost_weight=1.0, co2_emission_weight=1.0):
    total_distance, total_cost, total_time, total_co2_emission, _, _, _, _, _, _, _, _ = calculate_total_distance_cost_and_time(
        path)
    normalized_total_time = sigmoid(total_time)
    normalized_total_cost = sigmoid(total_cost)
    normalized_total_co2_emission = sigmoid(total_co2_emission)
    score = normalized_total_time + normalized_total_cost + normalized_total_co2_emission
    return score


# 生成单个货物的路径
def generate_individual(cargo, vehicle):
    best_path = None
    best_score = float('inf')
    num_candidates = 15

    for _ in range(num_candidates):
        path = [cargo.start_city]
        current_city = cargo.start_city
        visited_cities = set([current_city])

        while current_city != cargo.end_city:
            next_city_candidates = [
                city for city in cities.keys()
                if city not in visited_cities
                   and ((current_city, city) in transport_distances or (city, current_city) in transport_distances)
                   and (get_transport_info(current_city, city, vehicle.mode)[0] == vehicle.mode)
            ]

            if not next_city_candidates:
                break

            next_city = random.choice(next_city_candidates)
            path.append(next_city)
            visited_cities.add(next_city)
            current_city = next_city

        if path[-1] == cargo.end_city:
            score = evaluate_individual(path, time_weight=1.0, cost_weight=1.0, co2_emission_weight=1.0)
            if score < best_score:
                best_path = path
                best_score = score

    return best_path


# 分配货物给车辆
def assign_cargo_to_vehicle(cargo):
    for vehicle in vehicles:
        if vehicle.remaining_capacity >= cargo.weight:
            path = generate_individual(cargo, vehicle)
            if path:
                vehicle.path.append((cargo.id, path))
                vehicle.remaining_capacity -= cargo.weight
                return True
    return False


# 多货物优化函数
def optimize_multiple_cargoes():
    sorted_cargoes = sorted(cargoes, key=lambda x: x.weight, reverse=True)
    vehicle_pool = vehicles[:]

    for cargo in sorted_cargoes:
        assigned = False
        for vehicle in vehicle_pool:
            if vehicle.remaining_capacity >= cargo.weight:
                path = generate_individual(cargo, vehicle)
                if path:
                    vehicle.path.append((cargo.id, path))
                    vehicle.remaining_capacity -= cargo.weight
                    assigned = True
                    break
        if not assigned:
            print(f"货物 {cargo.id} 无法分配到任何车辆中")

    for vehicle in vehicle_pool:
        print(f"车辆 {vehicle.id} (模式: {vehicle.mode}) 的路径: {vehicle.path}，剩余容量: {vehicle.remaining_capacity}")


def simulated_annealing(initial_solution, initial_temperature, cooling_rate, iterations):
    current_solution = initial_solution
    best_solution = current_solution
    temperature = initial_temperature

    for i in range(iterations):
        new_solution = generate_neighbor_solution(current_solution)
        current_energy = evaluate_solution(current_solution)
        new_energy = evaluate_solution(new_solution)

        if accept_solution(current_energy, new_energy, temperature):
            current_solution = new_solution

        if new_energy < evaluate_solution(best_solution):
            best_solution = new_solution

        temperature *= cooling_rate

    return best_solution


def generate_neighbor_solution(solution):
    new_solution = []
    for vehicle in solution:
        new_vehicle = Vehicle(vehicle.id, vehicle.capacity, vehicle.mode)
        new_vehicle.remaining_capacity = vehicle.remaining_capacity
        new_vehicle.path = vehicle.path.copy()
        new_solution.append(new_vehicle)

    # 随机选择一个车辆和一个货物，尝试调整货物路径
    vehicle_index = random.randint(0, len(new_solution) - 1)
    if new_solution[vehicle_index].path:
        cargo_index = random.randint(0, len(new_solution[vehicle_index].path) - 1)
        cargo_id, path = new_solution[vehicle_index].path[cargo_index]

        # 尝试生成一个新的路径
        cargo = next(c for c in cargoes if c.id == cargo_id)
        new_path = generate_individual(cargo, new_solution[vehicle_index])
        if new_path:
            new_solution[vehicle_index].path[cargo_index] = (cargo_id, new_path)

    return new_solution


def evaluate_solution(solution):
    total_cost = 0
    total_time = 0
    total_co2_emission = 0

    for vehicle in solution:
        for cargo_id, path in vehicle.path:
            _, cost, time, co2_emission, _, _, _, _, _, _, _, _ = calculate_total_distance_cost_and_time(path)
            total_cost += cost
            total_time += time
            total_co2_emission += co2_emission

    normalized_total_time = sigmoid(total_time)
    normalized_total_cost = sigmoid(total_cost)
    normalized_total_co2_emission = sigmoid(total_co2_emission)

    score = normalized_total_time + normalized_total_cost + normalized_total_co2_emission
    return score


def accept_solution(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return True
    else:
        acceptance_probability = math.exp((current_energy - new_energy) / temperature)
        return acceptance_probability > random.random()


# 主优化函数
def optimize_paths():
    initial_solution = vehicles
    best_solution = simulated_annealing(initial_solution, initial_temperature=100, cooling_rate=0.95, iterations=1000)
    print("优化后的路径:", best_solution)


if __name__ == "__main__":
    optimize_multiple_cargoes()
    optimize_paths()
