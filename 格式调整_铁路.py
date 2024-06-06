import pandas as pd

# Create a DataFrame to hold train data
train_data = {
    "train_id": [],
    "start_city": [],
    "end_city": [],
    "start_time": [],
    "end_time": [],
    "capacity": []
}


# Define a function to parse data files and populate the DataFrame
def parse_train_data(file_path, capacity=20):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("  ")
            if len(parts) >= 5:
                start_time, start_city, duration, train_id, end_time, end_city = parts[0], parts[1], parts[2], parts[3], \
                parts[4], parts[-1]
                train_data["train_id"].append(train_id)
                train_data["start_city"].append(start_city)
                train_data["end_city"].append(end_city)
                train_data["start_time"].append(start_time)
                train_data["end_time"].append(end_time)
                train_data["capacity"].append(capacity)


# Parse each provided file
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-chongqing.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-chengdu.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/wuhan-shanghai.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-wuhan.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-changsha.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-nanchang.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-hefei.txt")
parse_train_data("C:/Users/12082/Downloads/ceshi/shanghai-hangzhou.txt")

# Create DataFrame
df = pd.DataFrame(train_data)

# Save to CSV file
output_path = "C:/Users/12082/Downloads/ceshi/trains.csv"
df.to_csv(output_path, index=False)

print(f"Train data saved to {output_path}")
