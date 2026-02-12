import csv
import math
def read_csv(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data
def entropy(col):
    counts = {}
    for val in col:
        counts[val] = counts.get(val, 0) + 1
    ent = 0
    for count in counts.values():
        p = count / len(col)
        ent -= p * math.log2(p)
    return ent
def gain_ratio(data, col_idx):
    target_col = [row[-1] for row in data]   # Always last column
    total_entropy = entropy(target_col)
    vals = set(row[col_idx] for row in data)
    weighted_entropy = 0
    split_info = 0
    for v in vals:
        subset = [row for row in data if row[col_idx] == v]
        ratio = len(subset) / len(data)
        subset_target = [row[-1] for row in subset]
        weighted_entropy += ratio * entropy(subset_target)
        split_info -= ratio * math.log2(ratio)
    if split_info == 0:
        return 0
    info_gain = total_entropy - weighted_entropy
    return info_gain / split_info
def build_tree(data, attributes):
    target_col = [row[-1] for row in data]
    if target_col.count(target_col[0]) == len(target_col):
        return target_col[0]
    if not attributes:
        return max(set(target_col), key=target_col.count)
    gains = [gain_ratio(data, i) for i in range(len(attributes))]
    best_idx = gains.index(max(gains))
    best_attr = attributes[best_idx]
    tree = {best_attr: {}}
    values = set(row[best_idx] for row in data)
    for v in values:
        subset = []
        for row in data:
            if row[best_idx] == v:
                # Remove best attribute column
                new_row = row[:best_idx] + row[best_idx+1:]
                subset.append(new_row)
        new_attributes = attributes[:best_idx] + attributes[best_idx+1:]
        if not subset:
            tree[best_attr][v] = max(set(target_col), key=target_col.count)
        else:
            tree[best_attr][v] = build_tree(subset, new_attributes)

    return tree
def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key in tree:
            print(indent + str(key))
            for value in tree[key]:
                print(indent + "  ->", value)
                print_tree(tree[key][value], indent + "    ")
    else:
        print(indent + "Answer:", tree)
filename = "tennis.csv"
attributes, data = read_csv(filename)
tree_c45 = build_tree(data, attributes[:-1])
print("\n--- C4.5 Decision Tree ---")
print_tree(tree_c45)
