import re
import numpy as np
import pandas as pd


def parse_result(text):
    # Updated regex to extract S between D and L.
    pattern = re.compile(
        r"E\s*=\s*(\[[\s\S]*?\])\s*,\s*D\s*=\s*(\[[\s\S]*?\])\s*,\s*S\s*=\s*([\d\.\-Ee]+)\s*,\s*L\s*=\s*([\d\.\-Ee]+)"
        r"\s*,\s*M\s*=\s*([\d\.\-Ee]+)\s*,\s*L \+ M\s*=\s*([\d\.\-Ee]+)",
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        E_str, D_str, S_str, L_str, M_str, LM_str = m.groups()

        def process_array(arr_str):
            # Remove outer brackets.
            arr_str = arr_str.strip()
            if arr_str.startswith("[") and arr_str.endswith("]"):
                arr_str = arr_str[1:-1]
            # Split into rows (assuming rows are separated by newlines);
            # note: we ignore empty rows in case of extra blank lines.
            rows = [row for row in arr_str.strip().split("\n") if row.strip() != ""]
            processed_rows = []
            for row in rows:
                row = row.strip()
                # Remove any remaining brackets.
                if row.startswith("[") and row.endswith("]"):
                    row = row[1:-1]
                # Convert whitespace-separated numbers into floats.
                row_vals = [float(x) for x in row.split()]
                processed_rows.append(row_vals)
            return np.array(processed_rows)

        E = process_array(E_str)
        D = process_array(D_str)
        S = float(S_str)
        L = float(L_str)
        M = float(M_str)
        LM = float(LM_str)
        return {"E": E, "D": D, "S": S, "L": L, "M": M, "LM": LM}
    else:
        return None


def load_results_from_file(filename):
    """
    Reads the file and groups lines into individual simulation results by detecting the start
    of a new result (e.g. when a line starts with 'E ='). This avoids splitting the multi-line
    arrays into several groups.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    groups = []
    current_group = []
    for line in lines:
        # If a line starts with "E =", assume it's the beginning of a new simulation result.
        if line.strip().startswith("E =") and current_group:
            # Append the previous group and start a new one.
            groups.append("".join(current_group).strip())
            current_group = []
        current_group.append(line)
    if current_group:
        groups.append("".join(current_group).strip())

    # Parse each group; only keep those that parse correctly.
    results = [parse_result(group) for group in groups if parse_result(group) is not None]
    return results


def extract_features_from_result(res, sigma):
    """
    Builds a flat dictionary of features including the arrays E and D (flattened),
    and the constants S, L, M, LM. The sigma flag controls inclusion of S.
    """
    features = {}
    E = res["E"]
    D = res["D"]
    if sigma:
        features["S"] = res["S"]
    features["L"] = res["L"]
    features["M"] = res["M"]
    features["LM"] = res["LM"]

    # Flatten E and D and add each element as a feature.
    E_flat = E.flatten()
    D_flat = D.flatten()
    for i, val in enumerate(E_flat):
        features[f"E_{i}"] = val
    for i, val in enumerate(D_flat):
        features[f"D_{i}"] = val

    return features


def build_dataframe_from_file(filename, sigma):
    """
    Loads the simulation results from a file and builds a pandas DataFrame.
    The parameter `sigma` indicates whether to include S in the features.
    """
    results = load_results_from_file(filename)
    feature_list = [extract_features_from_result(r, sigma) for r in results]
    df = pd.DataFrame(feature_list)
    return df

def load_csv_and_prune(filename, to_drop):

    df = pd.read_csv(filename)
    # drop the known / unwanted cols

    df = df.drop(columns=to_drop, errors="ignore")
    shuffle = True
    if shuffle:
        # shuffle all rows, then reset the index
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    return df
# Example usage:
if __name__ == "__main__":
    # For instance, if your result file has sigma information and you want to include it
    filename = "/path/to/your/results.txt"
    df = build_dataframe_from_file(filename, sigma=True)
    print(df)