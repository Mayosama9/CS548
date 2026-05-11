import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv(output_dir):
    metrics_dir = os.path.join(output_dir, "metrics")

    files = ["accuracy.csv", "loss.csv", "mse.csv"]

    for file in files:
        path = os.path.join(metrics_dir, file)

        if not os.path.exists(path) or os.stat(path).st_size == 0:
            continue

        df = pd.read_csv(path)

        x = list(range(1, 1 + len(df)))
        y = df.iloc[:, 1]

        plt.figure()
        plt.plot(x, y)

        plt.title(file.replace(".csv", "").capitalize())
        plt.xlabel("Round")
        plt.ylabel(file.replace(".csv", "").capitalize())

        save_path = os.path.join(output_dir, "plots", file.replace(".csv", ".png"))
        plt.savefig(save_path)
        plt.close()