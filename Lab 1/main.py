import time
import random
import numpy as np
import matplotlib.pyplot as plt
from river import naive_bayes, tree, linear_model, metrics, preprocessing, neighbors
from incremental_elm import IncrementalELM


def random_walk_param(value, delta=0.01, min_val=0.0, max_val=1.0):
    new_value = value + np.random.uniform(-delta, delta)
    return min(max(new_value, min_val), max_val)

def data_stream_generator_with_concept_drift(total_samples=20000, drift_period=5000):
    # Початкові параметри
    distributions = [
        {"mu": 0.5, "sigma": 0.05},
        {"mu": 0.52, "sigma": 0.02},
        {"mu": 0.74, "sigma": 0.06},
        {"mu": 0.70, "sigma": 0.05},
        {"mu": 0.38, "sigma": 0.04},
    ]

    count = 0
    num_dists = len(distributions)

    while count < total_samples:
        # На початку кожного періоду — дрейф параметрів
        if count % drift_period == 0 and count > 0:
            for dist in distributions:
                dist["mu"] = random_walk_param(dist["mu"], delta=0.02, min_val=0.0, max_val=1.0)
                dist["sigma"] = random_walk_param(dist["sigma"], delta=0.005, min_val=0.001, max_val=0.1)

        for i, dist in enumerate(distributions, start=1):
            R_i = random.randint(-50, 50)
            T_i = 2000 + R_i
            for _ in range(T_i):
                if count >= total_samples:
                    return
                x = np.random.normal(loc=dist["mu"], scale=dist["sigma"])
                yield {"x": x}, i
                count += 1

def data_stream_generator_mixed_with_concept_drift(total_samples=20000, drift_period=5000):
    distributions = [
        {"type": "normal", "params": {"mu": 0.5118, "sigma": 0.0634}},
        {"type": "uniform", "params": {"low": 0.48, "high": 0.55}},
        {"type": "exponential", "params": {"scale": 1 / 0.7463}},
        {"type": "lognormal", "params": {"mean": np.log(0.7035), "sigma": 0.1}},
        {"type": "binomial", "params": {"n": 10, "p": 0.38}},
    ]

    count = 0

    while count < total_samples:
        # На початку кожного періоду — змінюємо параметри
        if count % drift_period == 0 and count > 0:
            for dist in distributions:
                if dist["type"] == "normal":
                    dist["params"]["mu"] = random_walk_param(dist["params"]["mu"], delta=0.02, min_val=0.0, max_val=1.0)
                    dist["params"]["sigma"] = random_walk_param(dist["params"]["sigma"], delta=0.005, min_val=0.001,
                                                                max_val=0.2)
                elif dist["type"] == "uniform":
                    low = dist["params"]["low"]
                    high = dist["params"]["high"]
                    dist["params"]["low"] = random_walk_param(low, delta=0.01, min_val=0.0, max_val=high - 0.01)
                    dist["params"]["high"] = random_walk_param(high, delta=0.01, min_val=dist["params"]["low"] + 0.01,
                                                               max_val=1.0)
                elif dist["type"] == "exponential":
                    dist["params"]["scale"] = random_walk_param(dist["params"]["scale"], delta=0.1, min_val=0.1,
                                                                max_val=5.0)
                elif dist["type"] == "lognormal":
                    dist["params"]["mean"] = random_walk_param(dist["params"]["mean"], delta=0.1)
                    dist["params"]["sigma"] = random_walk_param(dist["params"]["sigma"], delta=0.02, min_val=0.01,
                                                                max_val=1.0)
                elif dist["type"] == "binomial":
                    dist["params"]["n"] = max(1, int(round(
                        random_walk_param(dist["params"]["n"], delta=1, min_val=1, max_val=100))))
                    dist["params"]["p"] = random_walk_param(dist["params"]["p"], delta=0.02, min_val=0.0, max_val=1.0)

        for i, dist in enumerate(distributions, start=1):
            R_i = random.randint(-50, 50)
            T_i = 2000 + R_i

            for _ in range(T_i):
                if count >= total_samples:
                    return

                params = dist["params"]
                if dist["type"] == "normal":
                    x = np.random.normal(loc=params["mu"], scale=params["sigma"])
                elif dist["type"] == "uniform":
                    x = np.random.uniform(low=params["low"], high=params["high"])
                elif dist["type"] == "exponential":
                    x = np.random.exponential(scale=params["scale"])
                elif dist["type"] == "lognormal":
                    x = np.random.lognormal(mean=params["mean"], sigma=params["sigma"])
                elif dist["type"] == "binomial":
                    x = np.random.binomial(n=int(round(params["n"])), p=params["p"])
                else:
                    x = 0

                yield {"x": x}, i
                count += 1

def init_models():
    models = {
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "SGD Perceptron": preprocessing.StandardScaler() | linear_model.PAClassifier(mode=1, C=1.0),
        "IELM": IncrementalELM(n_hidden=30, random_state=42),
        "KNN": neighbors.KNNClassifier(n_neighbors=5)
    }
    return models

def train_and_evaluate(models, total_samples=20000, eval_step=500, dsgn_type="normal"):
    accuracies = {name: [] for name in models}
    metrics_track = {name: metrics.Accuracy() for name in models if name != "IELM"}
    times = {name: 0.0 for name in models}  # для збереження часу навчання

    if dsgn_type == "mixed":
        stream = data_stream_generator_mixed_with_concept_drift(total_samples=total_samples)
    else:
        stream = data_stream_generator_with_concept_drift(total_samples=total_samples)

    for i, (x, y) in enumerate(stream, start=1):
        for name, model in models.items():
            start_time = time.time()
            if name == "IELM":
                model.partial_fit([x["x"]], y)
                y_pred = model.predict([x["x"]])[0]
                # збираємо час навчання
                times[name] += time.time() - start_time

                if name not in accuracies:
                    accuracies[name] = []
                if 'ielm_true' not in locals():
                    ielm_true = []
                    ielm_pred = []
                ielm_true.append(y)
                ielm_pred.append(y_pred)
                if i % eval_step == 0:
                    acc = np.mean(np.array(ielm_true) == np.array(ielm_pred))
                    accuracies[name].append(acc)
                    ielm_true, ielm_pred = [], []
            else:
                y_pred = model.predict_one(x)
                model.learn_one(x, y)
                times[name] += time.time() - start_time
                if y_pred is not None:
                    metrics_track[name].update(y, y_pred)
                if i % eval_step == 0:
                    accuracies[name].append(metrics_track[name].get())

    return accuracies, times, eval_step

def plot_accuracies(accuracies, eval_step, names="normal"):
    plt.figure(figsize=(12, 6))
    for name, acc in accuracies.items():
        plt.plot(np.arange(eval_step, eval_step * (len(acc) + 1), eval_step), acc, label=name)
    plt.title(f"Точність класифікації на потоці даних {names}")
    plt.xlabel("Кількість зразків")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    models = init_models()
    accuracies, times, step = train_and_evaluate(models, dsgn_type="normal")
    print("Час навчання моделей на нормальному потоці:")
    for name, t in times.items():
        print(f"{name}: {t:.4f} секунд")
    plot_accuracies(accuracies, step, names='normal')

    accuracies, times, step = train_and_evaluate(models, dsgn_type="mixed")
    print("Час навчання моделей на змішаному потоці:")
    for name, t in times.items():
        print(f"{name}: {t:.4f} секунд")
    plot_accuracies(accuracies, step, names='mixed')