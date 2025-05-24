import time
import random
import numpy as np
import matplotlib.pyplot as plt
from river import naive_bayes, tree, linear_model, metrics, preprocessing, neighbors
from incremental_elm import IncrementalELM


def smooth_transition(val1, val2, alpha):
    return (1 - alpha) * val1 + alpha * val2

def data_stream_generator_with_concept_drift(total_samples=20000, drift_period=5000):
    distributions_epoch_1 = [
        {"mu": 0.5118, "sigma": 0.0634},
        {"mu": 0.5147, "sigma": 0.0134},
        {"mu": 0.7463, "sigma": 0.0600},
        {"mu": 0.7035, "sigma": 0.0592},
        {"mu": 0.3849, "sigma": 0.0357},
    ]
    distributions_epoch_2 = [
        {"mu": 0.5188, "sigma": 0.0579},
        {"mu": 0.4015, "sigma": 0.0513},
        {"mu": 0.5203, "sigma": 0.0012},
        {"mu": 0.4533, "sigma": 0.0489},
        {"mu": 0.3843, "sigma": 0.0527},
    ]

    count = 0
    num_dists = len(distributions_epoch_1)

    while count < total_samples:
        # Визначаємо позицію у "циклі дрейфу"
        drift_cycle_pos = (count % drift_period) / drift_period  # 0..1, позиція в періоді

        # Визначаємо, чи це фаза переходу (приблизно останні 20% періоду)
        transition_start = 0.8

        if drift_cycle_pos < transition_start:
            # Поки що "концепт 1"
            alpha = 0.0
        else:
            # Плавний перехід від 0 до 1
            alpha = (drift_cycle_pos - transition_start) / (1 - transition_start)

        # Для кожного дистрибутиву генеруємо дані, чергуючи їх по черзі
        for i in range(num_dists):
            # Інтерполяція параметрів mu і sigma між двома концептами
            mu = smooth_transition(distributions_epoch_1[i]["mu"], distributions_epoch_2[i]["mu"], alpha)
            sigma = smooth_transition(distributions_epoch_1[i]["sigma"], distributions_epoch_2[i]["sigma"], alpha)

            R_i = random.randint(-50, 50)
            T_i = 2000 + R_i

            for _ in range(T_i):
                if count >= total_samples:
                    return
                x = np.random.normal(loc=mu, scale=sigma)
                yield {"x": x}, i + 1
                count += 1

def data_stream_generator_mixed_with_concept_drift(total_samples=20000, drift_period=5000):
    distributions_epoch_1 = [
        {"type": "normal", "params": {"mu": 0.5118, "sigma": 0.0634}},
        {"type": "uniform", "params": {"low": 0.48, "high": 0.55}},
        {"type": "exponential", "params": {"scale": 1 / 0.7463}},
        {"type": "lognormal", "params": {"mean": np.log(0.7035), "sigma": 0.1}},
        {"type": "binomial", "params": {"n": 10, "p": 0.38}},
    ]

    distributions_epoch_2 = [
        {"type": "normal", "params": {"mu": 0.5188, "sigma": 0.0579}},
        {"type": "uniform", "params": {"low": 0.4, "high": 0.6}},
        {"type": "exponential", "params": {"scale": 1.0}},
        {"type": "lognormal", "params": {"mean": 0.0, "sigma": 0.25}},
        {"type": "binomial", "params": {"n": 10, "p": 0.5}},
    ]

    count = 0
    num_dists = len(distributions_epoch_1)

    while count < total_samples:
        drift_cycle_pos = (count % drift_period) / drift_period
        transition_start = 0.8
        alpha = 0.0 if drift_cycle_pos < transition_start else (drift_cycle_pos - transition_start) / (
                    1 - transition_start)

        for i in range(num_dists):
            dist1 = distributions_epoch_1[i]
            dist2 = distributions_epoch_2[i]

            # Для параметрів, які є числами, робимо інтерполяцію, інакше беремо dist1 або dist2 залежно від alpha
            def interp_param(key):
                val1 = dist1["params"][key]
                val2 = dist2["params"][key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    return smooth_transition(val1, val2, alpha)
                return val1 if alpha < 0.5 else val2

            R_i = random.randint(-50, 50)
            T_i = 2000 + R_i

            for _ in range(T_i):
                if count >= total_samples:
                    return

                dist_type = dist1["type"] if alpha < 0.5 else dist2["type"]

                if dist_type == "normal":
                    mu = interp_param("mu")
                    sigma = interp_param("sigma")
                    x = np.random.normal(loc=mu, scale=sigma)
                elif dist_type == "uniform":
                    low = interp_param("low")
                    high = interp_param("high")
                    x = np.random.uniform(low=low, high=high)
                elif dist_type == "exponential":
                    scale = interp_param("scale")
                    x = np.random.exponential(scale=scale)
                elif dist_type == "lognormal":
                    mean = interp_param("mean")
                    sigma = interp_param("sigma")
                    x = np.random.lognormal(mean=mean, sigma=sigma)
                elif dist_type == "binomial":
                    n = interp_param("n")
                    p = interp_param("p")
                    # n та p мають бути цілими/плаваючими - округлюємо n для безпеки
                    x = np.random.binomial(n=int(round(n)), p=p)
                else:
                    x = 0

                yield {"x": x}, i + 1
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