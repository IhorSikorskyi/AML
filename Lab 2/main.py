import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling
from modAL.disagreement import vote_entropy_sampling

def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def initialize_pools(X_train, y_train, n_initial=100):
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)
    return X_initial, y_initial, X_pool, y_pool

def create_active_learner(X_initial, y_initial):
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling,
        X_training=X_initial,
        y_training=y_initial
    )
    return learner

def create_committee(X_initial, y_initial, n_members=3):
    members = []
    for _ in range(n_members):
        member = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=entropy_sampling,
            X_training=X_initial,
            y_training=y_initial
        )
        members.append(member)
    committee = Committee(learner_list=members, query_strategy=vote_entropy_sampling)
    return committee

def query_and_label_automatic_batch(learner_entropy, committee, X_pool, y_pool, batch_size=10):
    query_idx_entropy, query_instance_entropy = learner_entropy.query(X_pool, n_instances=batch_size)
    query_idx_committee, query_instance_committee = committee.query(X_pool, n_instances=batch_size)

    user_labels_entropy = y_pool[query_idx_entropy]
    user_labels_committee = y_pool[query_idx_committee]

    image = query_instance_entropy[0].reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.title(
        f"E:{learner_entropy.predict(query_instance_entropy)[0]}, C:{committee.predict(query_instance_committee)[0]}")
    plt.axis('off')
    plt.show()

    return query_idx_entropy, query_instance_entropy, user_labels_entropy, query_idx_committee, query_instance_committee, user_labels_committee

def active_learning_loop_batch(n_queries, learner_entropy, committee, X_pool, y_pool, X_test, y_test, batch_size=9):
    acc_entropy_list = []
    acc_committee_list = []

    for idx in range(n_queries):
        print(f"\n=== Query {idx + 1} ===")

        (query_idx_entropy, query_instance_entropy, user_labels_entropy,
         query_idx_committee, query_instance_committee, user_labels_committee) = query_and_label_automatic_batch(
            learner_entropy, committee, X_pool, y_pool, batch_size)

        # Навчання моделей на нових даних (пакетом)
        learner_entropy.teach(query_instance_entropy, user_labels_entropy)
        committee.teach(query_instance_committee, user_labels_committee)

        # Оновлення пулу непозначених даних (видаляємо весь пакет, унікальні індекси)
        remove_idx = np.union1d(query_idx_entropy, query_idx_committee)
        X_pool = np.delete(X_pool, remove_idx, axis=0)
        y_pool = np.delete(y_pool, remove_idx, axis=0)

        # Обчислення точності моделей
        acc_entropy = accuracy_score(y_test, learner_entropy.predict(X_test))
        acc_committee = accuracy_score(y_test, committee.predict(X_test))
        acc_entropy_list.append(acc_entropy)
        acc_committee_list.append(acc_committee)

    print("\n=== Підсумкова середня точність ===")
    print(f"Entropy Learner: {np.mean(acc_entropy_list):.4f}")
    print(f"Committee: {np.mean(acc_committee_list):.4f}")

    # Побудова графіка точності
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_queries + 1), acc_entropy_list, label='Entropy Learner')
    plt.plot(range(1, n_queries + 1), acc_committee_list, label='Committee')
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Comparison of Accuracy during Active Learning')
    plt.legend()
    plt.grid(True)
    plt.show()

    return learner_entropy, committee

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    X_initial, y_initial, X_pool, y_pool = initialize_pools(X_train, y_train)

    learner_entropy = create_active_learner(X_initial, y_initial)
    committee = create_committee(X_initial, y_initial)

    active_learning_loop_batch(15, learner_entropy, committee, X_pool, y_pool, X_test, y_test)