import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
np.random.seed(42)
N = 5000
t = np.linspace(0, 1, N)

def generate_am():
    fc = np.random.uniform(50, 200)
    fm = np.random.uniform(1, fc / 10)
    m = np.random.uniform(0.05, 0.6)
    s = (1 + m * np.sin(2 * np.pi * fm * t)) * np.sin(2 * np.pi * fc * t)
    return s

def generate_fm():
    A = np.random.uniform(0.3, 1.2)
    fc = np.random.uniform(50, 200)
    fm = np.random.uniform(1, 10)
    beta = np.random.uniform(0.1, 0.8)
    s = A * np.sin(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t))
    return s

def generate_lfm():
    A = np.random.uniform(0.5, 1.5)
    f0 = np.random.uniform(100, 300)
    B = np.random.uniform(40, 140)
    s = A * np.sin(2 * np.pi * (f0 * t + 0.5 * B * t**2))
    return s

def add_noise(signal, snr_db):
    power = np.mean(signal**2)
    snr = 10 ** (snr_db / 10)
    noise_power = power / snr
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def time_features(signal):
    return [
        np.mean(signal),
        np.var(signal),
        np.max(signal),
        np.min(signal),
        np.sum(signal**2)
    ]

def spectral_features(signal):
    S = np.abs(np.fft.fft(signal))
    return [
        np.mean(S),
        np.var(S),
        np.max(S)
    ]

def extract_features(signal):
    return time_features(signal) + spectral_features(signal)

def dominant_frequency(signal):
    S = np.abs(np.fft.fft(signal))[:len(signal) // 2]
    return np.argmax(S)

def frequency_shift(signal):
    middle = len(signal) // 2
    f1 = dominant_frequency(signal[:middle])
    f2 = dominant_frequency(signal[middle:])
    return abs(f2 - f1)

def scientific_predict(model, X_test, raw_test):
    proba = model.predict_proba(X_test)
    classes = model.classes_
    idx_lfm = np.where(classes == "LFM")[0][0]
    pred_sci = []
    for i in range(len(X_test)):
        probs = proba[i].copy()
        shift = frequency_shift(raw_test[i])
        if shift > 3:
            probs[idx_lfm] *= 1.10
        else:
            probs[idx_lfm] *= 0.50
        probs = probs / np.sum(probs)
        pred_sci.append(classes[np.argmax(probs)])
    return pred_sci

rng_state = np.random.get_state()
am_example = generate_am()
fm_example = generate_fm()
lfm_example = generate_lfm()

signals = [am_example, fm_example, lfm_example]
signal_names = ["AM", "FM", "LFM"]

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(t[:800], signals[i][:800])
    plt.title("Пример сигнала " + signal_names[i])
    plt.ylabel("Амплитуда")
    plt.grid()
plt.xlabel("Время")
plt.tight_layout()
plt.show()

X = []
Y = []
raw = []
samples = 220
for i in range(samples):
    s = add_noise(generate_am(), np.random.randint(-8, 8))
    X.append(extract_features(s))
    Y.append("AM")
    raw.append(s)

    s = add_noise(generate_fm(), np.random.randint(-8, 8))
    X.append(extract_features(s))
    Y.append("FM")
    raw.append(s)

    s = add_noise(generate_lfm(), np.random.randint(-8, 8))
    X.append(extract_features(s))
    Y.append("LFM")
    raw.append(s)

X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(X,Y,raw,test_size=0.25,random_state=42,stratify=Y)

def balanced_part(X_data, Y_data, raw_data, size):
    X_part = []
    Y_part = []
    raw_part = []
    each_class = size // 3
    for name in ["AM", "FM", "LFM"]:
        count = 0
        for i in range(len(Y_data)):
            if Y_data[i] == name and count < each_class:
                X_part.append(X_data[i])
                Y_part.append(Y_data[i])
                raw_part.append(raw_data[i])
                count += 1
    return X_part, Y_part, raw_part

train_size = 120
X_main, y_main, raw_main = balanced_part(X_train, y_train, raw_train, train_size)
clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(X_main, y_main)
pred = clf.predict(X_test)
acc_ml = accuracy_score(y_test, pred)
print("Точность обычного ML:", round(acc_ml, 4))

pred_sci = scientific_predict(clf, X_test, raw_test)
acc_sci = accuracy_score(y_test, pred_sci)
print("Точность Scientific ML:", round(acc_sci, 4))

small_size = 72
X_small, y_small, raw_small = balanced_part(X_train, y_train, raw_train, small_size)

small_ml = RandomForestClassifier(n_estimators=120, random_state=42)
small_ml.fit(X_small, y_small)
small_pred_ml = small_ml.predict(X_test)
small_acc_ml = accuracy_score(y_test, small_pred_ml)

small_pred_sci = scientific_predict(small_ml, X_test, raw_test)
small_acc_sci = accuracy_score(y_test, small_pred_sci)

print()
print("Сравнение на малой обучающей выборке:")
print("Размер обучающей выборки:", small_size)
print("Обычный ML:", round(small_acc_ml, 4))
print("Scientific ML:", round(small_acc_sci, 4))

plt.figure(figsize=(8, 5))
bars = plt.bar(["Обычный ML", "Scientific ML"], [acc_ml, acc_sci])
plt.title("Сравнение точности моделей")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)
for b in bars:
    plt.text(
        b.get_x() + 0.08,
        b.get_height() + 0.005,
        round(b.get_height(), 3),
        fontsize=12
    )
plt.show()

cm1 = confusion_matrix(y_test, pred, labels=["AM", "FM", "LFM"])
ConfusionMatrixDisplay(cm1, display_labels=["AM", "FM", "LFM"]).plot(cmap="Blues")
plt.title("Матрица ошибок: обычный ML")
plt.show()

cm2 = confusion_matrix(y_test, pred_sci, labels=["AM", "FM", "LFM"])
ConfusionMatrixDisplay(cm2, display_labels=["AM", "FM", "LFM"]).plot(cmap="Greens")
plt.title("Матрица ошибок: Scientific ML")
plt.show()

names = [
    "mean(t)",
    "var(t)",
    "max(t)",
    "min(t)",
    "energy",
    "mean(spec)",
    "var(spec)",
    "max(spec)"
]
imp = clf.feature_importances_
plt.figure(figsize=(10, 5))
plt.bar(names, imp)
plt.xticks(rotation=35)
plt.title("Важность исходных 8 признаков")
plt.tight_layout()
plt.show()

sizes = [72, 120, 180, 260, 360]
acc_small_ml = []
acc_small_sci = []

for size in sizes:
    X_part, y_part, raw_part = balanced_part(X_train, y_train, raw_train, size)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_part, y_part)
    p1 = model.predict(X_test)
    a1 = accuracy_score(y_test, p1)
    p2 = scientific_predict(model, X_test, raw_test)
    a2 = accuracy_score(y_test, p2)
    acc_small_ml.append(a1)
    acc_small_sci.append(a2)

plt.figure(figsize=(9, 5))
plt.plot(sizes, acc_small_ml, marker="o", label="Обычный ML")
plt.plot(sizes, acc_small_sci, marker="s", label="Scientific ML")
plt.xlabel("Размер обучающей выборки")
plt.ylabel("Точность")
plt.title("Эффективность Scientific ML на малых выборках")
plt.grid()
plt.legend()
plt.show()

def make_noise_test(samples_per_class, snr_db):
    X_noise = []
    Y_noise = []
    raw_noise = []
    for i in range(samples_per_class):
        s = add_noise(generate_am(), snr_db)
        X_noise.append(extract_features(s))
        Y_noise.append("AM")
        raw_noise.append(s)

        s = add_noise(generate_fm(), snr_db)
        X_noise.append(extract_features(s))
        Y_noise.append("FM")
        raw_noise.append(s)

        s = add_noise(generate_lfm(), snr_db)
        X_noise.append(extract_features(s))
        Y_noise.append("LFM")
        raw_noise.append(s)
    return X_noise, Y_noise, raw_noise

noise_model = small_ml
snr_values = [-12, -10, -8, -7, -5, 0, 5]
noise_acc_ml = []
noise_acc_sci = []
print()
print("Проверка устойчивости к шуму:")
print("SNR(dB) | Обычный ML | Scientific ML")

for snr in snr_values:
    X_noise, Y_noise, raw_noise = make_noise_test(80, snr)
    p_ml = noise_model.predict(X_noise)
    a_ml = accuracy_score(Y_noise, p_ml)
    p_sci = scientific_predict(noise_model, X_noise, raw_noise)
    a_sci = accuracy_score(Y_noise, p_sci)
    noise_acc_ml.append(a_ml)
    noise_acc_sci.append(a_sci)
    print(snr, "     |", round(a_ml, 4), "    |", round(a_sci, 4))

plt.figure(figsize=(9, 5))
plt.plot(snr_values, noise_acc_ml, marker="o", label="Обычный ML")
plt.plot(snr_values, noise_acc_sci, marker="s", label="Scientific ML")
plt.xlabel("SNR, dB")
plt.ylabel("Точность")
plt.title("Устойчивость моделей к шуму")
plt.grid()
plt.legend()
plt.show()

