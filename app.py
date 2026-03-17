from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
)

app = Flask(__name__)

# ── Train Naive Bayes ─────────────────────────────────────────────────────────
def train_model():
    df = pd.read_csv('cleaned_dataset.csv')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    le_gender = LabelEncoder()
    le_bmi    = LabelEncoder()
    le_target = LabelEncoder()

    df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
    df['BMI_enc']    = le_bmi.fit_transform(df['BMI Category'])
    df['Target']     = le_target.fit_transform(df['Sleep Disorder'])

    FEATURES = [
        'Gender_enc', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI_enc', 'Heart Rate'
    ]

    X = df[FEATURES].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "="*55)
    print("   NAIVE BAYES - MODEL TRAINING RESULTS")
    print("="*55)
    print(f"  Accuracy       : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Kappa Statistic: {cohen_kappa_score(y_test, y_pred):.4f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    print("="*55 + "\n")

    with open('sleep_disorder_model.pkl', 'wb') as f:
        pickle.dump({
            'model'    : model,
            'le_gender': le_gender,
            'le_bmi'   : le_bmi,
            'le_target': le_target,
        }, f)
    print("  Model saved as sleep_disorder_model.pkl\n")

# ── Train K-Means and save ────────────────────────────────────────────────────
def train_clustering():
    df = pd.read_csv('cleaned_dataset.csv')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    le_gender = LabelEncoder()
    le_bmi    = LabelEncoder()
    le_target = LabelEncoder()

    df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
    df['BMI_enc']    = le_bmi.fit_transform(df['BMI Category'])
    df['Target']     = le_target.fit_transform(df['Sleep Disorder'])

    FEATURES = [
        'Gender_enc', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI_enc', 'Heart Rate'
    ]

    X = df[FEATURES].values
    y = df['Target'].values

    kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Unique cluster-to-class assignment (greedy best match)
    class_names = le_target.classes_
    scores      = np.zeros((3, 3))
    for c in range(3):
        mask = clusters == c
        for k in range(3):
            scores[c, k] = np.sum(y[mask] == k)

    cluster_labels = {}
    used_classes   = set()
    for _ in range(3):
        best_score = -1
        best_c = best_k = 0
        for c in range(3):
            if c in cluster_labels:
                continue
            for k in range(3):
                if k in used_classes:
                    continue
                if scores[c, k] > best_score:
                    best_score = scores[c, k]
                    best_c, best_k = c, k
        cluster_labels[best_c] = class_names[best_k]
        used_classes.add(best_k)

    # Accuracy
    predicted = np.array([cluster_labels[c] for c in clusters])
    actual    = le_target.inverse_transform(y)
    correct   = np.sum(predicted == actual)
    accuracy  = correct / len(y) * 100

    feature_names = ['Gender', 'Age', 'Sleep Dur', 'Quality',
                     'Activity', 'Stress', 'BMI', 'Heart Rate']

    print("\n" + "="*55)
    print("   K-MEANS CLUSTERING RESULTS (K=3)")
    print("="*55)
    print(f"  Total Instances       : {len(y)}")
    print(f"  Correctly Clustered   : {correct} ({accuracy:.2f}%)")
    print(f"  Incorrectly Clustered : {len(y)-correct} ({100-accuracy:.2f}%)")
    print("\n  Cluster Assignments:")
    for c in range(3):
        count = np.sum(clusters == c)
        pct   = count / len(y) * 100
        print(f"  Cluster {c} --> {cluster_labels[c]:<15} | {count} instances ({pct:.1f}%)")
    print("\n  Cluster Centroids (feature averages):")
    for c in range(3):
        print(f"\n  [{cluster_labels[c]}]")
        for fname, val in zip(feature_names, kmeans.cluster_centers_[c]):
            print(f"    {fname:<12}: {val:.2f}")
    print("\n" + "="*55 + "\n")

    # Save kmeans + its label map
    with open('sleep_disorder_kmeans.pkl', 'wb') as f:
        pickle.dump({
            'kmeans'        : kmeans,
            'cluster_labels': cluster_labels,
        }, f)
    print("  KMeans saved as sleep_disorder_kmeans.pkl\n")

# ── Run on startup ────────────────────────────────────────────────────────────
if not os.path.exists('sleep_disorder_model.pkl'):
    train_model()

if not os.path.exists('sleep_disorder_kmeans.pkl'):
    train_clustering()
else:
    # Still print clustering results to terminal every run
    train_clustering()

# Load Naive Bayes
with open('sleep_disorder_model.pkl', 'rb') as f:
    saved     = pickle.load(f)
model     = saved['model']
le_gender = saved['le_gender']
le_bmi    = saved['le_bmi']
le_target = saved['le_target']

# Load KMeans
with open('sleep_disorder_kmeans.pkl', 'rb') as f:
    ksaved         = pickle.load(f)
kmeans_model   = ksaved['kmeans']
cluster_labels = ksaved['cluster_labels']

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender   = request.form['gender']
        age      = float(request.form['age'])
        sleep_d  = float(request.form['sleep_duration'])
        quality  = float(request.form['quality'])
        activity = float(request.form['activity'])
        stress   = float(request.form['stress'])
        bmi      = request.form['bmi']
        hr       = float(request.form['heart_rate'])

        gender_enc = le_gender.transform([gender])[0]
        bmi_enc    = le_bmi.transform([bmi])[0]

        features = np.array([[gender_enc, age, sleep_d, quality,
                               activity, stress, bmi_enc, hr]])

        # ── Naive Bayes prediction ──
        pred_enc   = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        pred_label = le_target.inverse_transform([pred_enc])[0]
        classes    = le_target.classes_
        probs      = {cls: round(float(prob)*100, 2)
                      for cls, prob in zip(classes, pred_proba)}

        # ── K-Means cluster prediction ──
        cluster_num   = int(kmeans_model.predict(features)[0])
        cluster_name  = cluster_labels[cluster_num]

        # Distance to each centroid (closer = more similar)
        distances     = kmeans_model.transform(features)[0]
        nearest_dist  = round(float(distances[cluster_num]), 2)

        return render_template('index.html',
                               prediction=pred_label,
                               probs=probs,
                               cluster_name=cluster_name,
                               cluster_num=cluster_num,
                               nearest_dist=nearest_dist,
                               input_data=request.form)
    except Exception as e:
        return render_template('index.html',
                               prediction="Error",
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)