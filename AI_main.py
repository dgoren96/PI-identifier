"""
Full integrated script with:
- RF, XGBoost, LightGBM, CART, REP Tree, NN, KNN, SVM, PCA+KNN
- PCA, t-SNE, UMAP visualization of raw data
- 2D PCA visualization colored by model predictions (KNN, SVM)
- Reports for all models
- Summary table at the end
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ------------------------
# Helper functions
# ------------------------
def load_and_preprocess_data(features_csv, target_csv):
    """
    Load features and target CSVs, merge them, and preprocess for analysis.

    Parameters
    ----------
    features_csv : str
        Path to CSV containing extracted features per sample.
    target_csv : str
        Path to CSV containing target labels (e.g., gestational info) per sample.

    Returns
    -------
    df : pandas.DataFrame
        Merged and preprocessed DataFrame with:
        - 'frames' converted to numpy arrays
        - 'top3_f_amp' and 'top3_f_freq' expanded into separate columns:
          'amp1','amp2','amp3','freq1','freq2','freq3'
    """
    features_df = pd.read_csv(features_csv)
    target_df = pd.read_csv(target_csv)
    df = pd.merge(features_df, target_df, on='id_int', how='inner')

    # Convert 'frames' column from string to numpy array if needed
    df['frames'] = df['frames'].apply(lambda x: np.fromstring(x[1:-1], sep=' ') if isinstance(x,str) else np.array([]))

    # Safely extract first three elements from top3 frequency/amplitude columns
    def safe_eval(cell):
        try:
            val = eval(cell)
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                return list(val[:3])
        except:
            pass
        return [0.0,0.0,0.0]

    df[['amp1','amp2','amp3']] = df['top3_f_amp'].apply(lambda x: pd.Series(safe_eval(x)))
    df[['freq1','freq2','freq3']] = df['top3_f_freq'].apply(lambda x: pd.Series(safe_eval(x)))
    df.drop(['top3_f_amp','top3_f_freq'], axis=1, inplace=True, errors='ignore')

    return df


def create_neural_network(input_dim):
    """
    Build and compile a fully connected neural network for binary classification.

    Parameters
    ----------
    input_dim : int
        Number of input features for the network.

    Returns
    -------
    model : keras.Model
        Compiled Keras Sequential model with the following architecture:
        - Dense layer with 128 units, ReLU activation, BatchNormalization, Dropout 0.3
        - Dense layer with 64 units, ReLU activation, BatchNormalization, Dropout 0.3
        - Dense layer with 32 units, ReLU activation, BatchNormalization, Dropout 0.3
        - Dense layer with 8 units, ReLU activation, BatchNormalization
        - Output Dense layer with 1 unit and sigmoid activation
        - Optimizer: Adam with learning rate 0.0002
        - Loss: Binary crossentropy
        - Metrics: accuracy, precision, recall
    """
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    """
    Train a given model, evaluate it on the test set, and return predictions for visualization.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g., "Neural Network", "KNN", "SVM".
    model : estimator or keras.Model
        The machine learning model to train.
    X_train, X_test : array-like or pandas.DataFrame
        Training and test features.
    y_train, y_test : array-like or pandas.Series
        Training and test labels.
    X_val, y_val : array-like, optional
        Validation set for neural networks (used for early stopping).

    Returns
    -------
    model_name : str
        Name of the model.
    accuracy : float
        Accuracy on the test set.
    report : str
        Classification report of precision, recall, f1-score.
    y_pred : ndarray
        Predicted labels for the test set.
    """
    X_train_np = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
    y_train_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    if X_val is not None:
        X_val_np = X_val.values if hasattr(X_val, 'values') else np.asarray(X_val)
        y_val_np = y_val.values if hasattr(y_val, 'values') else np.asarray(y_val)

    # Neural Network
    if model_name == "Neural Network":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)
        X_val_scaled = scaler.transform(X_val_np)

        class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)
        class_weights = dict(enumerate(class_weights_arr))

        early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

        history = model.fit(
            X_train_scaled, y_train_np,
            validation_data=(X_val_scaled, y_val_np),
            epochs=100,
            batch_size=4,
            callbacks=[early_stop],
            verbose=0,
            class_weight=class_weights
        )

        y_probs = model.predict(X_test_scaled).ravel()
        threshold = 0.5
        y_pred = (y_probs>threshold).astype(int)
        accuracy = accuracy_score(y_test_np, y_pred)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y_pred.dtype.kind not in ('i','u'):
            y_pred = (y_pred>0.5).astype(int)
        accuracy = accuracy_score(y_test_np, y_pred)

    report = classification_report(y_test_np, y_pred)
    print(f"\n{model_name} Accuracy: {accuracy:.3f}")
    print(f"{model_name} Classification Report:\n{report}")

    return model_name, accuracy, report, y_pred


def plot_predictions_2d(X, y_true, y_pred, title="Predictions 2D"):
    """
    Visualize predicted vs. true labels in 2D using PCA.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix.
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    title : str, optional
        Title for the plot (default is "Predictions 2D").

    Returns
    -------
    None
        Displays a 2D scatter plot with true labels as 'x' and predicted labels as colored points.
    """
    X_np = X.values if hasattr(X,'values') else np.asarray(X)
    X_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_np))
    plt.figure(figsize=(6,5))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y_pred, alpha=0.6, cmap='coolwarm', label='Predicted')
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y_true, alpha=0.2, cmap='coolwarm', marker='x', label='True')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dim_reduction(X, y, method="PCA", title=None):
    """
    Visualize high-dimensional data in 2D using dimensionality reduction.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix.
    y : array-like
        Class labels used for coloring the points.
    method : str, optional
        Dimensionality reduction method: "PCA", "t-SNE", or "UMAP" (default is "PCA").
    title : str, optional
        Title for the plot. If None, the method name is used.

    Returns
    -------
    None
        Displays a 2D scatter plot of the reduced data colored by labels.
    """
    X_np = X.values if hasattr(X,'values') else np.asarray(X)
    X_scaled = StandardScaler().fit_transform(X_np)
    title = title or method
    if method=="PCA":
        X_2d = PCA(n_components=2).fit_transform(X_scaled)
    elif method=="t-SNE":
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
    elif method=="UMAP" and HAS_UMAP:
        X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        return
    plt.figure(figsize=(6,5))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, alpha=0.6, cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.show()

"""
Main script for training and evaluating multiple models on placental insufficiency detection.

Steps:
1. Load original and augmented feature CSVs and merge with target labels.
2. Split data into train, validation, and test sets by unique patient IDs.
3. Visualize raw training data using PCA, t-SNE, and UMAP.
4. Define multiple machine learning models, including tree-based, KNN, SVM, PCA+KNN, and a neural network.
5. Train each model, evaluate on the test set, and print classification reports.
6. Visualize predictions of selected models in 2D.
7. Print a summary table with model accuracies.
"""

if __name__=='__main__':
    # Load data
    features_csv = r'C:\Python_Projects\pythonProject\features.csv'
    features_augmented_csv = r'C:\Python_Projects\pythonProject\features_augmented.csv'
    target_csv = r'C:\Python_Projects\pythonProject\target.csv'

    df_real = load_and_preprocess_data(features_csv, target_csv)
    df_aug = load_and_preprocess_data(features_augmented_csv, target_csv)
    df = pd.concat([df_real, df_aug], ignore_index=True)
    #df = df_real  # keep original

    target = 'has_pi'
    features = [c for c in df.columns if c in ['PI','SD','RI','DVD_S','DI','mean_amp',
                                               'duty_cycle','HR','amp1','amp2','amp3','freq1','freq2','freq3']]

    # Train/val/test split by id
    unique_ids = df['id_int'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.33, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.33, random_state=42)

    train_df = df[df['id_int'].isin(train_ids)]
    val_df = df[df['id_int'].isin(val_ids)]
    test_df = df[df['id_int'].isin(test_ids)]

    X_train = train_df[features]; y_train = train_df[target].astype(int)
    X_val = val_df[features]; y_val = val_df[target].astype(int)
    X_test = test_df[features]; y_test = test_df[target].astype(int)

    # Dimensionality reduction plots on raw data
    plot_dim_reduction(X_train, y_train, method="PCA", title="PCA 2D (raw train)")
    plot_dim_reduction(X_train, y_train, method="t-SNE", title="t-SNE 2D (raw train)")
    plot_dim_reduction(X_train, y_train, method="UMAP", title="UMAP 2D (raw train)")

    # Define models
    models = [
        ("Random Forest", RandomForestClassifier(class_weight='balanced', n_estimators=256, random_state=42)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ("LightGBM", LGBMClassifier(n_estimators=256, random_state=42)),
        ("CART", DecisionTreeClassifier(random_state=42)),
        ("REP Tree", DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, ccp_alpha=0.01, random_state=42)),
        ("Neural Network", create_neural_network(input_dim=X_train.shape[1])),
        ("KNN (k=5)", Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))])),
        ("SVM", Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])),
        ("PCA+KNN", Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=8)), ('knn', KNeighborsClassifier(n_neighbors=3))]))
    ]

    # Train all models and collect results
    all_results = []
    predictions_for_plot = {}

    for name, model in models:
        try:
            res_name, acc, report, y_pred = train_and_evaluate(name, model, X_train, X_test, y_train, y_test, X_val,
                                                               y_val)
            # Always append every model to all_results
            all_results.append((res_name, acc, report))

            # Store y_pred for plotting only for certain models
            if "knn" in name or "svm" in name or "pca" in name:
                predictions_for_plot[name] = y_pred

        except Exception as e:
            print(f"{name} failed: {e}")

    # Plot predictions for KNN/SVM/PCA+KNN
    for model_name, y_pred in predictions_for_plot.items():
        plot_predictions_2d(X_test, y_test, y_pred, title=f"{model_name} Predictions (2D PCA)")

    # Print summary
    print("\n\n================= MODEL SUMMARY =================\n")
    summary_df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "Classification Report"])
    summary_df = summary_df.sort_values("Accuracy", ascending=False)
    print(summary_df[["Model", "Accuracy"]])
    print("\nFull classification reports printed above.\n")
