import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import argparse

def loadDataset():
    df = sns.load_dataset('titanic')

    print(f"Número de registros (filas): {df.shape[0]}")
    print(f"Número de variables (columnas): {df.shape[1]}")
    print("\n--- Tipos de datos y valores no nulos ---")
    print(df.info())

    print("\n--- Primeras 5 filas ---")
    print(df.head())
    return df

def eda(df):
    print(df.describe())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x='sex', hue='pclass', data=df[df['survived'] == 1], palette='Set2')
    plt.title('Conteo de Sobrevivientes por Género y Clase')
    plt.xlabel('Género')
    plt.ylabel('Número de Sobrevivientes')
    plt.legend(title='Clase')

    plt.subplot(1, 2, 2)
    sns.histplot(df['age'].dropna(), kde=True, bins=30, color='blue')
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()

def cleanData(df):
    df_clean = df.drop(columns=['deck', 'alive', 'embark_town', 'class', 'alone'])

    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

    df_clean = df_clean.drop_duplicates()

    df_clean = pd.get_dummies(df_clean, columns=['sex', 'embarked', 'who'], drop_first=True)

    print(f"Dimensiones después de la limpieza: {df_clean.shape}")
    return df_clean

def getPredictions(X_train, y_train, X_test,max_iter,random_state,C,solver):
    baseline_model = LogisticRegression(max_iter=max_iter, random_state=random_state,C=C,solver=solver)
    baseline_model.fit(X_train, y_train)

    y_pred = baseline_model.predict(X_test)
    return y_pred,baseline_model

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Accuracy (Exactitud): {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    plt.close()
    return accuracy, precision

parser = argparse.ArgumentParser()
parser.add_argument('--random_state',type=int,default=0)
parser.add_argument('--max_iter',type=int,default=100)
parser.add_argument('--C',type=float, default=1.0)
parser.add_argument('--solver',type=str, default='lbfgs')
args=parser.parse_args()

df=loadDataset()
eda(df)
df_clean=cleanData(df)
X = df_clean.drop(columns=['survived'])
y = df_clean['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)
y_pred, baseline_model=getPredictions(X_train=X_train,X_test=X_test,y_train=y_train,max_iter=args.max_iter,random_state=args.random_state, C=args.C, solver=args.solver)
accuracy,precision=evaluate_model(y_test,y_pred)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic_Baseline_Experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("C", args.C)
    mlflow.log_param("solver", args.solver)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(baseline_model, "logistic_regression_baseline")

    print("Experimento registrado exitosamente en MLflow.")