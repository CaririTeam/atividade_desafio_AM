import matplotlib
matplotlib.use('Agg')  # Use backend não interativo para evitar avisos de GUI

from flask import Flask, render_template, request, jsonify
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

# Armazenando múltiplos modelos
models = {
    "knn": None,
    "svm": None
}
X_train, X_test, y_train, y_test = None, None, None, None
iris_data = None

@app.route('/')
def home():
    global iris_data
    if iris_data is None:
        iris_data = load_iris()
    return render_template('front.html')

@app.route('/train', methods=['POST'])
def train():
    global models, X_train, X_test, y_train, y_test, iris_data
    if iris_data is None:
        iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target

    # Dividindo em treino e teste (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Treinando KNN ---
    models["knn"] = KNeighborsClassifier(n_neighbors=3)
    models["knn"].fit(X_train, y_train)

    # --- Treinando SVM ---
    models["svm"] = SVC(kernel='linear', probability=True, random_state=42)
    models["svm"].fit(X_train, y_train)

    return jsonify({"message": "Treinamento concluído para KNN e SVM"})

def generate_metrics_and_plots(model, X_data_train, y_data_train, X_data_test, y_data_test, model_name_display):
    """Helper function to generate metrics and plots for a given model."""
    global iris_data
    results = {}

    # --- Métricas de Treinamento ---
    y_pred_train = model.predict(X_data_train)
    acc_train = accuracy_score(y_data_train, y_pred_train)
    prec_train = precision_score(y_data_train, y_pred_train, average='weighted', zero_division=0)
    rec_train = recall_score(y_data_train, y_pred_train, average='weighted', zero_division=0)
    results["train_metrics"] = {"accuracy": round(acc_train, 3), "precision": round(prec_train, 3), "recall": round(rec_train, 3)}

    # --- Métricas de Teste ---
    y_pred_test = model.predict(X_data_test)
    acc_test = accuracy_score(y_data_test, y_pred_test)
    prec_test = precision_score(y_data_test, y_pred_test, average='weighted', zero_division=0)
    rec_test = recall_score(y_data_test, y_pred_test, average='weighted', zero_division=0)
    results["test_metrics"] = {"accuracy": round(acc_test, 3), "precision": round(prec_test, 3), "recall": round(rec_test, 3)}

    # --- Matriz de Confusão (Teste) ---
    cm_test = confusion_matrix(y_data_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão ({model_name_display} - Teste)')
    plt.colorbar()
    tick_marks = np.arange(len(iris_data.target_names))
    plt.xticks(tick_marks, iris_data.target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, iris_data.target_names)
    
    # Adicionar números dentro da matriz
    thresh = cm_test.max() / 2.
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            plt.text(j, i, format(cm_test[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_test[i, j] > thresh else "black")

    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    
    buf_cm = io.BytesIO()
    plt.savefig(buf_cm, format='png')
    buf_cm.seek(0)
    plt.close()
    results["confusion_matrix_b64"] = base64.b64encode(buf_cm.getvalue()).decode('utf-8')

    # --- Superfície de Decisão (Teste - usando apenas 2 atributos para visualização) ---
    # Seleciona apenas os dois últimos atributos (comprimento e largura da pétala) para visualização
    if X_data_train.shape[1] >= 4 and X_data_test.shape[1] >=4:
        X_train_2_features = X_data_train[:, 2:4]
        X_test_2_features = X_data_test[:, 2:4]

        # Treinar um novo modelo apenas com esses 2 atributos para a superfície de decisão
        if isinstance(model, KNeighborsClassifier):
            model_2_features = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        elif isinstance(model, SVC):
            # Para SVC, copie os parâmetros relevantes. Ex: kernel, C, gamma
            model_2_features = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma, random_state=model.random_state, probability=model.probability)
        else: # Fallback para outros modelos, ou não gerar DS
            model_2_features = None

        if model_2_features:
            model_2_features.fit(X_train_2_features, y_data_train)
            
            x_min, x_max = X_train_2_features[:, 0].min() - 0.5, X_train_2_features[:, 0].max() + 0.5
            y_min, y_max = X_train_2_features[:, 1].min() - 0.5, X_train_2_features[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            
            Z = model_2_features.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            plt.figure(figsize=(7, 6)) # Ajuste o tamanho
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm) # plt.cm.RdYlBu
            
            # Plotar os pontos de teste
            for i, target_name in enumerate(iris_data.target_names):
                plt.scatter(X_test_2_features[y_data_test == i, 0], X_test_2_features[y_data_test == i, 1],
                            edgecolors='k', label=target_name, cmap=plt.cm.coolwarm, s=50)

            plt.xlabel(f'{iris_data.feature_names[2]}')
            plt.ylabel(f'{iris_data.feature_names[3]}')
            plt.title(f'Superfície de Decisão ({model_name_display} - Teste)')
            plt.legend(title="Classes Reais (Teste)")
            plt.tight_layout()
            
            buf_ds = io.BytesIO()
            plt.savefig(buf_ds, format='png')
            buf_ds.seek(0)
            plt.close()
            results["decision_surface_b64"] = base64.b64encode(buf_ds.getvalue()).decode('utf-8')
        else:
            results["decision_surface_b64"] = None
    else:
        results["decision_surface_b64"] = None

    return results

@app.route('/test', methods=['GET'])
def test():
    global models, X_test, y_test, X_train, y_train
    
    response_data = {}

    # --- Testar KNN ---
    if models["knn"] is None:
        return jsonify({"error": "Modelo KNN não treinado"}), 400
    response_data["knn"] = generate_metrics_and_plots(
        models["knn"], X_train, y_train, X_test, y_test, "KNN"
    )

    # --- Testar SVM ---
    if models["svm"] is None:
        return jsonify({"error": "Modelo SVM não treinado"}), 400
    response_data["svm"] = generate_metrics_and_plots(
        models["svm"], X_train, y_train, X_test, y_test, "SVM"
    )
    
    return jsonify(response_data)


@app.route('/predict', methods=['POST'])
def predict():
    global models, X_train, y_train, iris_data # Adicionado iris_data
    # Por enquanto, a predição usará apenas o KNN.
    if models["knn"] is None:
        return jsonify({"error": "Modelo KNN não treinado"}), 400
    
    data = request.json
    try:
        # Usar os nomes das features do dataset Iris para garantir a ordem correta
        values = [
            float(data.get(iris_data.feature_names[0].replace(" (cm)", "").replace(" ", "_"), 0.0)), # sepal length (cm)
            float(data.get(iris_data.feature_names[1].replace(" (cm)", "").replace(" ", "_"), 0.0)), # sepal width (cm)
            float(data.get(iris_data.feature_names[2].replace(" (cm)", "").replace(" ", "_"), 0.0)), # petal length (cm)
            float(data.get(iris_data.feature_names[3].replace(" (cm)", "").replace(" ", "_"), 0.0))  # petal width (cm)
        ]
        # Fallback para os nomes antigos se os nomes do dataset não forem encontrados no request.json
        if not any(key in data for key in [name.replace(" (cm)", "").replace(" ", "_") for name in iris_data.feature_names]):
            values = [
                float(data["sepal_length"]),
                float(data["sepal_width"]),
                float(data["petal_length"]),
                float(data["petal_width"])
            ]

    except KeyError as e:
        return jsonify({"error": f"Chave ausente nos dados de entrada: {e}. Esperado: sepal_length, sepal_width, petal_length, petal_width ou nomes correspondentes do dataset."}), 400
    except ValueError:
        return jsonify({"error": "Valores inválidos. Certifique-se de que todos os campos são números."}), 400
    except Exception as e:
        return jsonify({"error": f"Erro ao processar dados: {str(e)}"}), 400


    pred = models["knn"].predict([values])[0]
    result = iris_data.target_names[pred]
    
    # Calcula a acurácia do modelo KNN no conjunto de treino
    acc_train_knn = models["knn"].score(X_train, y_train)
    
    # Retorna o resultado com a acurácia formatada
    return jsonify({"predicao": f"{result} (KNN - ACC Treino: {round(acc_train_knn*100, 0)}%)"})


if __name__ == '__main__':
    app.run(debug=True)