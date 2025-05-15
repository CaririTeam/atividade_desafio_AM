import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # Novo modelo importado
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

# Armazenando múltiplos modelos
models = {
    "knn": None,
    "svm": None,
    "decision_tree": None # Novo modelo adicionado
}
X_train, X_test, y_train, y_test = None, None, None, None
iris_data = None # Cache para os dados do Iris

def load_iris_dataset():
    """Carrega o dataset Iris se ainda não estiver carregado."""
    global iris_data
    if iris_data is None:
        iris_data = load_iris()
    return iris_data

@app.route('/')
def home():
    load_iris_dataset()
    return render_template('front.html')

@app.route('/train', methods=['POST'])
def train():
    global models, X_train, X_test, y_train, y_test
    
    current_iris_data = load_iris_dataset()
    X = current_iris_data.data
    y = current_iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinando KNN
    models["knn"] = KNeighborsClassifier(n_neighbors=3)
    models["knn"].fit(X_train, y_train)

    # Treinando SVM
    models["svm"] = SVC(kernel='linear', probability=True, random_state=42) # probability=True para superfície de decisão
    models["svm"].fit(X_train, y_train)

    # Treinando Árvore de Decisão
    models["decision_tree"] = DecisionTreeClassifier(random_state=42)
    models["decision_tree"].fit(X_train, y_train)

    return jsonify({"message": "Treinamento concluído para KNN, SVM e Árvore de Decisão"})

def generate_metrics_and_plots(model, X_data_train, y_data_train, X_data_test, y_data_test, model_name_display):
    """Gera métricas, matriz de confusão e superfície de decisão para um modelo."""
    current_iris_data = load_iris_dataset()
    results = {}

    # Métricas de Treinamento
    y_pred_train = model.predict(X_data_train)
    results["train_metrics"] = {
        "accuracy": round(accuracy_score(y_data_train, y_pred_train), 3),
        "precision": round(precision_score(y_data_train, y_pred_train, average='weighted', zero_division=0), 3),
        "recall": round(recall_score(y_data_train, y_pred_train, average='weighted', zero_division=0), 3)
    }

    # Métricas de Teste
    y_pred_test = model.predict(X_data_test)
    results["test_metrics"] = {
        "accuracy": round(accuracy_score(y_data_test, y_pred_test), 3),
        "precision": round(precision_score(y_data_test, y_pred_test, average='weighted', zero_division=0), 3),
        "recall": round(recall_score(y_data_test, y_pred_test, average='weighted', zero_division=0), 3)
    }

    # Matriz de Confusão (Teste)
    cm_test = confusion_matrix(y_data_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão ({model_name_display} - Teste)')
    plt.colorbar()
    tick_marks = np.arange(len(current_iris_data.target_names))
    plt.xticks(tick_marks, current_iris_data.target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, current_iris_data.target_names)
    
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
    plt.close() # Fechar a figura para liberar memória
    buf_cm.seek(0)
    results["confusion_matrix_b64"] = base64.b64encode(buf_cm.getvalue()).decode('utf-8')

    # Superfície de Decisão (Teste - usando apenas 2 atributos para visualização)
    # Usaremos comprimento e largura da pétala (índices 2 e 3)
    if X_data_train.shape[1] >= 4 and X_data_test.shape[1] >=4:
        X_train_2_features = X_data_train[:, 2:4] # petal length e petal width
        X_test_2_features = X_data_test[:, 2:4]   # petal length e petal width

        model_2_features = None
        if isinstance(model, KNeighborsClassifier):
            model_2_features = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        elif isinstance(model, SVC):
            model_2_features = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma, 
                                   random_state=model.random_state, probability=model.probability)
        elif isinstance(model, DecisionTreeClassifier):
            model_2_features = DecisionTreeClassifier(random_state=model.random_state, 
                                                      max_depth=model.get_params().get('max_depth')) # Copia max_depth se existir

        if model_2_features:
            model_2_features.fit(X_train_2_features, y_data_train)
            
            x_min, x_max = X_train_2_features[:, 0].min() - 0.5, X_train_2_features[:, 0].max() + 0.5
            y_min, y_max = X_train_2_features[:, 1].min() - 0.5, X_train_2_features[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            
            Z = model_2_features.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            plt.figure(figsize=(7, 6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            
            for i, target_name in enumerate(current_iris_data.target_names):
                plt.scatter(X_test_2_features[y_data_test == i, 0], X_test_2_features[y_data_test == i, 1],
                            edgecolors='k', label=target_name, cmap=plt.cm.coolwarm, s=50)

            plt.xlabel(f'{current_iris_data.feature_names[2]}') # Petal Length
            plt.ylabel(f'{current_iris_data.feature_names[3]}') # Petal Width
            plt.title(f'Superfície de Decisão ({model_name_display} - Teste)')
            plt.legend(title="Classes Reais (Teste)")
            plt.tight_layout()
            
            buf_ds = io.BytesIO()
            plt.savefig(buf_ds, format='png')
            plt.close() # Fechar a figura para liberar memória
            buf_ds.seek(0)
            results["decision_surface_b64"] = base64.b64encode(buf_ds.getvalue()).decode('utf-8')
        else:
            results["decision_surface_b64"] = None # Caso o modelo não seja um dos esperados para DS
    else:
        results["decision_surface_b64"] = None # Caso não haja features suficientes

    return results

@app.route('/test', methods=['GET'])
def test():
    global models, X_test, y_test, X_train, y_train
    
    if X_train is None: # Verifica se o treinamento já ocorreu
        return jsonify({"error": "Modelos não treinados. Por favor, treine os modelos primeiro."}), 400

    response_data = {}

    # Testar KNN
    if models["knn"] is None:
        return jsonify({"error": "Modelo KNN não treinado"}), 400
    response_data["knn"] = generate_metrics_and_plots(
        models["knn"], X_train, y_train, X_test, y_test, "KNN"
    )

    # Testar SVM
    if models["svm"] is None:
        return jsonify({"error": "Modelo SVM não treinado"}), 400
    response_data["svm"] = generate_metrics_and_plots(
        models["svm"], X_train, y_train, X_test, y_test, "SVM"
    )

    # Testar Árvore de Decisão
    if models["decision_tree"] is None:
        return jsonify({"error": "Modelo Árvore de Decisão não treinado"}), 400
    response_data["decision_tree"] = generate_metrics_and_plots(
        models["decision_tree"], X_train, y_train, X_test, y_test, "Árvore de Decisão"
    )
    
    return jsonify(response_data)


@app.route('/predict', methods=['POST'])
def predict():
    global models, X_train, y_train
    
    current_iris_data = load_iris_dataset()

    # Predição ainda usa KNN por padrão. Será alterado na HU-003.
    if models["knn"] is None:
        return jsonify({"error": "Modelo KNN não treinado para predição. Treine os modelos primeiro."}), 400
    
    data = request.json
    try:
        # Usar os nomes das features do dataset Iris para garantir a ordem correta, removendo "(cm)" e espaços.
        feature_keys_map = {name.replace(" (cm)", "").replace(" ", "_").lower(): name for name in current_iris_data.feature_names}
        
        # Tenta obter os valores usando os nomes mapeados. Se falhar, tenta os nomes originais como fallback.
        values = []
        for original_feature_name in current_iris_data.feature_names:
            key_in_data = original_feature_name.replace(" (cm)", "").replace(" ", "_").lower()
            if key_in_data in data:
                values.append(float(data[key_in_data]))
            elif original_feature_name.replace(" (cm)", "") in data: # Fallback para nome sem "_", mas com espaço
                 values.append(float(data[original_feature_name.replace(" (cm)", "")]))
            elif original_feature_name in data: # Fallback para nome original completo
                 values.append(float(data[original_feature_name]))
            else: # Fallback para os nomes fixos antigos se tudo mais falhar
                if "sepal_length" in data and original_feature_name == current_iris_data.feature_names[0]:
                    values.append(float(data["sepal_length"]))
                elif "sepal_width" in data and original_feature_name == current_iris_data.feature_names[1]:
                    values.append(float(data["sepal_width"]))
                elif "petal_length" in data and original_feature_name == current_iris_data.feature_names[2]:
                    values.append(float(data["petal_length"]))
                elif "petal_width" in data and original_feature_name == current_iris_data.feature_names[3]:
                    values.append(float(data["petal_width"]))
                else:
                    raise KeyError(f"Valor para '{original_feature_name}' não encontrado nos dados de entrada.")

        if len(values) != 4: # Garante que temos 4 valores
             raise ValueError("Número incorreto de features fornecidas.")

    except KeyError as e:
        return jsonify({"error": f"Chave ausente ou nome de feature inválido nos dados de entrada: {str(e)}. Esperado: {', '.join([name.replace(' (cm)', '').replace(' ', '_').lower() for name in current_iris_data.feature_names])}"}), 400
    except ValueError:
        return jsonify({"error": "Valores inválidos. Certifique-se de que todos os campos são números e forneça todas as 4 features."}), 400
    except Exception as e:
        return jsonify({"error": f"Erro ao processar dados de predição: {str(e)}"}), 400

    pred_index = models["knn"].predict([values])[0]
    pred_class_name = current_iris_data.target_names[pred_index]
    
    acc_train_knn = accuracy_score(y_train, models["knn"].predict(X_train)) # Recalcula para garantir
    
    return jsonify({"predicao": f"{pred_class_name} (Modelo KNN - Acurácia Treino: {round(acc_train_knn*100, 1)}%)"})


if __name__ == '__main__':
    app.run(debug=True)