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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

models = {
    "knn": {"model": None, "name": "KNN"},
    "svm": {"model": None, "name": "SVM"},
    "decision_tree": {"model": None, "name": "Árvore de Decisão"}
}
X_train, X_test, y_train, y_test = None, None, None, None
iris_data = None

def load_iris_dataset():
    global iris_data
    if iris_data is None:
        iris_data = load_iris()
    return iris_data

@app.route('/')
def home():
    load_iris_dataset()
    # Passa os nomes dos modelos para o template, para popular o menu de seleção
    model_names = {key: details["name"] for key, details in models.items()}
    return render_template('front.html', available_models=model_names)

@app.route('/train', methods=['POST'])
def train_all_models():
    global X_train, X_test, y_train, y_test
    
    current_iris_data = load_iris_dataset()
    X = current_iris_data.data
    y = current_iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models["knn"]["model"] = KNeighborsClassifier(n_neighbors=3)
    models["knn"]["model"].fit(X_train, y_train)

    models["svm"]["model"] = SVC(kernel='linear', probability=True, random_state=42)
    models["svm"]["model"].fit(X_train, y_train)

    models["decision_tree"]["model"] = DecisionTreeClassifier(random_state=42)
    models["decision_tree"]["model"].fit(X_train, y_train)

    return jsonify({"message": "Todos os modelos foram treinados com sucesso."})

def generate_metrics_and_plots(model_instance, X_data_train, y_data_train, X_data_test, y_data_test, model_display_name):
    current_iris_data = load_iris_dataset()
    results = {}

    y_pred_train = model_instance.predict(X_data_train)
    results["train_metrics"] = {
        "accuracy": round(accuracy_score(y_data_train, y_pred_train), 3),
        "precision": round(precision_score(y_data_train, y_pred_train, average='weighted', zero_division=0), 3),
        "recall": round(recall_score(y_data_train, y_pred_train, average='weighted', zero_division=0), 3)
    }

    y_pred_test = model_instance.predict(X_data_test)
    results["test_metrics"] = {
        "accuracy": round(accuracy_score(y_data_test, y_pred_test), 3),
        "precision": round(precision_score(y_data_test, y_pred_test, average='weighted', zero_division=0), 3),
        "recall": round(recall_score(y_data_test, y_pred_test, average='weighted', zero_division=0), 3)
    }

    cm_test = confusion_matrix(y_data_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão ({model_display_name} - Teste)')
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
    plt.ylabel('Verdadeiro'); plt.xlabel('Predito'); plt.tight_layout()
    buf_cm = io.BytesIO(); plt.savefig(buf_cm, format='png'); plt.close(); buf_cm.seek(0)
    results["confusion_matrix_b64"] = base64.b64encode(buf_cm.getvalue()).decode('utf-8')

    if X_data_train.shape[1] >= 4:
        X_train_2f = X_data_train[:, 2:4]; X_test_2f = X_data_test[:, 2:4]
        model_2f = None
        if isinstance(model_instance, KNeighborsClassifier):
            model_2f = KNeighborsClassifier(n_neighbors=model_instance.n_neighbors)
        elif isinstance(model_instance, SVC):
            model_2f = SVC(kernel=model_instance.kernel, C=model_instance.C, gamma=model_instance.gamma, 
                           random_state=model_instance.random_state, probability=model_instance.probability)
        elif isinstance(model_instance, DecisionTreeClassifier):
            model_2f = DecisionTreeClassifier(random_state=model_instance.random_state, 
                                              max_depth=model_instance.get_params().get('max_depth'))
        if model_2f:
            model_2f.fit(X_train_2f, y_data_train)
            x_min, x_max = X_train_2f[:, 0].min() - 0.5, X_train_2f[:, 0].max() + 0.5
            y_min, y_max = X_train_2f[:, 1].min() - 0.5, X_train_2f[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            Z = model_2f.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            plt.figure(figsize=(7, 6)); plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            for i, target_name in enumerate(current_iris_data.target_names):
                plt.scatter(X_test_2f[y_data_test == i, 0], X_test_2f[y_data_test == i, 1],
                            edgecolors='k', label=target_name, cmap=plt.cm.coolwarm, s=50)
            plt.xlabel(f'{current_iris_data.feature_names[2]}'); plt.ylabel(f'{current_iris_data.feature_names[3]}')
            plt.title(f'Superfície de Decisão ({model_display_name} - Teste)'); plt.legend(title="Classes Reais (Teste)"); plt.tight_layout()
            buf_ds = io.BytesIO(); plt.savefig(buf_ds, format='png'); plt.close(); buf_ds.seek(0)
            results["decision_surface_b64"] = base64.b64encode(buf_ds.getvalue()).decode('utf-8')
        else: results["decision_surface_b64"] = None
    else: results["decision_surface_b64"] = None
    return results

@app.route('/test_model', methods=['GET']) # Rota renomeada/alterada para testar um modelo específico
def test_single_model():
    global X_train, y_train, X_test, y_test
    
    model_key = request.args.get('model_key') # Espera um parâmetro 'model_key' (ex: 'knn', 'svm')

    if not model_key or model_key not in models:
        return jsonify({"error": "Chave de modelo inválida ou não fornecida."}), 400
    
    selected_model_dict = models[model_key]
    model_instance = selected_model_dict["model"]
    model_display_name = selected_model_dict["name"]

    if X_train is None or y_train is None or X_test is None or y_test is None:
        return jsonify({"error": "Os dados de treino/teste não estão disponíveis. Treine os modelos primeiro."}), 400
    
    if model_instance is None:
        return jsonify({"error": f"Modelo '{model_display_name}' não foi treinado ainda."}), 400

    results = generate_metrics_and_plots(
        model_instance, X_train, y_train, X_test, y_test, model_display_name
    )
    return jsonify({model_key: results}) # Retorna os resultados apenas para o modelo solicitado

@app.route('/test_all_models', methods=['GET']) # Nova rota para testar todos os modelos (opcional, pode ser útil)
def test_all_available_models():
    global X_train, y_train, X_test, y_test
    
    if X_train is None:
        return jsonify({"error": "Modelos não treinados. Por favor, treine os modelos primeiro."}), 400

    response_data = {}
    all_models_trained = True

    for key, details in models.items():
        if details["model"] is not None:
            response_data[key] = generate_metrics_and_plots(
                details["model"], X_train, y_train, X_test, y_test, details["name"]
            )
        else:
            all_models_trained = False
            response_data[key] = {"error": f"Modelo '{details['name']}' não treinado."}
            # Poderia optar por não incluir modelos não treinados ou retornar um status de erro geral
    
    if not any(details["model"] is not None for details in models.values()): # Verifica se pelo menos um modelo foi treinado
        return jsonify({"error": "Nenhum modelo foi treinado ainda."}), 400

    return jsonify(response_data)


@app.route('/predict', methods=['POST'])
def predict_with_model():
    global X_train, y_train # y_train é usado para calcular a acurácia de treino do modelo escolhido
    
    current_iris_data = load_iris_dataset()
    data = request.json
    
    model_key = data.get('model_key') # Espera 'model_key' no corpo do JSON da requisição

    if not model_key or model_key not in models:
        return jsonify({"error": "Chave de modelo inválida ou não fornecida para predição."}), 400

    selected_model_dict = models[model_key]
    model_instance = selected_model_dict["model"]
    model_display_name = selected_model_dict["name"]

    if model_instance is None:
        return jsonify({"error": f"Modelo '{model_display_name}' não treinado. Não é possível fazer predição."}), 400
    
    if X_train is None or y_train is None: # Checa se houve treino para calcular acc de treino
        return jsonify({"error": "Dados de treinamento não disponíveis. Treine os modelos primeiro."}), 400

    try:
        feature_keys_map = {name.replace(" (cm)", "").replace(" ", "_").lower(): name for name in current_iris_data.feature_names}
        values = []
        for original_feature_name in current_iris_data.feature_names:
            key_in_data = original_feature_name.replace(" (cm)", "").replace(" ", "_").lower()
            # Lógica de fallback para encontrar a feature nos dados de entrada
            if key_in_data in data: values.append(float(data[key_in_data]))
            elif original_feature_name.replace(" (cm)", "") in data: values.append(float(data[original_feature_name.replace(" (cm)", "")]))
            elif original_feature_name in data: values.append(float(data[original_feature_name]))
            else: raise KeyError(f"Valor para '{original_feature_name}' não encontrado.")

        if len(values) != 4: raise ValueError("Número incorreto de features.")

    except KeyError as e:
        return jsonify({"error": f"Chave ausente: {str(e)}. Esperado: {', '.join([name.replace(' (cm)', '').replace(' ', '_').lower() for name in current_iris_data.feature_names])}"}), 400
    except ValueError:
        return jsonify({"error": "Valores inválidos. Forneça 4 features numéricas."}), 400
    except Exception as e:
        return jsonify({"error": f"Erro ao processar dados: {str(e)}"}), 400

    pred_index = model_instance.predict([values])[0]
    pred_class_name = current_iris_data.target_names[pred_index]
    
    # Calcula a acurácia de treino para o modelo selecionado
    acc_train_selected_model = accuracy_score(y_train, model_instance.predict(X_train))
    
    return jsonify({
        "predicao": pred_class_name,
        "model_used": model_display_name,
        "accuracy_train_percent": round(acc_train_selected_model * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)