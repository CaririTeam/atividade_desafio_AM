document.addEventListener('DOMContentLoaded', () => {
    const btnTreinar = document.getElementById('btnTreinar');
    const btnTestar = document.getElementById('btnTestar');
    const formPredict = document.getElementById('formPredict');
    const btnPredict = document.getElementById('btnPredict');
    
    const modelSelector = document.getElementById('modelSelector');
    
    const testResultsDiv = document.getElementById('testResultsContainer');
    const loadingDiv = document.getElementById('loading');
    const resultsContentDiv = document.getElementById('resultsContent');
    const resultadoPredictDiv = document.getElementById('resultadoPredict');

    if (btnTreinar) {
        btnTreinar.addEventListener('click', treinarModelos);
    }

    if (btnTestar) {
        btnTestar.addEventListener('click', testarModeloSelecionado);
    }

    if (formPredict) {
        formPredict.addEventListener('submit', handlePredictSubmit);
    }

    function treinarModelos() {
        if (!btnTreinar || !btnTestar || !btnPredict || !resultadoPredictDiv || !testResultsDiv) {
            console.error("Elementos do DOM não encontrados na função treinarModelos.");
            return;
        }

        btnTreinar.disabled = true;
        btnTreinar.textContent = 'Treinando...';
        btnTestar.disabled = true;
        btnPredict.disabled = true;
        resultadoPredictDiv.style.display = 'none';
        resultadoPredictDiv.classList.remove('success', 'error');
        testResultsDiv.style.display = 'none';

        fetch('/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => { throw new Error(errData.error || `Erro HTTP: ${response.status}`); })
                                   .catch(() => { throw new Error(`Erro HTTP: ${response.status}`); });
            }
            return response.json();
        })
        .then(data => {
            alert(data.message || "Modelos treinados com sucesso!");
            btnTreinar.style.backgroundColor = "green";
            btnTreinar.textContent = 'Modelos Treinados';
            btnTestar.disabled = false;
            btnPredict.disabled = false;
            if (modelSelector && modelSelector.options.length > 0 && modelSelector.value === "") {
                if (modelSelector.options[0] && modelSelector.options[0].value) {
                     modelSelector.value = modelSelector.options[0].value;
                }
            }
        })
        .catch(error => {
            console.error('Erro ao treinar:', error);
            alert(`Erro ao treinar modelos: ${error.message}`);
            btnTreinar.style.backgroundColor = ""; 
            btnTreinar.textContent = 'Treinar Todos os Modelos';
        }).finally(() => {
            if (btnTreinar.textContent === 'Treinando...') {
                btnTreinar.textContent = 'Treinar Todos os Modelos';
            }
            btnTreinar.disabled = false; 
        });
    }

    function displayModelResults(modelData, modelDisplayName) {
        if (!modelData) {
            console.error(`Dados do modelo ${modelDisplayName} estão indefinidos ou nulos.`);
            return `<p style="color: red; text-align:center;">Erro: Dados não recebidos para o modelo ${modelDisplayName}.</p>`;
        }
        if (!modelData.train_metrics || !modelData.test_metrics) {
            console.error(`Dados de métricas incompletos para o modelo ${modelDisplayName}`, modelData);
            return `<p style="color: red; text-align:center;">Erro: Métricas incompletas para o modelo ${modelDisplayName}.</p>`;
        }

        let trainAcc = (modelData.train_metrics.accuracy * 100).toFixed(1);
        let trainPrec = (modelData.train_metrics.precision * 100).toFixed(1);
        let trainRec = (modelData.train_metrics.recall * 100).toFixed(1);
        
        let testAcc = (modelData.test_metrics.accuracy * 100).toFixed(1);
        let testPrec = (modelData.test_metrics.precision * 100).toFixed(1);
        let testRec = (modelData.test_metrics.recall * 100).toFixed(1);

        let modelHTML = `
            <div class="model-results-section">
                <h3>Resultados do Modelo: ${modelDisplayName.toUpperCase()}</h3>
                <div class="metrics-grid">
                    <div class="metrics-block">
                        <h4>Métricas de Treinamento (${modelDisplayName.toUpperCase()})</h4>
                        <p>Acurácia: ${trainAcc}%</p>
                        <p>Precisão: ${trainPrec}%</p>
                        <p>Recall: ${trainRec}%</p>
                    </div>
                    <div class="metrics-block">
                        <h4>Métricas de Teste (${modelDisplayName.toUpperCase()})</h4>
                        <p>Acurácia: ${testAcc}%</p>
                        <p>Precisão: ${testPrec}%</p>
                        <p>Recall: ${testRec}%</p>
                    </div>
                </div>
                <div class="graphs-grid">`;
        
        if (modelData.confusion_matrix_b64) {
            modelHTML += `
                    <div class="graph-container">
                        <h4>Matriz de Confusão (${modelDisplayName.toUpperCase()} - Teste)</h4>
                        <img class="graph" src="data:image/png;base64,${modelData.confusion_matrix_b64}" alt="Matriz de Confusão ${modelDisplayName}">
                    </div>`;
        } else {
            modelHTML += `<div class="graph-container"><p>Matriz de Confusão não disponível para ${modelDisplayName}.</p></div>`;
        }
        
        if (modelData.decision_surface_b64) {
            modelHTML += `
                    <div class="graph-container">
                        <h4>Superfície de Decisão (${modelDisplayName.toUpperCase()} - Teste)</h4>
                        <img class="graph" src="data:image/png;base64,${modelData.decision_surface_b64}" alt="Superfície de Decisão ${modelDisplayName}">
                    </div>`;
        } else {
             modelHTML += `<div class="graph-container"><p>Superfície de Decisão não disponível para ${modelDisplayName}.</p></div>`;
        }
        modelHTML += `</div></div>`;
        return modelHTML;
    }

    function testarModeloSelecionado() {
        if (!testResultsDiv || !loadingDiv || !resultsContentDiv || !modelSelector) {
            console.error("Elementos do DOM não encontrados para testarModeloSelecionado.");
            return;
        }

        const selectedModelKey = modelSelector.value;
        const selectedModelName = modelSelector.options[modelSelector.selectedIndex].text;

        if (!selectedModelKey) {
            alert("Por favor, selecione um modelo para testar.");
            return;
        }

        testResultsDiv.style.display = "block";
        loadingDiv.style.display = "flex";
        resultsContentDiv.style.display = "none";
        resultsContentDiv.innerHTML = ''; 
      
        fetch(`/test_model?model_key=${selectedModelKey}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => { 
                    throw new Error(errData.error || `Erro HTTP: ${response.status}, ${response.statusText}`); 
                }).catch(() => { 
                    throw new Error(`Erro HTTP: ${response.status}, ${response.statusText}. Resposta não é JSON ou erro ao parsear.`); 
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("Dados recebidos de /test_model:", JSON.stringify(data, null, 2));

            if (data.error) {
                throw new Error(data.error);
            }
            
            const modelKeysInData = Object.keys(data);
            if (modelKeysInData.length === 0) {
                throw new Error("Resposta do servidor vazia ou formato inesperado.");
            }
            const modelKeyFromResult = modelKeysInData[0]; 
            const modelData = data[modelKeyFromResult];

            console.log(`Chave do modelo na resposta: ${modelKeyFromResult}`);
            console.log("Dados específicos do modelo (modelData):", JSON.stringify(modelData, null, 2));
            console.log("Nome do modelo para display:", selectedModelName);

            if (modelData && (modelData.train_metrics || modelData.test_metrics)) { // Checa se há pelo menos métricas
                resultsContentDiv.innerHTML = displayModelResults(modelData, selectedModelName);
            } else if (modelData && modelData.error) {
                resultsContentDiv.innerHTML = `<p style="color: red; text-align:center;">Erro ao carregar resultados para ${selectedModelName}: ${modelData.error}</p>`;
            } else {
                 resultsContentDiv.innerHTML = `<p style="color: orange; text-align:center;">Não foi possível exibir os resultados para ${selectedModelName}. Verifique os logs ou a resposta do servidor.</p>`;
            }
            resultsContentDiv.style.display = "block";
        })
        .catch(error => {
            console.error(`Erro ao testar o modelo ${selectedModelName}:`, error);
            resultsContentDiv.innerHTML = `<p style="color: red; text-align:center;">Erro ao carregar resultados: ${error.message}. Verifique o console.</p>`;
            resultsContentDiv.style.display = "block";
        }).finally(() => {
            loadingDiv.style.display = "none";
        });
    }

    function handlePredictSubmit(e) {
        e.preventDefault();
      
        if (!resultadoPredictDiv || !modelSelector) {
            console.error("Elementos do DOM não encontrados para handlePredictSubmit.");
            return;
        }

        const selectedModelKeyForPredict = modelSelector.value;
        if (!selectedModelKeyForPredict) {
            alert("Por favor, selecione um modelo para fazer a predição.");
            return;
        }

        resultadoPredictDiv.classList.remove('success', 'error');
        resultadoPredictDiv.style.display = 'none';
      
        const formData = new FormData(this); 
        const payload = {
            model_key: selectedModelKeyForPredict,
            sepal_length: parseFloat(formData.get('sepal_length')),
            sepal_width: parseFloat(formData.get('sepal_width')),
            petal_length: parseFloat(formData.get('petal_length')),
            petal_width: parseFloat(formData.get('petal_width'))
        };
      
        for (const key in payload) {
            if (key !== 'model_key' && isNaN(payload[key])) {
                const friendlyKeyName = key.replace(/_/g, ' ');
                alert(`Por favor, insira um valor numérico válido para ${friendlyKeyName}.`);
                resultadoPredictDiv.textContent = `Erro: Valor inválido para ${friendlyKeyName}.`;
                resultadoPredictDiv.classList.add('error');
                resultadoPredictDiv.style.display = "block";
                return;
            }
        }

        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || `Erro HTTP: ${response.status}`) });
            }
            return response.json();
        })
        .then(result => {
            if (result.error) {
                throw new Error(result.error);
            }
            resultadoPredictDiv.textContent = `Predição (${result.model_used}): ${result.predicao} (Acc. Treino: ${result.accuracy_train_percent}%)`;
            resultadoPredictDiv.classList.add('success');
            resultadoPredictDiv.style.display = "block";
        })
        .catch(error => {
            console.error('Erro na predição:', error);
            resultadoPredictDiv.textContent = "Erro na predição: " + error.message;
            resultadoPredictDiv.classList.add('error');
            resultadoPredictDiv.style.display = "block";
        });
    }
});