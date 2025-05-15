// static/js/script.js

// Garantir que o DOM esteja carregado antes de tentar acessar os elementos
document.addEventListener('DOMContentLoaded', (event) => {
    const btnTreinar = document.getElementById('btnTreinar');
    const btnTestar = document.getElementById('btnTestar');
    const btnPredict = document.getElementById('btnPredict'); // Botão submit do formulário de predição
    const formPredict = document.getElementById('formPredict'); // O formulário de predição
    const testResultsDiv = document.getElementById('testResultsContainer');
    const loadingDiv = document.getElementById('loading');
    const resultsContentDiv = document.getElementById('resultsContent');
    const resultadoPredictDiv = document.getElementById('resultadoPredict');

    // Verificar se os elementos existem antes de adicionar event listeners
    if (btnTreinar) {
        btnTreinar.addEventListener('click', treinar);
    }

    if (btnTestar) {
        btnTestar.addEventListener('click', testar);
    }

    if (formPredict) { // Adicionar listener ao formulário de predição
        formPredict.addEventListener('submit', handlePredictSubmit);
    }


    function treinar() {
      if (!btnTreinar || !btnTestar || !btnPredict || !resultadoPredictDiv || !testResultsDiv) {
        console.error("Um ou mais elementos do DOM não foram encontrados na função treinar.");
        return;
      }

      btnTreinar.disabled = true;
      btnTreinar.textContent = 'Treinando...';
      btnTestar.disabled = true;
      btnPredict.disabled = true; // Desabilitar o botão de submit do formulário também
      resultadoPredictDiv.style.display = 'none';
      resultadoPredictDiv.classList.remove('success', 'error'); // Limpar classes
      testResultsDiv.style.display = 'none';

      fetch('/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
      })
      .then(response => {
        if (!response.ok) {
            // Tenta obter mais detalhes do erro se a resposta for JSON
            return response.json().then(errData => {
                throw new Error(errData.error || `Erro HTTP: ${response.status}`);
            }).catch(() => {
                // Se não for JSON ou falhar ao parsear, usa o status
                throw new Error(`Erro HTTP: ${response.status}`);
            });
        }
        return response.json();
      })
      .then(data => {
        alert(data.message || "Modelos treinados com sucesso!");
        btnTreinar.style.backgroundColor = "green"; // Ou use uma classe CSS
        btnTreinar.textContent = 'Modelos Treinados';
        btnTestar.disabled = false;
        btnPredict.disabled = false; // Habilitar o botão de submit do formulário
      })
      .catch(error => {
        console.error('Erro ao treinar:', error);
        alert(`Erro ao treinar modelos: ${error.message}`);
        btnTreinar.style.backgroundColor = ""; // Volta a cor original do CSS
        btnTreinar.textContent = 'Treinar Modelos';
      }).finally(() => {
        // Reabilita o botão de treino mesmo se houver erro, para permitir nova tentativa,
        // mas mantém o texto como 'Treinar Modelos' se não foi bem sucedido.
        if (btnTreinar.textContent === 'Treinando...') {
            btnTreinar.textContent = 'Treinar Modelos';
        }
        btnTreinar.disabled = false;
      });
    }

    function displayModelResults(modelData, modelName) {
        if (!modelData || !modelData.train_metrics || !modelData.test_metrics) {
            console.error(`Dados incompletos para o modelo ${modelName}`);
            return '<p>Erro: Dados incompletos para exibir resultados.</p>';
        }

        let trainAcc = (modelData.train_metrics.accuracy * 100).toFixed(1);
        let trainPrec = (modelData.train_metrics.precision * 100).toFixed(1);
        let trainRec = (modelData.train_metrics.recall * 100).toFixed(1);
        
        let testAcc = (modelData.test_metrics.accuracy * 100).toFixed(1);
        let testPrec = (modelData.test_metrics.precision * 100).toFixed(1);
        let testRec = (modelData.test_metrics.recall * 100).toFixed(1);

        let modelHTML = `
            <div class="model-results-section">
                <h3>Resultados do Modelo: ${modelName.toUpperCase()}</h3>
                <div class="metrics-grid">
                    <div class="metrics-block">
                        <h4>Métricas de Treinamento (${modelName.toUpperCase()})</h4>
                        <p>Acurácia: ${trainAcc}%</p>
                        <p>Precisão: ${trainPrec}%</p>
                        <p>Recall: ${trainRec}%</p>
                    </div>
                    <div class="metrics-block">
                        <h4>Métricas de Teste (${modelName.toUpperCase()})</h4>
                        <p>Acurácia: ${testAcc}%</p>
                        <p>Precisão: ${testPrec}%</p>
                        <p>Recall: ${testRec}%</p>
                    </div>
                </div>
                <div class="graphs-grid">`;
        
        if (modelData.confusion_matrix_b64) {
            modelHTML += `
                    <div class="graph-container">
                        <h4>Matriz de Confusão (${modelName.toUpperCase()} - Teste)</h4>
                        <img class="graph" src="data:image/png;base64,${modelData.confusion_matrix_b64}" alt="Matriz de Confusão ${modelName}">
                    </div>`;
        } else {
            modelHTML += `<div class="graph-container"><p>Matriz de Confusão não disponível.</p></div>`;
        }
        
        if (modelData.decision_surface_b64) {
            modelHTML += `
                    <div class="graph-container">
                        <h4>Superfície de Decisão (${modelName.toUpperCase()} - Teste)</h4>
                        <img class="graph" src="data:image/png;base64,${modelData.decision_surface_b64}" alt="Superfície de Decisão ${modelName}">
                    </div>`;
        } else {
             modelHTML += `<div class="graph-container"><p>Superfície de Decisão não disponível.</p></div>`;
        }

        modelHTML += `
                </div>
            </div>
        `;
        return modelHTML;
    }

    function testar() {
      if (!testResultsDiv || !loadingDiv || !resultsContentDiv) {
        console.error("Um ou mais elementos do DOM não foram encontrados na função testar.");
        return;
      }
      testResultsDiv.style.display = "block";
      loadingDiv.style.display = "flex"; // Mostrar o spinner
      const spinnerElement = loadingDiv.querySelector('.spinner');
      const loadingTextElement = loadingDiv.querySelector('span');
      if(spinnerElement) spinnerElement.style.display = 'inline-block'; // Garantir que o spinner está visível
      if(loadingTextElement) loadingTextElement.textContent = 'Carregando resultados...'; // Resetar texto do loading
      
      resultsContentDiv.style.display = "none";
      resultsContentDiv.innerHTML = ''; 
      
      fetch('/test')
      .then(response => {
        if (!response.ok) {
            return response.json().then(errData => {
                throw new Error(errData.error || `Erro HTTP: ${response.status}`);
            }).catch(() => {
                throw new Error(`Erro HTTP: ${response.status}`);
            });
        }
        return response.json();
      })
      .then(data => {
        let finalHTML = '';
        if (data.knn) {
            finalHTML += displayModelResults(data.knn, "KNN");
        } else {
            finalHTML += '<p>Resultados do KNN não disponíveis.</p>';
        }
        if (data.svm) {
            finalHTML += displayModelResults(data.svm, "SVM");
        } else {
            finalHTML += '<p>Resultados do SVM não disponíveis.</p>';
        }
        
        resultsContentDiv.innerHTML = finalHTML;
        resultsContentDiv.style.display = "block";
      })
      .catch(error => {
        console.error('Erro ao testar:', error);
        resultsContentDiv.innerHTML = `<p style="color: red; text-align:center;">Erro ao carregar resultados: ${error.message}. Verifique o console.</p>`;
        resultsContentDiv.style.display = "block"; // Mostrar a mensagem de erro
      }).finally(() => {
        loadingDiv.style.display = "none"; // Esconder o spinner e texto de loading
      });
    }

    function handlePredictSubmit(e) {
      e.preventDefault(); // Prevenir o submit padrão do formulário
      
      if (!resultadoPredictDiv) {
        console.error("Elemento resultadoPredictDiv não encontrado.");
        return;
      }

      // Limpar classes anteriores e esconder
      resultadoPredictDiv.classList.remove('success', 'error');
      resultadoPredictDiv.style.display = 'none';
      
      // 'this' dentro de um event listener de formulário refere-se ao próprio formulário
      const formData = new FormData(this); 
      const data = {
        sepal_length: parseFloat(formData.get('sepal_length')),
        sepal_width: parseFloat(formData.get('sepal_width')),
        petal_length: parseFloat(formData.get('petal_length')),
        petal_width: parseFloat(formData.get('petal_width'))
      };
      
      // Validação básica no frontend
      for (const key in data) {
        if (isNaN(data[key])) {
          const friendlyKeyName = key.replace('_', ' ');
          alert(`Por favor, insira um valor numérico válido para ${friendlyKeyName}.`);
          // Adicionar classe de erro e exibir
          resultadoPredictDiv.textContent = `Erro: Valor inválido para ${friendlyKeyName}.`;
          resultadoPredictDiv.classList.add('error');
          resultadoPredictDiv.style.display = "block";
          return; // Interrompe a função
        }
      }

      fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      })
      .then(response => {
        if (!response.ok) {
            // Tenta pegar a mensagem de erro do JSON da resposta
            return response.json().then(err => { 
                throw new Error(err.error || `Erro HTTP: ${response.status}`) 
            });
        }
        return response.json();
      })
      .then(result => {
        resultadoPredictDiv.textContent = (result.predicao ? "Resultado da Predição: " + result.predicao : "Predição não retornada.");
        resultadoPredictDiv.classList.add('success'); // Adiciona classe de sucesso
        resultadoPredictDiv.style.display = "block";
      })
      .catch(error => {
        console.error('Erro na predição:', error);
        resultadoPredictDiv.textContent = "Erro na predição: " + error.message;
        resultadoPredictDiv.classList.add('error'); // Adiciona classe de erro
        resultadoPredictDiv.style.display = "block";
      });
    }
}); // Fim do DOMContentLoaded