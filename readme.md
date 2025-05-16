# Plataforma Interativa de Classificação de Íris

Esta aplicação web permite a classificação de flores do dataset Íris utilizando múltiplos algoritmos de Machine Learning. Desenvolvida com Python, Flask para o backend, e HTML/CSS/JavaScript para o frontend, a plataforma oferece uma interface interativa para treinar modelos, visualizar seu desempenho através de métricas (acurácia, precisão, recall), matrizes de confusão, superfícies de decisão, e realizar predições para novas amostras de flores.

Atualmente, os seguintes modelos estão implementados:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Árvore de Decisão (Decision Tree)

## Equipe
- [Francisco Eudo da Silva](https://github.com/franciscoeudo)
- [Gabriel Vasconcelos Andrade da Silva](https://github.com/vasconcel)
- [Wagner Fernando Lavandoski Padilha](https://github.com/WagnerLavandoski)

## Pré-requisitos

- Python 3.8 ou superior instalado
- Pip (gerenciador de pacotes do Python)

## Como Configurar e Executar o Projeto

1.  **Clone o Repositório da Equipe:**
    ```bash
    git clone https://github.com/CaririTeam/atividade_desafio_AM
    cd atividade_desafio_AM
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    É altamente recomendável usar um ambiente virtual para isolar as dependências do projeto.

    ```bash
    python -m venv venv
    ```

    Para ativar o ambiente virtual:
    *   No Windows (PowerShell/CMD):
        ```powershell
        .\venv\Scripts\Activate
        ```
    *   No Linux/MacOS:
        ```bash
        source venv/bin/activate
        ```
    Você saberá que o ambiente está ativo se vir `(venv)` no início do seu prompt de comando.

3.  **Instale as Dependências:**
    Com o ambiente virtual ativo, instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a Aplicação Flask:**
    Após a instalação bem-sucedida das dependências, inicie o servidor Flask:
    ```bash
    python back.py
    ```

5.  **Acesse a Aplicação:**
    Abra seu navegador de internet e acesse o seguinte endereço:
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Estrutura do Projeto Atualizada

A estrutura de diretórios principal do projeto é:

```
atividade_desafio_AM/
├── static/
│   ├── assets/               
│   ├── img/
│   │   └── ctt-logo.png      # Logo da equipe
│   ├── js/
│   │   └── script.js         # Lógica do frontend
│   └── estilos.css           # Folha de estilo principal da aplicação
├── templates/
│   └── novapagina.html       # Arquivo HTML principal do frontend
├── venv/                     # Ambiente virtual (gerado localmente, não versionado)
├── back.py                   # Lógica do backend (Flask, modelos de ML)
├── requirements.txt          # Lista de dependências Python
└── README.md                 # Este arquivo
```

## Funcionalidades da Aplicação

A interface principal (`novapagina.html`) permite as seguintes interações:

*   **Seleção de Modelo:** Um menu dropdown permite escolher qual modelo de Machine Learning (KNN, SVM, Árvore de Decisão) será utilizado para teste e predição.
*   **Treinar Todos os Modelos:** Ao clicar no botão "Treinar Todos os Modelos", o backend treina instâncias de KNN, SVM e Árvore de Decisão com o dataset Íris.
*   **Testar Modelo Selecionado:** Após o treinamento, selecione um modelo e clique em "Testar Modelo Selecionado". Isso exibirá:
    *   Métricas de desempenho (Acurácia, Precisão, Recall) para os conjuntos de treinamento e teste do modelo escolhido.
    *   Gráfico da Matriz de Confusão para o conjunto de teste.
    *   Gráfico da Superfície de Decisão (visualizado com base em duas features do dataset) para o conjunto de teste.
*   **Fazer Nova Predição:**
    *   Insira as medidas (comprimento e largura da sépala e da pétala) de uma nova amostra de flor Íris.
    *   Certifique-se de que um modelo esteja selecionado no dropdown.
    *   Clique em "Enviar Valores para Predição".
    *   O sistema usará o modelo selecionado para classificar a flor e exibirá o resultado (espécie predita) juntamente com a acurácia de treinamento do modelo utilizado.

## Tecnologias Utilizadas

*   **Backend:**
    *   Python
    *   Flask (Microframework web)
    *   Scikit-learn (Para os modelos de Machine Learning e métricas)
    *   Matplotlib (Para geração dos gráficos)
    *   NumPy
*   **Frontend:**
    *   HTML5
    *   CSS3 (Layout com Flexbox/Grid, estilização customizada)
    *   JavaScript (Manipulação do DOM, requisições Fetch API para o backend)
*   **Outros:**
    *   Git & GitHub (Controle de versão)
    *   Leonardo.ai (Para a geração da imagem de fundo - mencionar se foi o caso)

---

Este projeto foi desenvolvido como parte da disciplina de Aprendizagem de Máquina, com o objetivo de aplicar conceitos teóricos e práticos em uma aplicação web funcional.
