let modelo = null;
let classes = [];

const TAMANHO_IMAGEM = 224;

const imagemInput = document.getElementById("imagemInput");
const preview = document.getElementById("preview");
const classificarButton = document.getElementById("classificarButton");
const statusElemento = document.getElementById("status");
const resultadoElemento = document.getElementById("resultado");
const classePrevistaElemento = document.getElementById("classePrevista");
const confiancaElemento = document.getElementById("confianca");

/*
=========================================================
CARREGAMENTO DO MODELO E DAS CLASSES
=========================================================
*/

async function carregarAplicacao() {
    try {
        if (typeof tf === "undefined") {
            throw new Error(
                "TensorFlow.js não foi carregado. Verifique o index.html."
            );
        }

        statusElemento.textContent = "Carregando as classes...";

        const respostaClasses = await fetch("./modelo/classes.json");

        if (!respostaClasses.ok) {
            throw new Error(
                `Não foi possível carregar classes.json. Código: ${respostaClasses.status}`
            );
        }

        classes = await respostaClasses.json();

        if (!Array.isArray(classes) || classes.length === 0) {
            throw new Error(
                "O arquivo classes.json está vazio ou possui formato inválido."
            );
        }

        console.log("Classes carregadas:", classes);

        statusElemento.textContent = "Carregando o modelo...";

        modelo = await tf.loadGraphModel(
            "./modelo/model.json",
            {
                onProgress: function (progresso) {
                    const percentual = Math.round(progresso * 100);

                    statusElemento.textContent =
                        `Carregando o modelo: ${percentual}%`;
                }
            }
        );

        console.log("Modelo carregado:", modelo);
        console.log("Entradas do modelo:", modelo.inputs);
        console.log("Saídas do modelo:", modelo.outputs);

        statusElemento.textContent =
            "Modelo carregado. Selecione uma imagem.";

        if (preview.src) {
            classificarButton.disabled = false;
        }
    } catch (erro) {
        console.error("Erro ao carregar aplicação:", erro);

        statusElemento.textContent =
            `Erro ao carregar: ${erro.message}`;
    }
}

/*
=========================================================
SELEÇÃO DA IMAGEM
=========================================================
*/

imagemInput.addEventListener("change", function (evento) {
    const arquivo = evento.target.files[0];

    if (!arquivo) {
        return;
    }

    if (!arquivo.type.startsWith("image/")) {
        statusElemento.textContent =
            "Selecione um arquivo de imagem válido.";

        classificarButton.disabled = true;
        return;
    }

    resultadoElemento.classList.add("oculto");

    const enderecoTemporario = URL.createObjectURL(arquivo);

    preview.onload = function () {
        URL.revokeObjectURL(enderecoTemporario);

        preview.style.display = "block";

        if (modelo) {
            classificarButton.disabled = false;

            statusElemento.textContent =
                "Imagem selecionada. Toque em classificar.";
        } else {
            classificarButton.disabled = true;

            statusElemento.textContent =
                "Imagem selecionada. Aguarde o carregamento do modelo.";
        }
    };

    preview.onerror = function () {
        URL.revokeObjectURL(enderecoTemporario);

        statusElemento.textContent =
            "Não foi possível abrir a imagem selecionada.";

        classificarButton.disabled = true;
    };

    preview.src = enderecoTemporario;
});

/*
=========================================================
BOTÃO DE CLASSIFICAÇÃO
=========================================================
*/

classificarButton.addEventListener(
    "click",
    classificarImagem
);

/*
=========================================================
CLASSIFICAÇÃO DA IMAGEM
=========================================================
*/

async function classificarImagem() {
    if (!modelo) {
        statusElemento.textContent =
            "O modelo ainda não foi carregado.";

        return;
    }

    if (!preview.src) {
        statusElemento.textContent =
            "Selecione uma imagem antes de classificar.";

        return;
    }

    classificarButton.disabled = true;
    statusElemento.textContent = "Analisando a imagem...";

    let tensorEntrada = null;
    let predicao = null;

    try {
        /*
         * Preparação da imagem:
         *
         * 1. Converte a imagem para tensor.
         * 2. Redimensiona para 224 × 224.
         * 3. Converte os valores para float.
         * 4. Aplica preprocess_input da MobileNetV2.
         * 5. Adiciona a dimensão do lote.
         */

        tensorEntrada = tf.tidy(() => {
            let tensor = tf.browser.fromPixels(
                preview,
                3
            );

            tensor = tf.image.resizeBilinear(
                tensor,
                [TAMANHO_IMAGEM, TAMANHO_IMAGEM],
                true
            );

            tensor = tensor.toFloat();

            /*
             * Mesmo preprocess_input da MobileNetV2:
             *
             * pixel / 127.5 - 1
             *
             * Intervalo final: aproximadamente -1 até 1.
             */

            tensor = tensor.div(127.5).sub(1);

            tensor = tensor.expandDims(0);

            return tensor;
        });

        console.log(
            "Formato da entrada:",
            tensorEntrada.shape
        );

        /*
         * Executa o GraphModel.
         *
         * Alguns modelos retornam um tensor.
         * Outros podem retornar uma lista de tensores
         * ou um objeto com tensores.
         */

        const resultado = modelo.execute(tensorEntrada);

        if (resultado instanceof tf.Tensor) {
            predicao = resultado;
        } else if (Array.isArray(resultado)) {
            if (resultado.length === 0) {
                throw new Error(
                    "O modelo não retornou nenhuma saída."
                );
            }

            predicao = resultado[0];

            for (let i = 1; i < resultado.length; i++) {
                resultado[i].dispose();
            }
        } else if (
            resultado !== null &&
            typeof resultado === "object"
        ) {
            const nomesSaidas = Object.keys(resultado);

            if (nomesSaidas.length === 0) {
                throw new Error(
                    "O modelo retornou um objeto sem saídas."
                );
            }

            const nomePrimeiraSaida = nomesSaidas[0];

            predicao = resultado[nomePrimeiraSaida];

            for (let i = 1; i < nomesSaidas.length; i++) {
                resultado[nomesSaidas[i]].dispose();
            }
        } else {
            throw new Error(
                "O formato da saída do modelo não foi reconhecido."
            );
        }

        if (!(predicao instanceof tf.Tensor)) {
            throw new Error(
                "A saída principal do modelo não é um tensor."
            );
        }

        console.log(
            "Formato da saída:",
            predicao.shape
        );

        const probabilidades = await predicao.data();

        if (probabilidades.length !== classes.length) {
            throw new Error(
                `O modelo retornou ${probabilidades.length} valores, ` +
                `mas o classes.json possui ${classes.length} classes.`
            );
        }

        let melhorIndice = 0;

        for (let i = 1; i < probabilidades.length; i++) {
            if (
                probabilidades[i] >
                probabilidades[melhorIndice]
            ) {
                melhorIndice = i;
            }
        }

        const classeOriginal = classes[melhorIndice];

        const classeTraduzida =
            traduzirClasse(classeOriginal);

        const confianca =
            probabilidades[melhorIndice] * 100;

        classePrevistaElemento.textContent =
            classeTraduzida;

        confiancaElemento.textContent =
            `${confianca.toFixed(2)}%`;

        resultadoElemento.classList.remove("oculto");

        statusElemento.textContent =
            "Classificação concluída.";

        console.table(
            classes.map((classe, indice) => ({
                classe: classe,
                probabilidade:
                    `${(
                        probabilidades[indice] * 100
                    ).toFixed(2)}%`
            }))
        );
    } catch (erro) {
        console.error(
            "Erro durante a classificação:",
            erro
        );

        statusElemento.textContent =
            `Erro durante a classificação: ${erro.message}`;
    } finally {
        if (tensorEntrada) {
            tensorEntrada.dispose();
        }

        if (predicao) {
            predicao.dispose();
        }

        classificarButton.disabled = false;
    }
}

/*
=========================================================
TRADUÇÃO DOS NOMES DAS CLASSES
=========================================================
*/

function traduzirClasse(classe) {
    const traducoes = {
        "Banana Black Sigatoka Disease":
            "Sigatoka-negra",

        "Banana Bract Mosaic Virus Disease":
            "Vírus do mosaico das brácteas",

        "Banana Healthy Leaf":
            "Folha saudável",

        "Banana Insect Pest Disease":
            "Danos causados por insetos",

        "Banana Moko Disease":
            "Moko da bananeira",

        "Banana Panama Disease":
            "Mal-do-Panamá",

        "Banana Yellow Sigatoka Disease":
            "Sigatoka-amarela"
    };

    return traducoes[classe] || classe;
}

/*
=========================================================
INICIALIZAÇÃO
=========================================================
*/

carregarAplicacao();