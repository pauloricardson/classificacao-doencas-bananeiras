let modelo = null;
let classes = [];

const imagemInput = document.getElementById("imagemInput");
const preview = document.getElementById("preview");
const classificarButton = document.getElementById("classificarButton");
const statusElemento = document.getElementById("status");
const resultadoElemento = document.getElementById("resultado");
const classePrevistaElemento = document.getElementById("classePrevista");
const confiancaElemento = document.getElementById("confianca");

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
                `Erro ao carregar classes.json: ${respostaClasses.status}`
            );
        }

        classes = await respostaClasses.json();

        console.log("Classes carregadas:", classes);

        if (!Array.isArray(classes) || classes.length === 0) {
            throw new Error(
                "O arquivo classes.json está vazio ou possui formato inválido."
            );
        }

        statusElemento.textContent =
            "Classes carregadas. Baixando o modelo...";

        modelo = await tf.loadLayersModel(
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
        console.log("Entrada:", modelo.inputs[0].shape);
        console.log("Saída:", modelo.outputs[0].shape);

        const quantidadeSaidas = modelo.outputs[0].shape[1];

        if (classes.length !== quantidadeSaidas) {
            throw new Error(
                `O modelo possui ${quantidadeSaidas} saídas, ` +
                `mas classes.json possui ${classes.length} classes.`
            );
        }

        statusElemento.textContent =
            "Modelo carregado. Selecione uma imagem.";

        if (preview.src) {
            classificarButton.disabled = false;
        }
    } catch (erro) {
        console.error("Erro no carregamento:", erro);

        statusElemento.textContent =
            `Erro ao carregar: ${erro.message}`;
    }
}

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

    const enderecoTemporario = URL.createObjectURL(arquivo);

    preview.onload = function () {
        URL.revokeObjectURL(enderecoTemporario);

        preview.style.display = "block";
        resultadoElemento.classList.add("oculto");

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

classificarButton.addEventListener("click", classificarImagem);

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

    let predicao = null;

    try {
        predicao = tf.tidy(() => {
            const formatoEntrada = modelo.inputs[0].shape;

            const altura = formatoEntrada[1];
            const largura = formatoEntrada[2];

            if (!altura || !largura) {
                throw new Error(
                    "Não foi possível identificar o tamanho de entrada do modelo."
                );
            }

            let tensor = tf.browser.fromPixels(preview, 3);

            tensor = tf.image.resizeBilinear(
                tensor,
                [altura, largura],
                true
            );

            tensor = tensor.toFloat();

            /*
             * Mesmo preprocess_input usado pela MobileNetV2 no treinamento:
             *
             * pixel / 127.5 - 1
             *
             * Isso converte os pixels de 0–255 para o intervalo -1–1.
             */
            tensor = tensor.div(127.5).sub(1);

            tensor = tensor.expandDims(0);

            const resultado = modelo.predict(tensor);

            if (Array.isArray(resultado)) {
                return resultado[0];
            }

            return resultado;
        });

        const probabilidades = await predicao.data();

        if (probabilidades.length !== classes.length) {
            throw new Error(
                `A previsão retornou ${probabilidades.length} valores, ` +
                `mas existem ${classes.length} classes.`
            );
        }

        let melhorIndice = 0;

        for (let i = 1; i < probabilidades.length; i++) {
            if (probabilidades[i] > probabilidades[melhorIndice]) {
                melhorIndice = i;
            }
        }

        const classeOriginal = classes[melhorIndice];
        const classeTraduzida = traduzirClasse(classeOriginal);
        const confianca = probabilidades[melhorIndice] * 100;

        classePrevistaElemento.textContent = classeTraduzida;
        confiancaElemento.textContent = `${confianca.toFixed(2)}%`;

        resultadoElemento.classList.remove("oculto");

        statusElemento.textContent = "Classificação concluída.";

        console.table(
            classes.map((classe, indice) => ({
                classe,
                probabilidade:
                    `${(probabilidades[indice] * 100).toFixed(2)}%`
            }))
        );
    } catch (erro) {
        console.error("Erro durante a classificação:", erro);

        statusElemento.textContent =
            `Erro durante a classificação: ${erro.message}`;
    } finally {
        if (predicao) {
            predicao.dispose();
        }

        classificarButton.disabled = false;
    }
}

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

carregarAplicacao();