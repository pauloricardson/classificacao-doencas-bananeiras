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
        statusElemento.textContent = "Carregando o modelo...";

        const [modeloCarregado, respostaClasses] = await Promise.all([
            tf.loadLayersModel("./modelo/model.json"),
            fetch("./modelo/classes.json")
        ]);

        if (!respostaClasses.ok) {
            throw new Error("Não foi possível carregar o classes.json.");
        }

        modelo = modeloCarregado;
        classes = await respostaClasses.json();

        console.log("Entrada do modelo:", modelo.inputs[0].shape);
        console.log("Classes:", classes);

        statusElemento.textContent =
            "Modelo carregado. Selecione uma imagem.";
    } catch (erro) {
        console.error(erro);

        statusElemento.textContent =
            "Erro ao carregar o modelo. Verifique os arquivos do projeto.";
    }
}

imagemInput.addEventListener("change", function (evento) {
    const arquivo = evento.target.files[0];

    if (!arquivo) {
        return;
    }

    if (!arquivo.type.startsWith("image/")) {
        statusElemento.textContent = "Selecione um arquivo de imagem válido.";
        return;
    }

    const enderecoTemporario = URL.createObjectURL(arquivo);

    preview.onload = function () {
        URL.revokeObjectURL(enderecoTemporario);

        preview.style.display = "block";
        resultadoElemento.classList.add("oculto");

        classificarButton.disabled = modelo === null;

        statusElemento.textContent =
            "Imagem selecionada. Toque em classificar.";
    };

    preview.src = enderecoTemporario;
});

classificarButton.addEventListener("click", classificarImagem);

async function classificarImagem() {
    if (!modelo || !preview.src) {
        return;
    }

    classificarButton.disabled = true;
    statusElemento.textContent = "Analisando a imagem...";

    try {
        const predicao = tf.tidy(() => {
            const altura = modelo.inputs[0].shape[1];
            const largura = modelo.inputs[0].shape[2];

            let tensor = tf.browser.fromPixels(preview, 3);

            tensor = tf.image.resizeBilinear(
                tensor,
                [altura, largura]
            );

            tensor = tensor.toFloat();

            /*
             * Pré-processamento padrão da MobileNetV2:
             * pixels de 0–255 são convertidos para -1–1.
             */
            tensor = tensor.div(127.5).sub(1);

            tensor = tensor.expandDims(0);

            return modelo.predict(tensor);
        });

        const probabilidades = await predicao.data();
        predicao.dispose();

        let melhorIndice = 0;

        for (let i = 1; i < probabilidades.length; i++) {
            if (probabilidades[i] > probabilidades[melhorIndice]) {
                melhorIndice = i;
            }
        }

        const classePrevista = classes[melhorIndice];
        const confianca = probabilidades[melhorIndice] * 100;

        classePrevistaElemento.textContent =
            traduzirClasse(classePrevista);

        confiancaElemento.textContent =
            `${confianca.toFixed(2)}%`;

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
        console.error(erro);

        statusElemento.textContent =
            "Ocorreu um erro durante a classificação.";
    } finally {
        classificarButton.disabled = false;
    }
}

function traduzirClasse(classe) {
    const traducoes = {
        "Banana Black Sigatoka Disease": "Sigatoka-negra",
        "Banana Bract Mosaic Virus Disease":
            "Vírus do mosaico das brácteas",
        "Banana Healthy Leaf": "Folha saudável",
        "Banana Insect Pest Disease":
            "Danos causados por insetos",
        "Banana Moko Disease": "Moko da bananeira",
        "Banana Panama Disease": "Mal-do-Panamá",
        "Banana Yellow Sigatoka Disease": "Sigatoka-amarela"
    };

    return traducoes[classe] || classe;
}

carregarAplicacao();