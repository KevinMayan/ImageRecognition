# Reconhecimento de Objetos com ImageAI

Este projeto utiliza a biblioteca [ImageAI](https://github.com/OlafenwaMoses/ImageAI) para realizar o reconhecimento de objetos em imagens, com foco em itens para compra. O código usa o modelo TinyYOLOv3 para detectar e identificar objetos presentes na imagem de entrada.

## Pré-requisitos

- Python 3.6 ou superior
- TensorFlow 1.13 ou superior
- Keras 2.2.4
- ImageAI 2.1.5 ou superior

## Instalação

1. Clone este repositório:
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Instale as dependências:
    ```bash
    pip install tensorflow keras imageai
    ```

3. Baixe o modelo TinyYOLOv3:
    - [TinyYOLOv3](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5)

## Utilização

1. Coloque o modelo `yolo-tiny.h5` no diretório `Model`.

2. Adicione a imagem de entrada no diretório `InputImage`.

3. Execute o código:
    ```python
    from imageai.Detection import ObjectDetection
    
    recognizer = ObjectDetection()
    
    path_model = "./Model/yolo-tiny.h5"
    path_input = "./InputImage/images.jpg"
    path_output = "./OutputImage/newimage.jpg"
    
    recognizer.setModelTypeAsTinyYOLOv3()
    recognizer.setModelPath(path_model)
    recognizer.loadModel()
    
    recognition = recognizer.detectObjectsFromImage(
        input_image = path_input,
        output_image_path = path_output
    )
    
    for eachItem in recognition:
        print(eachItem["name"], ":", eachItem["percentage_probability"])
    ```

4. O código irá gerar uma nova imagem com os objetos reconhecidos no diretório `OutputImage` e imprimir no console os nomes dos objetos e a probabilidade de reconhecimento.

## Exemplo de Saída

```plaintext
bottle : 85.67
cell phone : 78.45
laptop : 95.12
