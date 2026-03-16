const IMAGE_SIZE = 224; // your model input size

let session;

async function loadModel() {
    session = await ort.InferenceSession.create("best_model_slim_quant_int8.onnx", {
        executionProviders: ['wasm']
    });

    console.log("Model loaded");
}

loadModel();

document.getElementById("imageInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const predictedClass = await predict(img);
        displayCorrectedImage(img, predictedClass);
    };
});

async function predict(img) {
    const inputTensor = preprocessImage(img);

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data;

    return argmax(output);
}

function preprocessImage(img) {
    const resizeSize = IMAGE_SIZE + 32; // 416

    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = resizeSize;
    resizeCanvas.height = resizeSize;

    const resizeCtx = resizeCanvas.getContext("2d");
    resizeCtx.drawImage(img, 0, 0, resizeSize, resizeSize);

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = IMAGE_SIZE;
    cropCanvas.height = IMAGE_SIZE;

    const cropCtx = cropCanvas.getContext("2d");

    const start = (resizeSize - IMAGE_SIZE) / 2;

    cropCtx.drawImage(
        resizeCanvas,
        start, start,
        IMAGE_SIZE, IMAGE_SIZE,
        0, 0,
        IMAGE_SIZE, IMAGE_SIZE
    );

    const imageData = cropCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = imageData.data;

    const floatData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        floatData[i] = (r - mean[0]) / std[0];
        floatData[i + IMAGE_SIZE * IMAGE_SIZE] = (g - mean[1]) / std[1];
        floatData[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = (b - mean[2]) / std[2];
    }

    return new ort.Tensor("float32", floatData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function argmax(arr) {
    return arr.reduce((maxIdx, val, idx, array) =>
        val > array[maxIdx] ? idx : maxIdx, 0);
}

function displayCorrectedImage(img, orientationClass) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  // rotate-to-upright mapping for your label scheme
  const rotateToUprightDeg = [0, -90, 180, 90][orientationClass] ?? 0;
  const radians = rotateToUprightDeg * Math.PI / 180;

  const swapWH = Math.abs(rotateToUprightDeg) === 90;
  canvas.width = swapWH ? img.height : img.width;
  canvas.height = swapWH ? img.width : img.height;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate(radians);
  ctx.drawImage(img, -img.width / 2, -img.height / 2);
}
