const IMAGE_SIZE = 224;
const MODEL_URL = "orientation_model_best.onnx"; // FP32 (straight) ONNX

let session;

async function loadModel() {
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
  });
  console.log("Model loaded:", MODEL_URL);
}

function letterboxToSquareCanvas(img, size, fill = 255) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = `rgb(${fill},${fill},${fill})`;
  ctx.fillRect(0, 0, size, size);

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;

  const scale = Math.min(size / w, size / h);
  const nw = Math.max(1, Math.round(w * scale));
  const nh = Math.max(1, Math.round(h * scale));

  const left = Math.floor((size - nw) / 2);
  const top = Math.floor((size - nh) / 2);

  ctx.drawImage(img, left, top, nw, nh);
  return canvas;
}

function preprocessImage(img) {
  const canvas = letterboxToSquareCanvas(img, IMAGE_SIZE, 255);
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
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

async function predict(img) {
  if (!session) await loadModel();

  const inputTensor = preprocessImage(img);
  const feeds = { [session.inputNames[0]]: inputTensor };

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  return argmax(output);
}

function argmax(arr) {
  return arr.reduce((maxIdx, val, idx, array) => (val > array[maxIdx] ? idx : maxIdx), 0);
}

function displayCorrectedImage(img, orientationClass) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  const rotateToUprightDeg = [0, -90, 180, 90][orientationClass] ?? 0;
  const radians = (rotateToUprightDeg * Math.PI) / 180;

  const swapWH = Math.abs(rotateToUprightDeg) === 90;
  canvas.width = swapWH ? img.height : img.width;
  canvas.height = swapWH ? img.width : img.height;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate(radians);
  ctx.drawImage(img, -img.width / 2, -img.height / 2);
}

loadModel();

document.getElementById("imageInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const predictedClass = await predict(img);
    displayCorrectedImage(img, predictedClass);
  };
});
