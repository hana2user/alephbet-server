import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

const corsOptions = {
    origin: ['http://localhost:3000', 'http://localhost:5500', 'http://127.0.0.1:5500', 'https://alephbet-ui.onrender.com'],
};

const app = express();
app.use(cors(corsOptions));
app.use(express.json());

const DATA_FILE = 'data.jsonl';
let labelToIndex = {}; // словарь для кодирования меток
let indexToLabel = {}; // обратный словарь для декодирования

// ✅ Добавление примера (изображение 28x28 и метка)
app.post('/api/add-example', async (req, res) => {
  const { image, label } = req.body;

  if (
    !Array.isArray(image) ||
    image.length !== 28 ||
    !Array.isArray(image[0]) ||
    image[0].length !== 28
  ) {
    return res.status(400).send('Неверный формат данных');
  }

  const entry = JSON.stringify({ image, label }) + '\n';

  try {
    await fs.appendFile(DATA_FILE, entry);

    const lines = readFileSync(DATA_FILE, 'utf-8')
      .split('\n')
      .filter(l => l.trim() !== '');

    const labels = {};

    for (const line of lines) {
      const { label } = JSON.parse(line);
      labels[label] = (labels[label] || 0) + 1;
    }

    res.status(200).json({labels});
  } catch (err) {
    console.error('Ошибка записи:', err);
    res.status(500).send('Ошибка сервера');
  }
});

// ✅ Обучение модели
app.post('/api/train', async (req, res) => {
  try {
    const lines = readFileSync(DATA_FILE, 'utf-8')
      .split('\n')
      .filter(l => l.trim() !== '');

    const images = [];
    const labels = [];

    for (const line of lines) {
      const { image, label } = JSON.parse(line);
      images.push(image);
      labels.push(label);
    }

    const imagesExpanded = images.map(img => img.map(row => row.map(val => [val])));
    const x = tf.tensor4d(imagesExpanded, [images.length, 28, 28, 1]).div(255);

    // Кодирование меток в индексы
    const uniqueLabels = [...new Set(labels)];
    labelToIndex = {};
    indexToLabel = {};
    uniqueLabels.forEach((label, i) => {
      labelToIndex[label] = i;
      indexToLabel[i] = label;
    });

    console.log('labelToIndex:', labelToIndex);
    console.log('indexToLabel:', indexToLabel);

    const encodedLabels = labels.map(label => labelToIndex[label]);
    const y = tf.tensor1d(encodedLabels, 'float32');

    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: uniqueLabels.length, activation: 'softmax' }));

    model.compile({
      optimizer: 'adam',
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    await model.fit(x, y, {
      epochs: 20,
      batchSize: 32,
      shuffle: true,
    });

    await model.save('file://model');
    res.send('Модель обучена и сохранена!');
  } catch (err) {
    console.error('Ошибка обучения:', err);
    res.status(500).send('Ошибка обучения');
  }
});

app.post('/api/predict', async (req, res) => {
  try {
    const { image } = req.body;

    const imageExpanded = image.map(row => row.map(v => [v]));
    const input = tf.tensor4d([imageExpanded], [1, 28, 28, 1]);

    const model = await tf.loadLayersModel('file://model/model.json');
    const prediction = model.predict(input);
    const result = prediction.argMax(1).dataSync()[0];
    const label = indexToLabel[result] || 'unknown';

    res.json({ prediction: result, label });
  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).send('Ошибка предсказания');
  }
});


app.post('/api/hello', async (req, res) => {

  res.status(200).send('Hello from server!');
})  

// ✅ Запуск сервера
app.listen(3000, () => {
  console.log('Сервер запущен: http://localhost:3000');
});
