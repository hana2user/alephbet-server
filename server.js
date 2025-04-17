import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

const corsOptions = {
    origin: ['http://localhost:3000', 'http://localhost:5500', 'http://127.0.0.1:5500'],
};

const app = express();
app.use(cors(corsOptions));
app.use(express.json());

const DATA_FILE = 'data.jsonl';

// ✅ Добавление примера (изображение 28x28 и метка)
app.post('/api/add-example', async (req, res) => {
  const { image, label } = req.body;

  if (
    !Array.isArray(image) ||
    image.length !== 28 ||
    !Array.isArray(image[0]) ||
    image[0].length !== 28
    // typeof label !== 'number' ||
    // ![0, 1, 2].includes(label)
  ) {
    return res.status(400).send('Неверный формат данных');
  }

  const entry = JSON.stringify({ image, label }) + '\n';

  try {
    await fs.appendFile(DATA_FILE, entry);
    res.sendStatus(200);
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
    const y = tf.tensor1d(labels, 'float32');

    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
      optimizer: 'adam',
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    await model.fit(x, y, {
      epochs: 5,
      batchSize: 32,
      shuffle: true,
    });

    await model.save('downloads://model');
    res.send('Модель обучена и сохранена!');
  } catch (err) {
    console.error('Ошибка обучения:', err);
    res.status(500).send('Ошибка обучения');
  }
});

app.post('/api/predict', async (req, res) => {
    try {
      const { image } = req.body; // [28][28]
  
      const x = tf.tensor4d([image], [1, 28, 28, 1]).div(255);
      const model = await tf.loadLayersModel('file://model/model.json');
  
      const prediction = model.predict(x);
      const result = prediction.argMax(1).dataSync()[0];
  
      res.json({ prediction: result });
    } catch (err) {
      console.error(err);
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
