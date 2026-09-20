#!/bin/bash
# Заливка проекта на арендованный сервер.
# Использование: ./upload.sh root@адрес.сервера

SERVER=$1
REMOTE=/root/scoring

if [ -z "$SERVER" ]; then
  echo "Укажи сервер: ./upload.sh root@адрес"
  exit 1
fi

echo ">> Создаю структуру каталогов"
ssh $SERVER "mkdir -p $REMOTE/{data/raw,data/processed,models,app}"

echo ">> Код (без __pycache__)"
rsync -avz --progress \
  --exclude='__pycache__' --exclude='*.pyc' \
  app/ $SERVER:$REMOTE/app/

echo ">> Модель классификатора"
rsync -avz --progress models/microcat_classifier.joblib $SERVER:$REMOTE/models/

echo ">> Сырые данные"
rsync -avz --progress \
  data/raw/train.parquet \
  data/raw/benchmark_items.parquet \
  data/raw/benchmark_queries.parquet \
  $SERVER:$REMOTE/data/raw/

echo ">> Объединение"
rsync -avz --progress data/processed/all_items.parquet $SERVER:$REMOTE/data/processed/

echo ">> Матрицы эмбеддингов (самое долгое)"
rsync -avz --progress \
  data/processed/e5_union_embeddings.npz \
  data/processed/frida_union_embeddings.npz \
  $SERVER:$REMOTE/data/processed/

echo ">> Готово"