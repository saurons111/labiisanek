# Лабораторная работа №5 — Классификация изображений животных (PyTorch)

## Данные
- Локальный датасет `date/animals/`
- Классы: **cheetah**, **leopard**, **lion**, **tiger**
- Структура: `animals/<класс>/<картинки>.jpg`

## Предобработка
- Проверка битых файлов через `PIL.Image.verify()`
- Приведение всех изображений к размеру **224×224**
- Нормализация по каналам (ImageNet):
  - `mean = (0.485, 0.456, 0.406)`
  - `std  = (0.229, 0.224, 0.225)`
- Разделение: **80% train / 20% val**
- `ImageFolder` + `Subset` → `DataLoader(batch_size=32)`

## Аугментации (только для train)
- `RandomHorizontalFlip()`
- `RandomRotation(10°)`
- Resize → ToTensor → Normalize

## Модель
Простая сверточная сеть **SimpleCNN**:
- Блоки: Conv2d → ReLU → MaxPool (3 раза, каналы 3→16→32→64)
- После свёрток: `Flatten`
- Полносвязный классификатор:
  - `Linear(n_features → 128)` → ReLU → Dropout(0.3)
  - `Linear(128 → 4)` (по числу классов)
- Loss: `CrossEntropyLoss`
- Оптимизатор: `Adam(lr=1e-3)`
- Эпох: **10**

## Результаты
- Accuracy на валидации: **≈ 0.67**
- По классам:
  - **lion**, **tiger** — распознаются лучше всего (более высокий recall / f1)
  - **cheetah** и **leopard** чаще путаются между собой (похожий пятнистый окрас)
- Кривые обучения:
  - train_loss < val_loss, train_acc > val_acc → лёгкое переобучение

## Возможные улучшения
- Более сильные аугментации (crop, color jitter и т.п.)
- Более сложная архитектура или transfer learning (ResNet и др.)
- Дольше обучать + подобрать lr, scheduler, weight decay
- При необходимости использовать веса классов или балансировку выборки
