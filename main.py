from ultralytics import YOLO

model = YOLO("yolo11x.pt")

train_results = model.train(
    data="coco8.yaml",  # Путь к файлу конфигурации набора данных
    epochs=100,  # Количество периодов обучения
    imgsz=640,  # Размер изображения для тренировки
    device="cpu", # Устройство для запуска (например, 'cpu', 0, [0,1,2,3])

    project="runs/train",   # главная папка результатов
    name="exp",             # имя эксперимента
    exist_ok=True           # не перезаписывать, а создавать exp2, exp3...
)

metrics = model.val()

results = model("path/to/image.jpg")  # Предсказать по изображению
results[0].show()  # Отображение результатов

# Экспортируйте модель в формат ONNX для развертывания
path = model.export(format="onnx")  # Возвращает путь к экспортированной модели
