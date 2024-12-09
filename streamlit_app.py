import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

class_names = {0: "Стул", 1: "Диван", 2: "Стол"}

# URL вашего BentoML-сервиса
API_URL = "http://127.0.0.1:3333/detect"

st.title("Lab 3.2 | YOLOv8 Object Detection")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    # Отправка изображения в BentoML
    if st.button("Распознать объекты"):
        try:
            # Преобразование изображения в поток байтов
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            
            files = {
                'image': ('image', img_byte_arr, 'image/jpeg')  # или 'image/png' в зависимости от типа файла
            }
            
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                results = response.json()  # Считываем результаты
                detections = results  # Предположим, что сервер возвращает список объектов
                
                # Открываем изображение для рисования
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()  # Можно заменить на TTF шрифт для улучшения качества

                # Рисуем рамки и пишем классы
                for det in detections:
                    class_id = det["class"]
                    class_name = class_names.get(class_id, "Неизвестный класс")
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    #class_name = det["class_name"]
                    confidence = det["confidence"]
                    
                    # Рисуем прямоугольник
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

                    # Добавление текста с названием класса
                    text = f"{class_name} ({confidence:.2f})"
                    draw.text((x1, y1), text, fill="white", font=font)
                
                # Отображаем изображение с аннотациями
                st.image(image, caption="Распознанные объекты", use_container_width=True)
    
                st.write(f"Класс: {class_name}, Уверенность: {confidence:.2f}")
                
            else:
                st.error(f"Ошибка распознавания: Статус-код {response.status_code}")
                st.write(f"Текст ошибки: {response.text}")
                
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
