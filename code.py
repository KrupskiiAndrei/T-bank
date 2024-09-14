import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Устанавливаем устройство (GPU, если доступен)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка предобученной модели сегментации (DeepLabV3)
segmentation_model = deeplabv3_resnet101(pretrained=True).to(device).eval()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_background(image_path):
    # Загрузка изображения и применение сегментации
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    mask = output.argmax(0).cpu().numpy()
    
    # Создаем маску для фона и применяем её к изображению
    mask = (mask == 15).astype(np.uint8)  # Класс 'person' в COCO - 15
    mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    background_removed = np.array(image) * mask[:, :, np.newaxis]

    return background_removed, mask

def change_background(image, mask, background_color=(200, 200, 200)):
    # Замена фона на указанный цвет
    new_background = np.full_like(image, background_color)
    new_image = np.where(mask[:, :, np.newaxis] == 1, image, new_background)
    return new_image

# Замена фона
def process_image(image_path, background_color=(200, 200, 200)):
    image, mask = remove_background(image_path)
    return change_background(image, mask, background_color)

# Шаг 3: Генерация описания
def generate_description(prompt):
    # Загрузка предобученной модели GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt2large')
    model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt2large').to(device)
    
    # Генерация описания
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

# Пример использования
if __name__ == "__main__":
    # Удаление фона и замена его на белый
    image_path = 'path/to/image.jpg'
    processed_image = process_image(image_path, background_color=(255, 255, 255))
    
    # Сохранение результата
    cv2.imwrite('processed_image.jpg', processed_image)
    
    # Генерация описания товара
    prompt = "Описание товара: красивый и стильный аксессуар."
    description = generate_description(prompt)
    print("Описание товара:", description)
