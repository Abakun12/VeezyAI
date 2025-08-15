# File: embedding_service.py

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Import kiến trúc model trực tiếp từ thư viện
from facenet_pytorch import InceptionResnetV1

class CustomEmbeddingService:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding Service is using device: {self.device}")

        # 1. Khởi tạo kiến trúc model InceptionResnetV1 (không dùng pre-trained)
        self.model = InceptionResnetV1(pretrained=None, classify=False).to(self.device)

        # 2. Tải các trọng số đã được train, dùng strict=False như trong code của bạn
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        # 3. Chuyển model sang chế độ đánh giá
        self.model.eval()

        # 4. Định nghĩa các bước tiền xử lý (COPY 100% TỪ CODE TRAIN CỦA BẠN)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print("Preprocessing pipeline configured successfully.")

    def get_embedding(self, image_np):
        """
        Trích xuất vector embedding từ một ảnh numpy array (khuôn mặt đã cắt).
        """
        # Chuyển đổi từ BGR (OpenCV) sang RGB (PIL)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Áp dụng các bước tiền xử lý
        image_tensor = self.transform(image_pil).to(self.device)
        image_tensor = image_tensor.unsqueeze(0) # Thêm chiều batch

        # Trích xuất embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)

        # Chuyển về numpy array
        return embedding.squeeze().cpu().numpy()