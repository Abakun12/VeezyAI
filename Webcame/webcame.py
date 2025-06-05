import cv2
import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def collect_data(name, label, max_images=30):
    base_dir = "dataset"
    save_dir = os.path.join(base_dir, f"{name}_{label}")
    create_folder(save_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    count = 0

    instructions = [
        "Huong dan thu thap anh:",
        "- Nhìn thang vao camera",
        "- Xoay dau trai/phai nhe",
        "- Nguoc len, cui xuong",
        "- Doi bieu cam (cuoi, nghiem mat)",
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Khong the lay frame tu camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Ve khung xanh quanh khuon mat
        face_img = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))

        # Hien thi so luong anh da chup
        cv2.putText(frame, f"Da thu thap: {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hien thi huong dan
        y0, dy = 60, 25
        for i, line in enumerate(instructions):
            y = y0 + i*dy
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(f"Collecting data - {name} ({label})", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if face_img is not None:
                img_path = os.path.join(save_dir, f"{name}_{label}_{count+1}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                print(f"[INFO] Da luu anh thu {count}")
            else:
                print("[WARNING] Khong phat hien khuon mat, vui long dieu chinh lai!")

        elif key == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Thu thap du lieu hoan thanh cho {name} - {label}")

if __name__ == "__main__":
    person_name = input("Nhap ten nguoi thu thap: ")
    label = input("Nhap nhan (mask / no_mask): ")
    collect_data(person_name, label, max_images=30)
