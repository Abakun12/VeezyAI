from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np  # Để tạo embedding mẫu
import os  # Để lấy biến môi trường (tùy chọn)

# --- Cấu hình Kết nối MongoDB ---
# Sử dụng biến môi trường hoặc hardcode chuỗi kết nối của bạn
# CẢNH BÁO BẢO MẬT: KHÔNG NÊN HARDCODE CREDENTIALS TRONG CODE CHO PRODUCTION
MONGO_CONN_STR = os.environ.get("MONGO_CONN_STR",
                                "mongodb+srv://thuanchce170133:ZEQ16jqwjtolxbaV@cluster0.ajszsll.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "veezy_db")
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "Accounts")

client = None  # Khởi tạo client là None

try:
    client = MongoClient(MONGO_CONN_STR, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print(f"\nĐã chọn database '{MONGO_DB_NAME}' và collection '{MONGO_COLLECTION_NAME}'.")

    # --- CHỌN ACCOUNT VÀ THÊM EMBEDDING ---

    # 1. Xác định tài khoản bạn muốn cập nhật
    # Bạn có thể tìm bằng user_id, username, email, hoặc bất kỳ trường duy nhất nào khác
    # Ví dụ: tìm tài khoản có username là 'alice_w'
    target_username = "9fb58e7c-2aa5-499d-947e-9007e28a9019"  # THAY THẾ BẰNG USERNAME HOẶC ID BẠN MUỐN CẬP NHẬT

    # Hoặc nếu bạn muốn tạo mới nếu chưa có, rồi thêm embedding:
    # collection.update_one(
    #     {"username": target_username},
    #     {"$setOnInsert": {"name": "Alice Wonderland From Update", "email": "alice_new@example.com"}},
    #     upsert=True
    # )
    # print(f"Đã đảm bảo user '{target_username}' tồn tại.")

    # 2. Tạo một embedding mẫu (hoặc lấy embedding thật từ Service AI của bạn)
    sample_embedding_for_target = np.random.rand(128).tolist()  # Embedding 128 chiều

    # 3. Tạo lệnh cập nhật để thêm trường 'embedding'
    # Sử dụng toán tử $set để thêm trường mới hoặc cập nhật trường đã có
    update_operation = {
        "$set": {
            "embedding": sample_embedding_for_target,
            "embedding_added_at": "2025-06-05T14:00:00Z"  # (Tùy chọn) Thêm thời gian cập nhật
        }
    }

    # 4. Thực hiện cập nhật cho document khớp với tiêu chí
    print(f"\nĐang tìm và cập nhật embedding cho tài khoản có username: '{target_username}'...")
    update_result = collection.update_one(
        {"username": target_username},  # Điều kiện tìm kiếm (filter)
        update_operation
    )

    # 5. Kiểm tra kết quả cập nhật
    if update_result.matched_count > 0:
        print(f"Đã tìm thấy {update_result.matched_count} tài khoản khớp với username '{target_username}'.")
        if update_result.modified_count > 0:
            print(f"Đã cập nhật thành công trường 'embedding' cho tài khoản.")
            # Lấy lại document đã cập nhật để xem
            updated_account = collection.find_one({"username": target_username})
            print("\nTài khoản sau khi cập nhật embedding:")
            if updated_account:
                for key, value in updated_account.items():
                    if key == "embedding":
                        print(f"  {key}: [ {str(value[0])[:7]} ..., {str(value[-1])[:7]} ] (chiều: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("Không thể lấy lại document đã cập nhật.")
        else:
            print(
                "Tài khoản đã được tìm thấy nhưng không có trường nào được thay đổi (có thể embedding đã tồn tại và giống hệt).")
    else:
        print(f"Không tìm thấy tài khoản nào có username '{target_username}' để cập nhật embedding.")
        print("Vui lòng kiểm tra lại username hoặc đảm bảo tài khoản đó đã tồn tại trong collection.")
        print("Bạn có thể thêm một tài khoản mẫu với username này trước khi chạy lại script.")
        # Ví dụ cách thêm tài khoản mẫu nếu chưa có (chỉ chạy nếu bạn muốn tạo mới):
        # try:
        #     collection.insert_one({"username": target_username, "name": "Người Dùng Mẫu Cho Embedding"})
        #     print(f"Đã tạo tài khoản mẫu với username: {target_username}. Hãy chạy lại script để thêm embedding.")
        # except Exception as insert_e:
        #     print(f"Lỗi khi tạo tài khoản mẫu: {insert_e}")


except Exception as e:
    print(f"\nLỗi xảy ra: {e}")
finally:
    if client:
        client.close()
        print("\nĐã đóng kết nối MongoDB.")