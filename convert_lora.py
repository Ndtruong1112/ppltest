import torch
import os
from safetensors.torch import save_file
from peft import LoraConfig

# Đường dẫn đến folder lora_adapter của bạn
adapter_path = "D:/Users/Admin/Downloads/PhoMT/results/lora_adapter"
bin_file = os.path.join(adapter_path, "adapter_model.bin")
safe_file = os.path.join(adapter_path, "adapter_model.safetensors")

print(f"--- ĐANG CHUYỂN ĐỔI: {bin_file} ---")

# Buộc phải dùng cờ này để nạp file bin cũ 1 lần cuối
os.environ["HF_SKIP_PROTECTED_LOAD"] = "1"

try:
    # Nạp trọng số từ file .bin
    state_dict = torch.load(bin_file, map_location="cpu")
    
    # Lưu lại dưới dạng .safetensors
    save_file(state_dict, safe_file)
    print(f"✅ THÀNH CÔNG! Đã tạo file: {safe_file}")
    print("💡 Giờ bạn có thể xóa file adapter_model.bin để tránh bị hỏi lỗi nữa.")
except Exception as e:
    print(f"❌ Lỗi: {e}")