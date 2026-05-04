import cv2
import os
from process_cv import ImageProcessor

def main():
    input_file = r'C:\ComputerVision-main\CapturedImage\input_2.jpg'
    
    # 1. Kiểm tra xem file ảnh có tồn tại không
    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file '{input_file}' trong thư mục hiện tại.")
        print("💡 Gợi ý: Hãy copy một bức ảnh chụp cuốn sổ vào thư mục này và đổi tên thành 'input.png'.")
        return

    print(f"📸 Đang đọc ảnh từ '{input_file}'...")
    # Đọc ảnh gốc
    frame = cv2.imread(input_file)

    # 2. Khởi tạo class xử lý ảnh của bạn
    processor = ImageProcessor()

    # 3. Định nghĩa các bước cần test để xem sự thay đổi
    steps_to_test = ['edges', 'contours', 'warp', 'all']

    print("="*50)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE XỬ LÝ ẢNH")
    print("="*50)

    for step in steps_to_test:
        print(f"\n⚙️ Đang chạy bước: [{step.upper()}]...")
        try:
            # Truyền một bản sao (frame.copy()) để các bước không ghi đè lên ảnh gốc của nhau
            processed_img, results, process_time = processor.process_frame(frame.copy(), step=step)
            
            # --- In log ra Terminal ---
            print(f"   ⏱ Thời gian xử lý: {process_time:.2f} ms")
            print(f"   📝 Trạng thái: {results.get('status')}")
            
            # Nếu thuật toán có tính ra Ma trận Homography, in cho đẹp
            if 'homography_matrix' in results:
                print("   📐 Ma trận Homography H (3x3):")
                for row in results['homography_matrix']:
                    print(f"      [ {row[0]:>9.4f}, {row[1]:>9.4f}, {row[2]:>9.4f} ]")
            
            # Nếu thuật toán báo lỗi (ví dụ: không tìm thấy tờ giấy)
            if results.get('status') == 'failed':
                print(f"   ⚠️ Thông báo: {results.get('message')}")

            # --- Lưu kết quả ---
            output_filename = f'output_{step}.png'
            cv2.imwrite(output_filename, processed_img)
            print(f"   💾 Đã lưu kết quả ra file: '{output_filename}'")
            
        except Exception as e:
            print(f"   ❌ Gặp lỗi nghiêm trọng khi chạy bước '{step}': {e}")

    print("\n" + "="*50)
    print("✅ HOÀN TẤT TEST! Hãy kiểm tra các file 'output_*.png' trong thư mục.")

if __name__ == "__main__":
    main()