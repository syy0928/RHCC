import os


def rename_files_by_number(folder_path):
    # 获取文件夹中所有文件的列表
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 按数字顺序重命名文件
    for i, filename in enumerate(files, 1):
        # 获取文件扩展名
        file_ext = os.path.splitext(filename)[1]

        # 构建新文件名
        new_filename = f"{i}{file_ext}"

        # 构建完整的旧路径和新路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"已将 '{filename}' 重命名为 '{new_filename}'")


if __name__ == "__main__":
    # 请将下面的路径替换为你的目标文件夹路径
    target_folder = "C:/Users/syy/Desktop/input_folder_infer/input_folder_infer"

    # 检查路径是否存在
    if os.path.exists(target_folder) and os.path.isdir(target_folder):
        rename_files_by_number(target_folder)
        print("所有文件已按数字顺序重命名完成！")
    else:
        print(f"错误：路径 '{target_folder}' 不存在或不是一个文件夹")
