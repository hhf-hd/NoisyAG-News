import os
import shutil
import zipfile


def unzipBert():
    # 定义路径
    source_dir = '/root/autodl-pub/BERT-Pretrain-Model'
    target_dir = '/root/autodl-tmp'
    zip_file = os.path.join(source_dir, 'bert-base-uncased.zip')
    extracted_folder = os.path.join(target_dir, 'bert-base-uncased')

    # 1. 检查目标文件夹是否存在
    if not os.path.exists(extracted_folder):
        # 2. 拷贝文件
        if os.path.exists(zip_file):
            shutil.copy(zip_file, target_dir)  # 将 zip 文件拷贝到目标目录
            print(f"Copied {zip_file} to {target_dir}")

            # 3. 解压缩 zip 文件cd ..
            
            with zipfile.ZipFile(os.path.join(target_dir, 'bert-base-uncased.zip'), 'r') as zip_ref:
                zip_ref.extractall(target_dir)  # 解压到目标目录
                print(f"Extracted {zip_file} to {target_dir}")
        else:
            print(f"File {zip_file} does not exist.")
    else:
        print(f"The folder {extracted_folder} already exists.")

