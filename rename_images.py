import os

folders = [
    (r"E:\FINAL_YEAR_PROJECT\ORIGINAL_DATASET_Stroke Vs Normal - Copy\Brain_Stroke_CT-SCAN_image\val\normal", ""),              # 1,2,3...
    (r"E:\FINAL_YEAR_PROJECT\ORIGINAL_DATASET_Stroke Vs Normal - Copy\Brain_Stroke_CT-SCAN_image\val\stroke", ""),              # 1,2,3...
    (r"E:\FINAL_YEAR_PROJECT\ORIGINAL_DATASET_StrokeType - Copy\Brain_Stroke_CT-SCAN_image\val\ischaemic", "isc_scan__"),    # isc_scan__1...
    (r"E:\FINAL_YEAR_PROJECT\ORIGINAL_DATASET_StrokeType - Copy\Brain_Stroke_CT-SCAN_image\val\hemorrhagic", "hem_scan__")     # hem_scan__1...
]

for folder_path, prefix in folders:
    files = os.listdir(folder_path)
    
    # Step 1: Rename all to temporary names (to avoid overwrite issues)
    temp_files = []
    for i, file in enumerate(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            ext = os.path.splitext(file)[1]
            temp_name = f"temp_{i}{ext}"
            
            old_path = os.path.join(folder_path, file)
            temp_path = os.path.join(folder_path, temp_name)
            
            os.rename(old_path, temp_path)
            temp_files.append(temp_name)

    # Step 2: Rename from temp → final names
    count = 1
    for file in temp_files:
        ext = os.path.splitext(file)[1]
        
        if prefix == "":
            new_name = f"{count}{ext}"
        else:
            new_name = f"{prefix}{count}{ext}"
        
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        count += 1

    print(f"Done: {folder_path}")

print("All folders renamed successfully!")