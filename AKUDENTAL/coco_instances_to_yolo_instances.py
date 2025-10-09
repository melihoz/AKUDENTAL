import json
import shutil
from pathlib import Path
from tqdm import tqdm

def create_yolo_datasets_for_all_folds(base_path: Path):
    """
    Creates five organized folders for 5-fold cross-validation with YOLO.

    For each fold (0-4), this script generates a complete dataset directory
    containing organized images, YOLO-formatted labels, and a corresponding
    dataset.yaml file.

    Args:
        base_path (Path): The root directory of the AKUDENTAL dataset where
                          images and the 'metadata' folder are located.
    """
    # --- 1. Define Common Paths & Load Data ---
    source_images_dir = base_path
    coco_json_path = base_path / "metadata" / "akudental_instances.json"
    split_info_path = base_path / "metadata" / "split_info_recreated.json"

    print("🚀 Starting 5-fold dataset creation process...")
    print("🔄 Loading annotation and split data (this happens only once)...")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    # Create mappings for efficient lookup (done once)
    images_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    class_names_ordered = sorted([cat['name'] for cat in coco_data['categories']])
    class_name_to_id = {name: i for i, name in enumerate(class_names_ordered)}
    coco_cat_id_to_yolo_id = {
        cat['id']: class_name_to_id[cat['name']]
        for cat in coco_data['categories']
    }

    # --- 2. Loop Through Each Fold and Create Its Dataset ---
    for i in range(5):
        fold_name = f"fold_{i}"
        output_base_path = base_path / f"AKUDENTAL_YOLO_{fold_name.upper()}"
        
        print(f"\n{'='*20} PROCESSING {fold_name.upper()} {'='*20}")
        print(f"Output will be in: {output_base_path}")

        if output_base_path.exists():
            shutil.rmtree(output_base_path)

        # Create the directory structure for the current fold
        for split in ["train", "val", "test"]:
            (output_base_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_base_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        fold_splits = split_info.get(fold_name)
        if not fold_splits:
            print(f"❌ Error: '{fold_name}' not found in split_info.json. Skipping.")
            continue

        for split_name, image_files in fold_splits.items():
            print(f"  -> Processing '{split_name}' split for {fold_name}...")
            for image_filename in tqdm(image_files, desc=f"     Creating files"):
                # Find image info
                image_id = next((i_id for i_id, i_info in images_info.items() if i_info['file_name'] == image_filename), None)
                if image_id is None: continue

                # Copy Image File
                source_image_path = source_images_dir / image_filename
                dest_image_path = output_base_path / "images" / split_name / image_filename
                if source_image_path.exists():
                    shutil.copy(source_image_path, dest_image_path)
                else: continue

                # Generate and Save Label File
                img_info = images_info[image_id]
                img_width, img_height = img_info['width'], img_info['height']
                
                yolo_annotations = []
                if image_id in annotations_by_image_id:
                    for ann in annotations_by_image_id[image_id]:
                        if ann['category_id'] not in coco_cat_id_to_yolo_id: continue
                        yolo_class_id = coco_cat_id_to_yolo_id[ann['category_id']]
                        segmentation = ann['segmentation'][0]
                        normalized_points = [
                            coord / (img_width if i % 2 == 0 else img_height)
                            for i, coord in enumerate(segmentation)
                        ]
                        points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                        yolo_annotations.append(f"{yolo_class_id} {points_str}")

                label_filename = Path(image_filename).stem + ".txt"
                dest_label_path = output_base_path / "labels" / split_name / label_filename
                with open(dest_label_path, 'w') as f:
                    f.write("\n".join(yolo_annotations))

        # Create the dataset.yaml for the current fold
        yaml_content = f"""
# Dataset configuration for {fold_name.upper()}
path: {output_base_path.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names_ordered)}
names:
"""
        for j, name in enumerate(class_names_ordered):
            yaml_content += f"  {j}: {name}\n"
        with open(output_base_path / "dataset.yaml", 'w') as f:
            f.write(yaml_content)
        print(f"  -> `dataset.yaml` for {fold_name} created successfully.")

    print(f"\n{'='*20} ALL FOLDS PROCESSED {'='*20}")
    print("\n✅ Success! Your 5 YOLO dataset folders are ready.")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Set this to the root path of your AKUDENTAL dataset.
    dataset_base_path = Path("D:/dental/AKUDENTALlast") # <-- CHANGE THIS PATH

    if not dataset_base_path.is_dir():
        print(f"❌ Error: The path '{dataset_base_path}' does not exist.")
    else:
        create_yolo_datasets_for_all_folds(dataset_base_path)