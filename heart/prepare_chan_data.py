# heart/prepare_data.py
"""
准备心脏数据集 - 提取所有通道并整理文件结构
使用tifffile读取多页TIFF
Channel 0: DAPI (nuclei)
Channel 1: ALDH1A2 (epicardial cell)
Channel 2: WGA (cell membrane)
Channel 3: CD45 (immune cell)
Channel 4: PDGFRB (mural cells)
"""
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import tifffile
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_heart import HeartConfig

# 通道定义
CHANNELS = {
    0: {'name': 'dapi', 'description': 'DAPI (nuclei)'},
    1: {'name': 'aldh1a2', 'description': 'ALDH1A2 (epicardial cell)'},
    2: {'name': 'wga', 'description': 'WGA (cell membrane)'},
    3: {'name': 'cd45', 'description': 'CD45 (immune cell)'},
    4: {'name': 'pdgfrb', 'description': 'PDGFRB (mural cells)'}
}

def load_multichannel_tiff(image_path):
    """使用tifffile加载多通道TIFF"""
    with tifffile.TiffFile(image_path) as tif:
        data = tif.asarray()
        if data.ndim == 3 and data.shape[0] == 5:
            return data
        else:
            raise ValueError(f"Unexpected shape: {data.shape}, expected (5, H, W)")

def save_channel_tiff(channel_data, output_path):
    """保存单通道为TIFF"""
    tifffile.imwrite(output_path, channel_data)

def split_channels(image_path, output_dir, area_name):
    """拆分多通道图像为单独的TIFF文件
    
    文件命名: {area}_{channel}.tif
    例如: LA1_dapi.tif, LA1_aldh1a2.tif
    """
    # 加载图像
    img_array = load_multichannel_tiff(image_path)
    n_channels, height, width = img_array.shape
    
    saved_files = {}
    channel_stats = []
    
    # 保存每个通道
    for ch_idx, ch_info in CHANNELS.items():
        channel_data = img_array[ch_idx, :, :]
        channel_name = ch_info['name']
        
        # 输出文件名: {area}_{channel}.tif
        output_filename = f"{area_name}_{channel_name}.tif"
        output_path = output_dir / output_filename
        
        # 保存为TIFF
        save_channel_tiff(channel_data, output_path)
        
        saved_files[channel_name] = str(output_path)
        
        # 统计信息
        n_nonzero = np.count_nonzero(channel_data)
        pct_nonzero = n_nonzero / channel_data.size * 100
        
        channel_stats.append({
            'channel': channel_name,
            'min': int(channel_data.min()),
            'max': int(channel_data.max()),
            'mean': float(channel_data.mean()),
            'std': float(channel_data.std()),
            'nonzero_pct': float(pct_nonzero)
        })
    
    return saved_files, channel_stats

def main():
    config = HeartConfig()
    
    print("=" * 70)
    print("🔬 Preparing Heart Dataset - Extracting Channels from Multi-page TIFF")
    print("=" * 70)
    
    # 创建processed目录
    processed_dir = config.RAW_DIR.parent / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Input:  {config.RAW_DIR}")
    print(f"📂 Output: {processed_dir}")
    
    print(f"\n📺 Channels to extract:")
    for ch_idx, ch_info in CHANNELS.items():
        print(f"  Channel {ch_idx}: {ch_info['description']}")
    
    # 加载mapping
    mapping_df = pd.read_csv(config.MAPPING_FILE)
    
    # 获取所有唯一图像（每个area只处理一次）
    unique_images = mapping_df.groupby(['region', 'area']).first().reset_index()
    
    print(f"\n📊 Found {len(unique_images)} unique images to process")
    print(f"📊 Covering {len(mapping_df)} annotation regions")
    print(f"\n💡 File naming: {{area}}_{{channel}}.tif")
    print(f"   Example: LA1_dapi.tif, LA1_aldh1a2.tif")
    
    # 为每个区域创建输出目录
    for region in config.REGIONS:
        (processed_dir / region).mkdir(parents=True, exist_ok=True)
    
    all_data_info = []
    all_channel_stats = []
    failed = []
    
    # 处理每个图像
    for idx, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="Extracting"):
        region = row['region']
        area = row['area']  # e.g., LA1, RA2
        image_path = Path(row['image_absolute_path'])
        
        try:
            # 拆分通道
            region_output_dir = processed_dir / region
            saved_files, channel_stats = split_channels(image_path, region_output_dir, area)
            
            # 记录文件信息
            all_data_info.append({
                'region': region,
                'area': area,
                'original_image': str(image_path),
                **saved_files  # dapi, aldh1a2, wga, cd45, pdgfrb paths
            })
            
            # 记录统计信息
            for stat in channel_stats:
                stat.update({
                    'region': region,
                    'area': area
                })
                all_channel_stats.append(stat)
            
        except Exception as e:
            failed.append({'region': region, 'area': area, 'error': str(e)})
            print(f"\n  ✗ Failed {region}/{area}: {e}")
            continue
    
    # 保存数据信息
    if all_data_info:
        data_info_df = pd.DataFrame(all_data_info)
        data_info_csv = processed_dir / 'data_info.csv'
        data_info_df.to_csv(data_info_csv, index=False)
        
        # 保存通道统计
        channel_stats_df = pd.DataFrame(all_channel_stats)
        channel_stats_csv = processed_dir / 'channel_statistics.csv'
        channel_stats_df.to_csv(channel_stats_csv, index=False)
    else:
        print("\n❌ No data extracted!")
        return
    
    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Successfully processed: {len(all_data_info)}/{len(unique_images)} images")
    print(f"📁 Total channel files created: {len(all_data_info) * 5}")
    
    if failed:
        print(f"\n⚠️  Failed: {len(failed)} images")
        for f in failed:
            print(f"  • {f['region']}/{f['area']}: {f['error']}")
    
    print(f"\n🫀 Files by region:")
    for region in config.REGIONS:
        region_count = len(data_info_df[data_info_df['region'] == region])
        if region_count > 0:
            print(f"  • {region}: {region_count} images × 5 channels = {region_count * 5} files")
    
    # 通道质量统计
    print(f"\n📊 Channel Statistics (across all images):")
    print(f"\n{'Channel':<40} {'Mean±STD':>20} {'Max':>10} {'NonZero%':>10}")
    print(f"{'-'*82}")
    
    for ch_idx, ch_info in CHANNELS.items():
        ch_name = ch_info['name']
        ch_stats = channel_stats_df[channel_stats_df['channel'] == ch_name]
        if len(ch_stats) > 0:
            mean_val = ch_stats['mean'].mean()
            std_val = ch_stats['mean'].std()
            max_val = ch_stats['max'].max()
            nonzero = ch_stats['nonzero_pct'].mean()
            print(f"{ch_info['description']:<40} "
                  f"{mean_val:>8.1f}±{std_val:<8.1f} {max_val:>10} {nonzero:>9.1f}%")
    
    print(f"\n💾 Data info saved to: {data_info_csv}")
    print(f"💾 Channel stats saved to: {channel_stats_csv}")
    
    # 创建完整的mapping文件（链接到GT）
    print("\n📝 Creating complete data mapping with ground truth...")
    
    complete_mapping = []
    
    for _, data_row in data_info_df.iterrows():
        region = data_row['region']
        area = data_row['area']
        
        # 为每个cell type创建映射
        for cell_type in config.CELL_TYPES:
            gt_mask_path = config.GT_DIR / region / f"{cell_type}-{area}_mask.npy"
            
            if gt_mask_path.exists():
                complete_mapping.append({
                    'region': region,
                    'area': area,
                    'cell_type': cell_type,
                    'dapi': data_row['dapi'],
                    'aldh1a2': data_row['aldh1a2'],
                    'wga': data_row['wga'],
                    'cd45': data_row['cd45'],
                    'pdgfrb': data_row['pdgfrb'],
                    'gt_nuclei_mask': str(gt_mask_path)
                })
    
    complete_mapping_df = pd.DataFrame(complete_mapping)
    complete_mapping_csv = processed_dir / 'complete_mapping.csv'
    complete_mapping_df.to_csv(complete_mapping_csv, index=False)
    
    print(f"✅ Complete mapping created: {complete_mapping_csv}")
    print(f"   {len(complete_mapping_df)} entries (image × cell_type combinations)")
    
    # 创建README
    readme_path = processed_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# Heart Dataset - Processed Data\n\n")
        f.write("## Channel Information\n\n")
        for ch_idx, ch_info in CHANNELS.items():
            f.write(f"- **Channel {ch_idx}**: {ch_info['description']}\n")
        f.write("\n## File Naming Convention\n\n")
        f.write("Files are named as: `{area}_{channel}.tif`\n\n")
        f.write("Examples:\n")
        f.write("- `LA1_dapi.tif` - Left Atrium area 1, DAPI channel\n")
        f.write("- `LA1_cd45.tif` - Left Atrium area 1, CD45 channel\n")
        f.write("- `RV2_aldh1a2.tif` - Right Ventricle area 2, ALDH1A2 channel\n")
        f.write("\n## File Structure\n\n")
        f.write("```\n")
        f.write("processed/\n")
        f.write("├── LA/\n")
        f.write("│   ├── LA1_dapi.tif\n")
        f.write("│   ├── LA1_aldh1a2.tif\n")
        f.write("│   ├── LA1_wga.tif\n")
        f.write("│   ├── LA1_cd45.tif\n")
        f.write("│   ├── LA1_pdgfrb.tif\n")
        f.write("│   └── ...\n")
        f.write("├── RA/, LV/, RV/, SEP/\n")
        f.write("├── data_info.csv            # All extracted channel files\n")
        f.write("├── channel_statistics.csv   # Channel quality statistics\n")
        f.write("├── complete_mapping.csv     # Links to ground truth masks\n")
        f.write("└── README.md\n")
        f.write("```\n\n")
        f.write("## Data Files\n\n")
        f.write(f"- **data_info.csv**: {len(data_info_df)} images with all channel paths\n")
        f.write(f"- **complete_mapping.csv**: {len(complete_mapping_df)} entries linking channels to GT masks\n")
        f.write(f"- **channel_statistics.csv**: Quality metrics for each channel in each image\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total images: {len(data_info_df)}\n")
        f.write(f"- Total channel files: {len(data_info_df) * 5}\n")
        f.write(f"- Regions: {', '.join(config.REGIONS)}\n")
        f.write(f"- Cell types with GT: {', '.join(config.CELL_TYPES)}\n")
    
    print(f"📄 README created: {readme_path}")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    print("\n📂 Output structure:")
    print(f"  {processed_dir}/")
    print(f"    ├── LA/, RA/, LV/, RV/, SEP/  (channel TIF files)")
    print(f"    │   └── {{area}}_{{channel}}.tif")
    print(f"    ├── data_info.csv             (file paths)")
    print(f"    ├── channel_statistics.csv    (quality metrics)")
    print(f"    ├── complete_mapping.csv      (links to GT)")
    print(f"    └── README.md")
    print("\n📊 Next steps:")
    print("  1. Run find_gt_channel.py to identify which marker corresponds to each cell type")
    print("  2. Use processed/*_dapi.tif for nuclei segmentation")
    print("  3. Use processed/*_[marker].tif for cell segmentation")
    print("=" * 70)

if __name__ == "__main__":
    main()