from datasets import load_dataset
import os

print("🚀 Downloading Wikitext-2...")

# Load ONLY Wikitext-2 (smaller than Wikitext-103)
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

print("✅ Dataset downloaded!")
print(f"Dataset structure: {ds}")

# Save to text files
for split_name, dataset in ds.items():
    filename = f"wikitext2_{split_name}.txt"
    print(f"📁 Saving {split_name} split to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            text = example['text']
            if text.strip():  # Only write non-empty lines
                f.write(text + '\n')
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} lines...")
    
    print(f"✅ Saved {len(dataset)} examples to {filename}")

print("🎉 All files saved!")
print("📊 File sizes:")
for split_name in ds.keys():
    filename = f"wikitext2_{split_name}.txt"
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB")