content = open("scripts/verify_pipelines.py", encoding="utf-8").read()
# Find and print the current disease_mapping lines to confirm
for i, line in enumerate(content.splitlines()):
    if "disease_mapping" in line.lower():
        print(i, repr(line))
print("---")
old = 'disease_mapping_path=Path("data/processed/disease_mapping.json"),'
new = 'disease_mapping_path=Path("data/processed/disease_mapping_41.json"),'
if old in content:
    content = content.replace(old, new)
    open("scripts/verify_pipelines.py", "w", encoding="utf-8").write(content)
    print("Patched OK: disease_mapping.json -> disease_mapping_41.json")
else:
    print("Pattern NOT found. Check current content above.")
