from pathlib import Path

template_dir = Path('templates')
required = ['index.html', 'clinical_form.html', 'clinical_result.html', 'symptoms_form.html', 'symptoms_result.html']

print(f"Template directory exists: {template_dir.exists()}")
if template_dir.exists():
    print("Files found:")
    for file in required:
        path = template_dir / file
        print(f"- {file}: {path.exists()}")
else:
    print("Error: 'templates' directory not found in:")
    print(Path.cwd())