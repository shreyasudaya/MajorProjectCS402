import nbformat

notebook_path = "unlearnpipeline.ipynb"  # replace with your notebook

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Fix metadata.widgets
widgets_meta = nb.metadata.get("widgets", {})
if "state" not in widgets_meta:
    widgets_meta["state"] = {}  # or set to None if that’s more appropriate
    nb.metadata["widgets"] = widgets_meta

# Save the fixed notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook metadata fixed!")
