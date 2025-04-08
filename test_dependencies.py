# test_dependencies.py
import ftfy
from lxml import etree

print(f"ftfy version: {ftfy.__version__}")  # Should be ≥6.1.1
print(f"lxml version: {etree.__version__}")  # Should be ≥4.9.3
print("BytesIO available:", hasattr(ftfy, 'fix_bytes'))
