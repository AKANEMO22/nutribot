import re

with open('streamlit_assets/embedded_fpt_build/assets/index-BSKeajN2.js', 'r', encoding='utf-8') as f:
    text = f.read()

matches = set(re.findall(r'className:"[^"]*text-2xl[^"]*"', text))
print("Found text-2xl instances:")
for m in matches:
    print(m)
