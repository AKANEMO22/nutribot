import re

filepath = 'streamlit_assets/embedded_fpt_build/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Add dashboard-sync.js before </head>
if "dashboard-sync.js" not in text:
    text = text.replace("</head>", '    <script src="/assets/dashboard-sync.js"></script>\n    </head>')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Patched successfully!")
else:
    print("Already patched.")
