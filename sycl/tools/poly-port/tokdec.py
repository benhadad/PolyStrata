import sys
try:
    from gguf import GGUFReader
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "gguf"])
    from gguf import GGUFReader
p = "/mnt/c/AI_Data/models/Inkling-2-Mini-downloads/GGUF/Inkling-2-Mini-UD-Q3_K_XL.gguf"
r = GGUFReader(p)
f = r.fields["tokenizer.ggml.tokens"]
def tok(i):
    return bytes(f.parts[f.data[i]]).decode("utf-8", "replace")
ids = [int(x) for x in sys.argv[1].split(",")]
print(" | ".join("%d=%r" % (i, tok(i)) for i in ids))
