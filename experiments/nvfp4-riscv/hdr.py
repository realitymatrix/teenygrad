import json, subprocess, sys
BASE="https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4/resolve/main/"
def rng(f,a,b,out):
    r=subprocess.run(["curl","-sL","-m","120","-r",f"{a}-{b}",BASE+f,"-o",out],capture_output=True)
    if r.returncode: sys.exit("curl fail "+r.stderr.decode()[:200])
def header(f):
    rng(f,0,7,"len.bin")
    n=int.from_bytes(open("len.bin","rb").read(8),"little")
    rng(f,8,8+n-1,"hdr.bin")
    return json.loads(open("hdr.bin","rb").read(n)), n+8
want=["model-00005-of-00009.safetensors","model-00001-of-00009.safetensors",
      "model-00008-of-00009.safetensors","model-00003-of-00009.safetensors",
      "model-00006-of-00009.safetensors"]
out={}
for f in want:
    h,off=header(f); out[f]={"data_start":off,"meta":h}
    print(f,"header bytes",off,"tensors",len(h))
json.dump(out,open("headers.json","w"))
