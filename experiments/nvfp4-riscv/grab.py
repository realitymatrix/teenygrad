import json, subprocess, sys
BASE="https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4/resolve/main/"
H=json.load(open("headers.json"))
def find(n):
    for f,v in H.items():
        if n in v["meta"]: return f, v["data_start"], v["meta"][n]
    sys.exit("missing "+n)
def get(name,out,nbytes=None):
    f,ds,m=find(name); a,b=m["data_offsets"]; a+=ds; b+=ds
    if nbytes: b=a+nbytes
    r=subprocess.run(["curl","-sL","-m","300","-r",f"{a}-{b-1}",BASE+f,"-o",out],capture_output=True)
    if r.returncode: sys.exit("curl fail")
    print(f"{out:14s} {m['dtype']:8s} {m['shape']}  got {nbytes or b-a} B")
P="model.llm.layers.20.mlp.experts.w13_weight"
get(P,            "w13_rows.bin", 8*2048)   # expert 0, rows 0..7
get(P+".scale",   "w13_scale.bin", 8*256)   # matching scale rows
get(P+".scale2",  "w13_scale2.bin")
