import numpy as np, soundfile as sf, json, warnings; warnings.filterwarnings('ignore')
from kokoro_onnx import Kokoro
from script import INTRO, WALK
k=Kokoro('kokoro-v1.0.onnx','voices-v1.0.bin'); VOICE='af_heart'; SPD=0.97
out={}
def gen(name,txt):
    x,sr=k.create(txt,voice=VOICE,speed=SPD,lang='en-us'); x=x.astype(np.float32)
    # trim leading/trailing silence
    a=np.abs(x); idx=np.nonzero(a>.01)[0]; x=x[max(idx[0]-int(.02*sr),0):idx[-1]+int(.05*sr)]
    sf.write(f'vo_{name}.wav',x,sr); out[name]=len(x)/sr; print(name,round(out[name],2),txt[:50])
for i,t in enumerate(INTRO): gen(f'i{i}',t)
for n,t in WALK.items(): gen(f'w{n}',t)
json.dump(out,open('vo_durations.json','w'))
