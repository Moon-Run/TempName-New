"""GPU regression: does a cold marker launch block an already-running observer?"""
import ctypes as C
import json,os,sys,time
from pathlib import Path
import torch
import flux  # Load the same frozen libraries as the measurement worker.

assert os.environ.get('SLURM_JOB_ID')
mode=sys.argv[1];assert mode in ('cold','warm')
torch.cuda.set_device(0)
lib=C.CDLL(os.environ['MECH_LIBRARY'])
lib.mech_observe.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
lib.mech_mark.argtypes=[C.c_void_p,C.c_int,C.c_void_p]
def check(code):assert code==0,code
flags=torch.zeros(1,dtype=torch.int32,device='cuda')
indices=torch.zeros(1,dtype=torch.int64,device='cuda')
seen=torch.zeros(1,dtype=torch.int64,device='cuda')
status=torch.zeros(132,dtype=torch.int64,device='cuda')
markers=torch.zeros(2,dtype=torch.int64,device='cuda')
current=torch.cuda.current_stream();observer=torch.cuda.Stream()
# Load observer and the flag publisher without ever waiting for another kernel.
flags.fill_(1);flags.zero_();torch.cuda.synchronize()
check(lib.mech_observe(flags.data_ptr(),indices.data_ptr(),1,seen.data_ptr(),status.data_ptr(),0,observer.cuda_stream))
observer.synchronize()
if mode=='warm':
 for index in (0,1):check(lib.mech_mark(markers.data_ptr(),index,current.cuda_stream))
 torch.cuda.synchronize()
flags.zero_();seen.zero_();status.zero_();torch.cuda.synchronize()
check(lib.mech_observe(flags.data_ptr(),indices.data_ptr(),1,seen.data_ptr(),status.data_ptr(),1,observer.cuda_stream))
start=time.perf_counter()
check(lib.mech_mark(markers.data_ptr(),0,current.cuda_stream))
launch_ms=(time.perf_counter()-start)*1000
flags.fill_(1);torch.cuda.synchronize()
st=status.cpu().tolist()
result=dict(mode=mode,cuda_module_loading=os.environ.get('CUDA_MODULE_LOADING'),
            marker_host_launch_ms=launch_ms,observer_timeout=bool(st[2]),observed=st[3],
            observer_span_ms=(st[1]-st[0])/1e6,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
(Path(os.environ['RESULT_DIR'])/f'startup-{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
# The cold case is diagnostic: eager loading may already avoid the problem.
if mode=='warm':assert not st[2] and st[3]==1,result
