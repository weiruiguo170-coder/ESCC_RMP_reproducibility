"""Verify frozen inputs and reproduce the reported statistics without rendering figures."""
from pathlib import Path
import argparse,os,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,help='Optional directory for regenerated verification tables');args=p.parse_args()
    temporary=tempfile.TemporaryDirectory(prefix='escc_verify_') if args.output is None else None
    output=Path(temporary.name) if temporary else args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',ESCC_OUTPUT_ROOT=str(output),MPLCONFIGDIR=str(output/'.matplotlib'))
    commands=['validation/inputs.py','evidence/grid.py','validation/discovery.py','replay.py','validation/statistics.py','validation/day15.py','validation/clinical.py','validation/framework.py','validation/figures.py']
    try:
        for name in commands:
            proc=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name)],cwd=ROOT,env=env,text=True,encoding='utf-8',errors='replace')
            if proc.returncode:raise SystemExit(proc.returncode)
        print('All scientific verification stages passed. Raw fits and figure rendering were not executed.')
    finally:
        if temporary:temporary.cleanup()
if __name__=='__main__':main()
