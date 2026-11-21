"""Acquire checksum-identified official reference resources into a private input tree."""
from pathlib import Path
import sys,argparse,hashlib,gzip,io,zipfile,urllib.request
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
from _paths import INPUT_ROOT,read_table

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=INPUT_ROOT/'references');args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True)
 resources=read_table('reference_resources.tsv').set_index('Resource')
 for key in ['Zhang_workbook','Gavish_RDS']:
  row=resources.loc[key]
  with urllib.request.urlopen(row.Source_URL,timeout=90) as response:body=response.read()
  if key=='Zhang_workbook':
   with zipfile.ZipFile(io.BytesIO(body)) as archive:
    members=[n for n in archive.namelist() if Path(n).name=='41467_2021_25539_MOESM8_ESM.xlsx'];assert len(members)==1
    body=archive.read(members[0])
  assert hashlib.sha256(body).hexdigest()==row.SHA256,'Source bytes changed: '+key
  if key=='Gavish_RDS':body=gzip.decompress(body)
  path=args.output/row.Local_Filename;path.write_bytes(body);print('Retrieved verified resource:',path.name)
if __name__=='__main__':main()
