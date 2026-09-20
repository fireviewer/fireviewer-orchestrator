"""Build from the tested checkout wheel, pinned dependencies and admitted resources."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,tarfile,tomllib

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'ci.json').read_text())['container']

def run(*args,**kwargs):return subprocess.run(args,check=True,**kwargs)
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def prepare():
    from ci import private_dependencies, effective_requirements
    dependencies=private_dependencies()
    target=ROOT/'.ci/container'
    if target.exists():raise SystemExit('Existing container context preserved; use a fresh checkout')
    target.mkdir(parents=True)
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,names):
        if any(name==p or name.startswith(p.rstrip('/')+'/') for p in CONFIG['source_paths']):
            copy(ROOT/name,target/name)
    copy(ROOT/CONFIG['recipe'],target/'Dockerfile')
    for source,name in CONFIG.get('extra_files',{}).items():copy(ROOT/source,target/name)
    if CONFIG.get('resources'):
        entry=CONFIG['resources'];archive=ROOT/'.ci'/entry['asset']
        if not archive.exists():run('gh','release','download',entry['release'],'--repo',entry['repository'],
                                   '--pattern',entry['asset'],'--dir',str(archive.parent))
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=entry['sha256']:raise SystemExit('Native resource archive hash mismatch')
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                # Only admitted native resources, never old Python/UI source.
                if not member.name.startswith(('assets/','vendor/')):continue
                if not member.isfile() or any(p in ('','..','.') for p in member.name.split('/')):raise SystemExit('Unsafe resource member')
                dest=target/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as source,dest.open('xb') as out:shutil.copyfileobj(source,out)
    if CONFIG.get('requirements'):
        for source in dependencies.glob('*.whl'):copy(source,target/'wheels'/source.name)
        project=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
        wheels=list((ROOT/'dist').glob('*.whl'))
        if len(wheels)!=1:raise SystemExit('Build and test the checkout wheel first')
        own=wheels[0];copy(own,target/'wheels'/own.name)
        locked=effective_requirements((ROOT/CONFIG['requirements']).read_text())
        pattern=r'^'+re.escape(project['name'])+r'==[^\n]*(?:\n[ \t]+[^\n]*)*\n?'
        locked=re.sub(pattern,'',locked,flags=re.M)
        locked+='\n'+project['name']+'=='+project['version']+' \\\n    --hash=sha256:'+hashlib.sha256(own.read_bytes()).hexdigest()+'\n'
        (target/'requirements.linux.lock.txt').write_text(locked)
    manifest=[{'path':p.relative_to(target).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
               'bytes':p.stat().st_size} for p in sorted(target.rglob('*')) if p.is_file()]
    report=ROOT/'ci-results';report.mkdir(exist_ok=True)
    (report/'container-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push',action='store_true')
    parser.add_argument('--tag')
    args=parser.parse_args()
    target=prepare()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tag=args.tag or CONFIG['image']+':ci-'+commit[:12]
    if args.push:
        ref=os.environ.get('GITHUB_REF_NAME','')
        manifest=json.loads((ROOT/'dist/release-manifest.json').read_text())
        if ref!='v'+manifest['version'] or manifest['commit']!=commit:raise SystemExit('Only a tested release tag may publish an image')
        run('git','merge-base','--is-ancestor',commit,'origin/main',cwd=ROOT)
        tag=CONFIG['image']+':'+ref
    run('docker','buildx','build','--platform','linux/amd64','--load','--provenance=false',
        '--label','org.opencontainers.image.source=https://github.com/'+CONFIG['repository'],
        '--label','org.opencontainers.image.revision='+commit,'--tag',tag,str(target))
    if args.push:
        run('docker','login','ghcr.io','--username',os.environ['GITHUB_ACTOR'],'--password-stdin',
            input=os.environ['GH_TOKEN'],text=True,stdout=subprocess.DEVNULL)
        try:run('docker','push',tag)
        finally:run('docker','logout','ghcr.io',stdout=subprocess.DEVNULL)
    print(json.dumps({'tag':tag,'commit':commit,'pushed':args.push}))

if __name__=='__main__':main()
