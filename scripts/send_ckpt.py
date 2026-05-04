from pathlib import Path
from huggingface_hub import HfApi

folder = '/work/gr41/r41000/openpi/checkpoints/pi0_aic_cheatcode_lerobot/restful-feather-14/50000'
repo_id = 'uzumibi/aic_pi0_restful-feather-14_50000'
token = Path('/work/gr41/r41000/.cache/huggingface/token').read_text().strip()
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
print(f'repo_id={repo_id}')
print('repo ready')
res = api.upload_folder(
    repo_id=repo_id,
    folder_path=folder,
    path_in_repo='.',
    ignore_patterns=['optimizer.pt'],
    commit_message='Upload without optimizer.pt',
)
print('upload_result=', res)
print('repo_url=', f'https://huggingface.co/{repo_id}')