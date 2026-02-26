import torch

for name in ['CLEVR10', 'COCO', 'MOVi-C', 'MOVi-E']:
    ckpt = torch.load(
        f'checkpoints/slot_attention/adaslot_real/{name}.ckpt', map_location='cpu', weights_only=False)
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    print(f'  Top-level keys: {list(ckpt.keys())}')

    for k, v in ckpt.items():
        if k == 'state_dict':
            continue

        elif k == 'optimizer_states':
            print(f'  optimizer_states: {len(v)} optimizer(s)')
            for i, opt in enumerate(v):
                if 'param_groups' in opt:
                    for pg in opt['param_groups']:
                        simple = {k2: v2 for k2, v2 in pg.items()
                                  if not isinstance(v2, (list, dict, torch.Tensor))}
                        print(f'    param_group[{i}]: {simple}')

        elif k == 'callbacks':
            print(f'  callbacks ({type(v).__name__}):')
            if isinstance(v, dict):
                for ck, cv in v.items():
                    if isinstance(cv, dict):
                        print(f'    [{ck}]')
                        for ck2, cv2 in cv.items():
                            if not isinstance(cv2, torch.Tensor):
                                print(f'      {ck2}: {cv2}')
                    else:
                        print(f'    {ck}: {cv}')

        elif k == 'hyper_parameters':
            print(f'  hyper_parameters: {v}')

        elif k == 'hparams_file':
            print(f'  hparams_file: {v}')

        else:
            print(f'  {k}: {v}')
