import os
import json

import torch

device_cache = None

def get_device():
    global device_cache
    if device_cache is None:
        override = os.environ.get("MOTIFIESTA_DEVICE", "").strip()
        if override:
            device_cache = torch.device(override)
        elif torch.cuda.is_available():
            device_cache = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_cache = torch.device("mps")
        else:
            device_cache = torch.device("cpu")
        print(f"using device: {device_cache}")
    return device_cache

def load_data(run, batch_size=2, background_only=False):
    with open(f'models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)
    data = dataset_from_json(json_params)
    return data

def load_model(run, permissive=False, verbose=True):
    """
    Input the name of a run
    :param run:
    :return:
    """
    with open(f'models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)

    model = model_from_json(json_params)

    try:
        model_dict = torch.load(f'models/{run}/{run}.pth',
                                map_location='cpu')
        state_dict = model_dict['model_state_dict']

        # backward compat: EdgePool gin changed from single GINConv to ModuleList.
        # remap pool_layers.N.gin.X → pool_layers.N.gin.0.X
        if any(k.startswith('pool_layers.') and '.gin.nn.' in k and '.gin.0.' not in k
               for k in state_dict):
            remapped = {}
            for k, v in state_dict.items():
                if '.gin.nn.' in k and '.gin.0.' not in k:
                    remapped[k.replace('.gin.', '.gin.0.')] = v
                elif k.endswith('.gin.eps') and '.gin.0.' not in k:
                    remapped[k.replace('.gin.eps', '.gin.0.eps')] = v
                else:
                    remapped[k] = v
            state_dict = remapped

        # backward compat: dual-proj checkpoints used transform_hi/transform_lo;
        # remap transform_hi → transform and drop transform_lo / gate_net
        if any('.transform_hi.weight' in k for k in state_dict):
            fresh = model.state_dict()
            remapped = {}
            for k, v in state_dict.items():
                if '.transform_hi.' in k:
                    remapped[k.replace('.transform_hi.', '.transform.')] = v
                elif '.transform_lo.' in k or '.gate_net.' in k:
                    pass  # discard dual-proj-only keys
                else:
                    remapped[k] = v
            for k in fresh:
                if k not in remapped:
                    remapped[k] = fresh[k]
            state_dict = remapped

        # drop score_net keys if architecture changed (sequential vs linear, or size mismatch)
        # fresh init is fine since score_net weights are not transferable across architectures
        fresh = model.state_dict()
        needs_fresh_score_net = False
        if hasattr(model, 'layers') and len(model.layers) > 0 and hasattr(model.layers[0], 'score_net'):
            cur_layer = model.layers[0]
            ckpt_has_linear = 'layers.0.score_net.weight' in state_dict
            ckpt_has_seq = 'layers.0.score_net.0.weight' in state_dict
            cur_is_seq = isinstance(cur_layer.score_net, torch.nn.Sequential)
            if (ckpt_has_linear and cur_is_seq) or (ckpt_has_seq and not cur_is_seq):
                needs_fresh_score_net = True
            elif ckpt_has_linear:
                ckpt_in = state_dict['layers.0.score_net.weight'].shape[1]
                cur_in = cur_layer.score_net.weight.shape[1]
                if ckpt_in != cur_in:
                    needs_fresh_score_net = True
        if needs_fresh_score_net:
            for k in list(state_dict.keys()):
                if 'score_net' in k:
                    del state_dict[k]
            for k, v in fresh.items():
                if 'score_net' in k:
                    state_dict[k] = v

        # replace any remaining checkpoint keys whose shape doesn't match current model
        # (e.g. pool_layers.0.transform after dual-pathway arch change: dim×dim → dim×n_features)
        for k in list(state_dict.keys()):
            if k in fresh and state_dict[k].shape != fresh[k].shape:
                if verbose:
                    print(f"shape mismatch for {k}: ckpt {state_dict[k].shape} vs model {fresh[k].shape} — using fresh init")
                state_dict[k] = fresh[k]

        # drop checkpoint keys that don't exist in current model (arch removals)
        for k in list(state_dict.keys()):
            if k not in fresh:
                if verbose:
                    print(f"dropping unknown key {k}")
                del state_dict[k]

        # fill any keys present in model but missing from checkpoint with fresh weights
        for k, v in fresh.items():
            if k not in state_dict:
                state_dict[k] = v

        model.load_state_dict(state_dict)

        optimizer = torch.optim.Adam(model.parameters())
        try:
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        except (ValueError, KeyError):
            pass  # architecture changed; fresh optimizer is fine for inference

    except FileNotFoundError:
        if not permissive:
            raise FileNotFoundError('There are no weights for this experiment...')
    return {'model': model,
            'epoch': model_dict['epoch'],
            'optimizer':optimizer,
            'controller_state_dict': model_dict['controller_state_dict']
            }

def dump_model_hparams(name, hparams):
    with open(f'models/{name}/hparams.json', 'w') as j:
        json.dump(hparams, j)
    pass

def model_from_json(params):
    model_params = dict(params['model'])
    model_type = model_params.pop('model_type', 'motifiesta')
    # backward compat: old checkpoints used parallel_matching bool
    if 'parallel_matching' in model_params:
        val = model_params.pop('parallel_matching')
        model_params.setdefault('matching_mode', 'luby' if val else 'greedy')
    if model_type in ('subgraph', 'disc'):
        from MotiFiesta.training.disc_model import MotiFiestaDisc
        model = MotiFiestaDisc(**model_params)
    else:
        from MotiFiesta.training.model import MotiFiestaModel
        model = MotiFiestaModel(**model_params)
    return model

def dataset_from_json(params, background=False):
    from MotiFiesta.training.loading import get_loader
    data = get_loader(root=params['train']['dataset'],\
                                batch_size=params['train']['batch_size']\
                                )
    return data

def make_dirs(run):
    try:
        os.mkdir(f"models/{run}")
    except FileExistsError:
        pass

def one_hot_to_id(x):
    """ Create column vector with index where one-hot is 1. """
    return torch.nonzero(x, as_tuple=True)[1]
